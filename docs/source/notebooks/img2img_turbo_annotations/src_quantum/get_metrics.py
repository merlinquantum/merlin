import os
import shutil
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from cleanfid.fid import build_feature_extractor, frechet_distance, get_folder_features
from cyclegan_turbo import CycleGAN_Turbo
from diffusers import AutoencoderKL, UNet2DConditionModel
from model import make_1step_sched, my_vae_decoder_fwd, my_vae_encoder_fwd
from my_utils.dino_struct import DinoStructureLoss
from my_utils.training_utils import (
    build_transform,
    parse_args_unpaired_training,
)
from peft import LoraConfig
from PIL import Image
from quantum_encoder import ParallelQuantumEncoder
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

EXP_PATH = "/home/jupyter-pemeriau/img2img-turbo/all_outputs/exp-162"


def load_model_qVAE(path, accelerator, quantum_training=False):
    print("---- Loading from pretrained weights for quantum training ----")
    print(f"---- WEIGHTS = {path} ----")
    sd = torch.load(path)
    print("-- weights loaded --")
    cyclegan_q = CycleGAN_Turbo(pretrained_path=path)
    print("---- CycleGAN defined ----")
    vae_enc = cyclegan_q.vae_enc
    vae_dec = cyclegan_q.vae_dec
    vae_a2b = cyclegan_q.vae
    vae_b2a = cyclegan_q.vae_b2a
    unet = cyclegan_q.unet
    print("-- weights in load model qVAE defined --")
    weight_dtype = torch.float32
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    if quantum_training:
        # convolution to merge the VAE_enc output with the quantum embeddings
        module_adaptation = nn.Conv2d(8, 4, kernel_size=1, stride=1)
        # normal initialization of the adaptation convolution
        module_adaptation.load_state_dict(sd["conv_ad"])
        module_adaptation.to(accelerator.device, dtype=weight_dtype)
        print("-- module adaptation loaded --")

        _params_gen = cyclegan_q.get_traininable_params_q(
            unet, vae_a2b, vae_b2a, module_adaptation
        )
        return cyclegan_q, vae_enc, vae_dec, unet, vae_a2b, vae_b2a, module_adaptation
    else:
        _params_gen = cyclegan_q.get_traininable_params(unet, vae_a2b, vae_b2a)

        return cyclegan_q, vae_enc, vae_dec, unet, vae_a2b, vae_b2a, 0


def load_model_dyn_UNet(path, accelerator):
    print("- building the model")
    # print("-- weights loaded --")
    cyclegan_d = CycleGAN_Turbo(pretrained_path=args.quantum_start_path)
    print("---- initial CycleGAN defined ----")
    vae_enc = cyclegan_d.vae_enc
    vae_dec = cyclegan_d.vae_dec
    vae_a2b = cyclegan_d.vae
    vae_b2a = cyclegan_d.vae_b2a
    unet = cyclegan_d.unet
    # freeze the VAE enc and detach it from the gradient
    vae_enc.requires_grad_(False)
    vae_a2b.post_quant_conv.requires_grad_(True)
    vae_b2a.post_quant_conv.requires_grad_(True)
    print("---- Loading from pretrained weights for quantum training ----")

    print(f"---- WEIGHTS = {path} ----")
    sd_f = torch.load(path)
    print("-- weights loaded --")
    cyclegan_d.load_ckpt_from_state_dict(sd_f, dyn=True)
    print("---- CycleGAN defined ----")

    weight_dtype = torch.float32
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    _params_gen = cyclegan_d.get_traininable_params(unet, vae_a2b, vae_b2a)

    return cyclegan_d, vae_enc, vae_dec, unet, vae_a2b, vae_b2a, 0


def initialize_unet(rank, return_lora_module_names=False):
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/sd-turbo", subfolder="unet"
    )
    unet.requires_grad_(False)
    unet.train()
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "conv",
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv_out",
        "proj_out",
        "proj_in",
        "ff.net.2",
        "ff.net.0.proj",
    ]

    for n, _p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight", ""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight", ""))
                break

    _lora_conf_encoder = LoraConfig(
        r=rank,
        init_lora_weights="gaussian",
        target_modules=l_target_modules_encoder,
        lora_alpha=rank,
    )
    _lora_conf_decoder = LoraConfig(
        r=rank,
        init_lora_weights="gaussian",
        target_modules=l_target_modules_decoder,
        lora_alpha=rank,
    )
    _lora_conf_others = LoraConfig(
        r=rank,
        init_lora_weights="gaussian",
        target_modules=l_modules_others,
        lora_alpha=rank,
    )
    # unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    # unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    # unet.add_adapter(lora_conf_others, adapter_name="default_others")
    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
    if return_lora_module_names:
        return (
            unet,
            l_target_modules_encoder,
            l_target_modules_decoder,
            l_modules_others,
        )
    else:
        return unet


def initialize_vae(rank=4, return_lora_module_names=False, dynamic=False):
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-turbo", subfolder="vae"
    )  # ,adapter_name = "vae_skip_new"

    vae.requires_grad_(False)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()
    # add the skip connection convs
    vae.decoder.skip_conv_1 = (
        torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        .cuda()
        .requires_grad_(True)
    )
    vae.decoder.skip_conv_2 = (
        torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        .cuda()
        .requires_grad_(True)
    )
    vae.decoder.skip_conv_3 = (
        torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        .cuda()
        .requires_grad_(True)
    )
    vae.decoder.skip_conv_4 = (
        torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        .cuda()
        .requires_grad_(True)
    )
    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    l_vae_target_modules = [
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv",
        "conv_out",
        "skip_conv_1",
        "skip_conv_2",
        "skip_conv_3",
        "skip_conv_4",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ]

    ##################################
    if dynamic:
        # conv_in + conv_out fully trained
        l_vae_target_modules = [
            "conv1",
            "conv2",
            "conv_shortcut",
            "conv",
            "skip_conv_1",
            "skip_conv_2",
            "skip_conv_3",
            "skip_conv_4",
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ]
    vae.encoder.conv_in.requires_grad_(True)
    vae.decoder.conv_out.requires_grad_(True)
    ##################################

    _vae_lora_config = LoraConfig(
        r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules
    )
    # vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae


args = parse_args_unpaired_training()


def main(
    args,
    EXP_PATH="/home/jupyter-pemeriau/img2img-turbo/all_outputs/exp-162",
    dataset_folder="/mnt/bmw-challenge-volume/home/jupyter-pemeriau/data/dataset_full_scale",
    path="/home/jupyter-pemeriau/img2img-turbo/all_outputs/exp-162/checkpoints/model_5001.pkl",
    quantum_dynamic=False,
    seed=42,
):
    print(f"--- Working on path {path}")
    print(f"in {EXP_PATH}")
    weight_dtype = torch.float32
    global_step = os.path.basename(path)[:-4]
    args.quantum_dynamic = quantum_dynamic
    accelerator = Accelerator(gradient_accumulation_steps=1)
    args.seed = seed
    set_seed(args.seed)

    cyclegan_d, vae_enc, vae_dec, unet, vae_a2b, vae_b2a, module_adaptation = (
        load_model_dyn_UNet(path, accelerator)
    )
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    vae_b2a.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    if args.quantum_dynamic:
        dims = (4, 16, 16)
        num_process = 10
        print(
            f"-- Defining the quantum encoder with dims = {dims} and num_processes = {num_process}"
        )
        # Set the random seeds + use same one as in the experiment #verified
        torch.manual_seed(args.seed)
        pqencoder = ParallelQuantumEncoder(dims, num_processes=num_process)
        print("-- Quantum encoder defined --")

    print("-- Defining eval encoder --")
    eval_unet = accelerator.unwrap_model(unet)
    eval_vae_enc = accelerator.unwrap_model(vae_enc)
    eval_vae_dec = accelerator.unwrap_model(vae_dec)

    # text embeddings
    tokenizer = AutoTokenizer.from_pretrained(
        "stabilityai/sd-turbo",
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    noise_scheduler_1step = make_1step_sched()
    text_encoder = CLIPTextModel.from_pretrained(
        "stabilityai/sd-turbo", subfolder="text_encoder"
    ).cuda()
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    fixed_caption_src = "driving in the night"
    fixed_caption_tgt = "driving in the day"
    print("-- Defining the text embeddings --")
    fixed_a2b_tokens = tokenizer(
        fixed_caption_tgt,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids[0]
    fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.cuda().unsqueeze(0))[0].detach()
    fixed_b2a_tokens = tokenizer(
        fixed_caption_src,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids[0]
    fixed_b2a_emb_base = text_encoder(fixed_b2a_tokens.cuda().unsqueeze(0))[0].detach()
    del text_encoder, tokenizer  # free up some memory
    fixed_a2b_emb = fixed_a2b_emb_base.repeat(1, 1, 1).to(dtype=weight_dtype)
    fixed_b2a_emb = fixed_b2a_emb_base.repeat(1, 1, 1).to(dtype=weight_dtype)

    l_images_src_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_src_test.extend(glob(os.path.join(dataset_folder, "test_A", ext)))
    l_images_tgt_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_tgt_test.extend(glob(os.path.join(dataset_folder, "test_B", ext)))
    l_images_src_test, l_images_tgt_test = (
        sorted(l_images_src_test),
        sorted(l_images_tgt_test),
    )
    # l_images_src_test, l_images_tgt_test = l_images_src_test[:10], l_images_tgt_test[:10]
    print(
        f"-- FOR TEST PURPOSE: working with {len(l_images_src_test)} and {len(l_images_tgt_test)}"
    )
    T_val = build_transform(args.val_img_prep)
    # make the reference FID statistics
    _timesteps = torch.tensor(
        [noise_scheduler_1step.config.num_train_timesteps - 1] * 1, device="cuda"
    ).long()
    print("-- make the reference FID statistics --")
    if accelerator.is_main_process:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
        """
        FID reference statistics for A -> B translation
        """
        output_dir_ref = os.path.join(EXP_PATH, "fid_reference_a2b_o")

        if not os.path.isdir(output_dir_ref):
            os.makedirs(output_dir_ref, exist_ok=True)
            for _path in tqdm(l_images_tgt_test):
                _img = T_val(Image.open(_path).convert("RGB"))
                outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(
                    ".jpg", ".png"
                )
                if not os.path.exists(outf):
                    _img.save(outf)
        # compute the features for the reference images
        ref_features = get_folder_features(
            output_dir_ref,
            model=feat_model,
            num_workers=0,
            num=None,
            shuffle=False,
            seed=0,
            batch_size=8,
            device=torch.device("cuda"),
            mode="clean",
            custom_fn_resize=None,
            description="",
            verbose=True,
            custom_image_tranform=None,
        )
        a2b_ref_mu, a2b_ref_sigma = (
            np.mean(ref_features, axis=0),
            np.cov(ref_features, rowvar=False),
        )
        """
        FID reference statistics for B -> A translation
        """
        # transform all images according to the validation transform and save them
        output_dir_ref = os.path.join(EXP_PATH, "fid_reference_b2a_o")

        if not os.path.isdir(output_dir_ref):
            os.makedirs(output_dir_ref, exist_ok=True)
            for _path in tqdm(l_images_src_test):
                _img = T_val(Image.open(_path).convert("RGB"))
                outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(
                    ".jpg", ".png"
                )
                if not os.path.exists(outf):
                    _img.save(outf)
        # compute the features for the reference images
        ref_features = get_folder_features(
            output_dir_ref,
            model=feat_model,
            num_workers=0,
            num=None,
            shuffle=False,
            seed=0,
            batch_size=8,
            device=torch.device("cuda"),
            mode="clean",
            custom_fn_resize=None,
            description="",
            verbose=True,
            custom_image_tranform=None,
        )
        b2a_ref_mu, b2a_ref_sigma = (
            np.mean(ref_features, axis=0),
            np.cov(ref_features, rowvar=False),
        )
    print("-- references done --")

    fid_output_dir = os.path.join(EXP_PATH, f"fid-all-{global_step}/samples_a2b")
    os.makedirs(fid_output_dir, exist_ok=True)
    l_dino_scores_a2b = []
    net_dino = DinoStructureLoss()
    # get val input images from domain a
    for idx, input_img_path in enumerate(tqdm(l_images_src_test)):
        if idx > args.validation_num_images and args.validation_num_images > 0:
            break

        outf = os.path.join(fid_output_dir, f"{idx}.png")

        with torch.no_grad():
            input_img = T_val(Image.open(input_img_path).convert("RGB"))
            img_a = transforms.ToTensor()(input_img)
            img_a = transforms.Normalize([0.5], [0.5])(img_a).unsqueeze(0).cuda()
            # src_name = os.path.basename(input_img_path)

            if args.quantum_dynamic:
                eval_fake_b, q_emb_a_val = cyclegan_d.forward_with_networks_dynamic(
                    img_a,
                    "a2b",
                    eval_vae_enc,
                    eval_unet,
                    eval_vae_dec,
                    noise_scheduler_1step,
                    _timesteps,
                    fixed_a2b_emb[0:1],
                    qencoder=pqencoder,
                )
            else:
                eval_fake_b = cyclegan_d.forward_with_networks(
                    img_a,
                    "a2b",
                    eval_vae_enc,
                    eval_unet,
                    eval_vae_dec,
                    noise_scheduler_1step,
                    _timesteps,
                    fixed_a2b_emb[0:1],
                )

            eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)

            eval_fake_b_pil.save(outf)
            a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
            b = net_dino.preprocess(eval_fake_b_pil).unsqueeze(0).cuda()
            dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
            print(f"DINO a2b = {dino_ssim}")
            l_dino_scores_a2b.append(dino_ssim)
    dino_score_a2b = np.mean(l_dino_scores_a2b)
    gen_features = get_folder_features(
        fid_output_dir,
        model=feat_model,
        num_workers=0,
        num=None,
        shuffle=False,
        seed=0,
        batch_size=8,
        device=torch.device("cuda"),
        mode="clean",
        custom_fn_resize=None,
        description="",
        verbose=True,
        custom_image_tranform=None,
    )
    ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
    score_fid_a2b = frechet_distance(a2b_ref_mu, a2b_ref_sigma, ed_mu, ed_sigma)
    print(
        f"step={global_step}, fid(a2b)={score_fid_a2b:.2f}, dino(a2b)={dino_score_a2b:.3f}"
    )
    # remove folder
    shutil.rmtree(fid_output_dir)
    print("-- Folder for a2b has been removed --")
    """
    compute FID for "B->A"
    """
    fid_output_dir = os.path.join(EXP_PATH, f"fid-all-{global_step}/samples_b2a")
    os.makedirs(fid_output_dir, exist_ok=True)
    l_dino_scores_b2a = []
    # get val input images from domain b
    for idx, input_img_path in enumerate(tqdm(l_images_tgt_test)):
        if idx > args.validation_num_images and args.validation_num_images > 0:
            break
        outf = os.path.join(fid_output_dir, f"{idx}.png")
        with torch.no_grad():
            input_img = T_val(Image.open(input_img_path).convert("RGB"))
            img_b = transforms.ToTensor()(input_img)
            img_b = transforms.Normalize([0.5], [0.5])(img_b).unsqueeze(0).cuda()
            # src_name = os.path.basename(input_img_path)
            if args.quantum_dynamic:
                eval_fake_a, q_emb_b_val = cyclegan_d.forward_with_networks_dynamic(
                    img_b,
                    "b2a",
                    eval_vae_enc,
                    eval_unet,
                    eval_vae_dec,
                    noise_scheduler_1step,
                    _timesteps,
                    fixed_b2a_emb[0:1],
                    qencoder=pqencoder,
                )
            else:
                eval_fake_a = cyclegan_d.forward_with_networks(
                    img_b,
                    "b2a",
                    eval_vae_enc,
                    eval_unet,
                    eval_vae_dec,
                    noise_scheduler_1step,
                    _timesteps,
                    fixed_b2a_emb[0:1],
                )

            eval_fake_a_pil = transforms.ToPILImage()(eval_fake_a[0] * 0.5 + 0.5)

            eval_fake_a_pil.save(outf)
            a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
            b = net_dino.preprocess(eval_fake_a_pil).unsqueeze(0).cuda()
            dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
            print(f"DINO b2a = {dino_ssim}")
            l_dino_scores_b2a.append(dino_ssim)
    dino_score_b2a = np.mean(l_dino_scores_b2a)
    gen_features = get_folder_features(
        fid_output_dir,
        model=feat_model,
        num_workers=0,
        num=None,
        shuffle=False,
        seed=0,
        batch_size=8,
        device=torch.device("cuda"),
        mode="clean",
        custom_fn_resize=None,
        description="",
        verbose=True,
        custom_image_tranform=None,
    )
    ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
    score_fid_b2a = frechet_distance(b2a_ref_mu, b2a_ref_sigma, ed_mu, ed_sigma)
    print(
        f"step={global_step}, fid(b2a)={score_fid_b2a}, dino(b2a)={dino_score_b2a:.3f}"
    )
    # remove folder
    shutil.rmtree(fid_output_dir)
    print("-- Folder for b2a has been removed --")

    df = pd.DataFrame([
        {
            "fid(a2b)": score_fid_a2b,
            "fid(b2a)": score_fid_b2a,
            "dino(a2b)": dino_score_a2b,
            "dino(b2a)": dino_score_b2a,
        }
    ])
    df.to_csv(
        os.path.join(EXP_PATH, f"fid-all-{global_step}", "metrics.csv"), index=False
    )
    print("-- embeddings saved to csv --")


dico_ckpt = {
    "q84-1-251": [
        "/training-models/all_outputs/exp-10/checkpoints/model_251.pkl",
        "/training-models/all_outputs/exp-10",
        True,
        42,
    ],
    "q84-1-501": [
        "/training-models/all_outputs/exp-10/checkpoints/model_501.pkl",
        "/training-models/all_outputs/exp-10",
        True,
        42,
    ],
    "q84-1-751": [
        "/training-models/all_outputs/exp-10/checkpoints/model_751.pkl",
        "/training-models/all_outputs/exp-10",
        True,
        42,
    ],
    "q84-1-1001": [
        "/training-models/all_outputs/exp-10/checkpoints/model_1001.pkl",
        "/training-models/all_outputs/exp-10",
        True,
        42,
    ],
    "q84-1-1251": [
        "/training-models/all_outputs/exp-10/checkpoints/model_1251.pkl",
        "/training-models/all_outputs/exp-10",
        True,
        42,
    ],
    "q84-1-1500": [
        "/training-models/all_outputs/exp-10/checkpoints/model_1500.pkl",
        "/training-models/all_outputs/exp-10",
        True,
        42,
    ],
    "cl84-1-251": [
        "/training-models/all_outputs/exp-13/checkpoints/model_251.pkl",
        "/training-models/all_outputs/exp-13",
        False,
        42,
    ],
    "cl84-1-501": [
        "/training-models/all_outputs/exp-13/checkpoints/model_501.pkl",
        "/training-models/all_outputs/exp-13",
        False,
        42,
    ],
    "cl84-1-751": [
        "/training-models/all_outputs/exp-13/checkpoints/model_751.pkl",
        "/training-models/all_outputs/exp-13",
        False,
        42,
    ],
    "cl84-1-1001": [
        "/training-models/all_outputs/exp-13/checkpoints/model_1001.pkl",
        "/training-models/all_outputs/exp-13",
        False,
        42,
    ],
    "cl84-1-1251": [
        "/training-models/all_outputs/exp-13/checkpoints/model_1251.pkl",
        "/training-models/all_outputs/exp-13",
        False,
        42,
    ],
    "cl84-1-1500": [
        "/training-models/all_outputs/exp-13/checkpoints/model_1500.pkl",
        "/training-models/all_outputs/exp-13",
        False,
        42,
    ],
}

for model in dico_ckpt.keys():
    model_path, exp_path, q_dyn, seed = (
        dico_ckpt[model][0],
        dico_ckpt[model][1],
        dico_ckpt[model][2],
        dico_ckpt[model][3],
    )
    main(args, EXP_PATH=exp_path, path=model_path, quantum_dynamic=q_dyn, seed=seed)
