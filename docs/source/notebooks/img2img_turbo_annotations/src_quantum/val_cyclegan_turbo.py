import os
import re
from glob import glob

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from cleanfid.fid import build_feature_extractor, frechet_distance, get_folder_features
from cyclegan_turbo import (
    CycleGAN_Turbo,
)
from model import make_1step_sched
from my_utils.dino_struct import DinoStructureLoss
from my_utils.training_utils import (
    build_transform,
)
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

# TODO: not up to date

noise_scheduler_1step = make_1step_sched()
accelerator = Accelerator(gradient_accumulation_steps=1)

################
# PARAMETERS ##
################

OUTPUT_DIR = "/home/jupyter-pemeriau/img2img-turbo/all_outputs/exp-114/test"
validation_num_images = -1
dataset_folder = "/home/jupyter-pemeriau/data/dataset_full_scale"
q_emb_path = "/home/jupyter-pemeriau/q_embs/all_emb_128_dims_16_16_ckpt1001_test"
feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
T_val = build_transform("resize_128")

################
# REFERENCES ##
################

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
# l_images_src_test, l_images_tgt_test = l_images_src_test[:100], l_images_tgt_test[:100]
print(
    f"-- FOR TEST PURPOSE: working with {len(l_images_src_test)} and {len(l_images_tgt_test)}"
)

# make the reference FID statistics
print("-- make the reference FID statistics --")
if accelerator.is_main_process:
    feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
    """
    FID reference statistics for A -> B translation
    """
    output_dir_ref = os.path.join(OUTPUT_DIR, "fid_reference_a2b")
    os.makedirs(output_dir_ref, exist_ok=True)
    # transform all images according to the validation transform and save them
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
    output_dir_ref = os.path.join(OUTPUT_DIR, "fid_reference_b2a")
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

###############
# TEXT EMBS ##
###############
print("-- Defining the text embeddings --")
tokenizer = AutoTokenizer.from_pretrained(
    "stabilityai/sd-turbo",
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
text_encoder = CLIPTextModel.from_pretrained(
    "stabilityai/sd-turbo", subfolder="text_encoder"
).cuda()

fixed_caption_tgt = "driving in the day"
fixed_caption_src = "driving in the night"

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


################
# LOAD MODEL ##
################
def create_model_from(quantum_start_path):
    sd = torch.load(quantum_start_path)
    cyclegan_q = CycleGAN_Turbo(pretrained_path=model_path).to("cuda:0")

    vae_enc = cyclegan_q.vae_enc
    vae_dec = cyclegan_q.vae_dec
    vae_a2b = cyclegan_q.vae
    # vae_b2a = cyclegan_q.vae_b2a
    unet = cyclegan_q.unet

    weight_dtype = torch.float32
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # convolution to merge the VAE_enc output with the quantum embeddings
    module_adaptation = nn.Conv2d(8, 4, kernel_size=1, stride=1)
    module_adaptation.load_state_dict(sd["conv_ad"])
    module_adaptation.to(accelerator.device, dtype=weight_dtype)
    print("--- Model loaded ---")
    return cyclegan_q, unet, vae_enc, vae_dec, module_adaptation


def append_to_csv(file_name, data):
    # Convert dictionary to a DataFrame (1 row)
    df = pd.DataFrame([data])  # Convert dictionary to a one-row DataFrame

    # Append to the CSV file
    df.to_csv(file_name, mode="a", header=not os.path.isfile(file_name), index=False)


csv_name = os.path.join(OUTPUT_DIR, "metrics.csv")
# with open(csv_name, mode='w', newline='') as file:
#     pass  # Do nothing, just create the file
################
# GET SCORES ##
################
all_paths_files = os.listdir(
    "/home/jupyter-pemeriau/img2img-turbo/all_outputs/exp-114/checkpoints"
)
all_paths = [
    os.path.join(
        "/home/jupyter-pemeriau/img2img-turbo/all_outputs/exp-114/checkpoints", path
    )
    for path in all_paths_files
]
for model_path in all_paths:
    print(f"For model path = {model_path}")
    names = re.search(r"model_(\d+)\.pkl", os.path.basename(model_path))
    global_step = int(names.group(1))
    print(f"Step {global_step}")
    weight_dtype = torch.float32

    # load model

    cyclegan_q, unet, vae_enc, vae_dec, module_adaptation = create_model_from(
        model_path
    )
    eval_unet = accelerator.unwrap_model(unet)
    eval_vae_enc = accelerator.unwrap_model(vae_enc)
    eval_vae_dec = accelerator.unwrap_model(vae_dec)
    eval_module = accelerator.unwrap_model(module_adaptation)

    _timesteps = torch.tensor(
        [noise_scheduler_1step.config.num_train_timesteps - 1] * 1, device="cuda"
    ).long()
    net_dino = DinoStructureLoss()
    """
    Evaluate "A->B"
    """
    fid_output_dir = os.path.join(OUTPUT_DIR, f"fid-val-{global_step}/samples_a2b")
    os.makedirs(fid_output_dir, exist_ok=True)
    l_dino_scores_a2b = []
    # get val input images from domain a
    for idx, input_img_path in enumerate(tqdm(l_images_src_test)):
        if idx > validation_num_images and validation_num_images > 0:
            break
        outf = os.path.join(fid_output_dir, f"{idx}.png")
        with torch.no_grad():
            input_img = T_val(Image.open(input_img_path).convert("RGB"))
            img_a = transforms.ToTensor()(input_img)
            img_a = transforms.Normalize([0.5], [0.5])(img_a).unsqueeze(0).cuda()
            bsz = img_a.shape[0]
            fixed_a2b_emb = (
                fixed_a2b_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype).cuda()
            )
            # embeddings in test_A
            src_name = os.path.basename(input_img_path)
            with h5py.File(os.path.join(q_emb_path, "test_A.h5"), "r") as f:
                if src_name in f:
                    qt_t_src = f[src_name]
                    q_emb_a = torch.tensor(qt_t_src)
                else:
                    print(f"{src_name} not found in q_embs_A")
            q_emb_a = q_emb_a.unsqueeze(0).cuda()

            eval_fake_b = cyclegan_q.forward_with_networks_q(
                img_a,
                "a2b",
                eval_vae_enc,
                eval_unet,
                eval_vae_dec,
                noise_scheduler_1step,
                _timesteps,
                fixed_a2b_emb[0:1],
                q_emb_a,
                eval_module,
            )

            eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
            # only save images for which idx is 0-100
            if idx < 100:
                eval_fake_b_pil.save(outf)
            a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
            b = net_dino.preprocess(eval_fake_b_pil).unsqueeze(0).cuda()
            dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
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

    """
    compute FID for "B->A"
    """
    fid_output_dir = os.path.join(OUTPUT_DIR, f"fid-val-{global_step}/samples_b2a")
    os.makedirs(fid_output_dir, exist_ok=True)
    l_dino_scores_b2a = []
    # get val input images from domain b
    for idx, input_img_path in enumerate(tqdm(l_images_tgt_test)):
        if idx > validation_num_images and validation_num_images > 0:
            break
        outf = os.path.join(fid_output_dir, f"{idx}.png")
        with torch.no_grad():
            input_img = T_val(Image.open(input_img_path).convert("RGB"))
            img_b = transforms.ToTensor()(input_img)
            img_b = transforms.Normalize([0.5], [0.5])(img_b).unsqueeze(0).cuda()
            bsz = img_b.shape[0]
            fixed_b2a_emb = (
                fixed_b2a_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype).cuda()
            )
            # quantum embeddings
            src_name = os.path.basename(input_img_path)
            with h5py.File(os.path.join(q_emb_path, "test_B.h5"), "r") as f:
                if src_name in f:
                    qt_t_src = f[src_name]
                    q_emb_b = torch.tensor(qt_t_src)
                else:
                    print(f"{src_name} not found in q_embs_B")
            q_emb_b = q_emb_b.unsqueeze(0).cuda()

            eval_fake_a = cyclegan_q.forward_with_networks_q(
                img_b,
                "b2a",
                eval_vae_enc,
                eval_unet,
                eval_vae_dec,
                noise_scheduler_1step,
                _timesteps,
                fixed_b2a_emb[0:1],
                q_emb_b,
                eval_module,
            )

            eval_fake_a_pil = transforms.ToPILImage()(eval_fake_a[0] * 0.5 + 0.5)

            # only save images for which idx is 0-100
            if idx < 100:
                eval_fake_a_pil.save(outf)
            a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
            b = net_dino.preprocess(eval_fake_a_pil).unsqueeze(0).cuda()
            dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
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

    metrics_dict = {
        "Step": global_step,
        "fid(a2b)": score_fid_a2b,
        "fid(b2a)": score_fid_b2a,
        "dino(a2b)": dino_score_a2b,
        "dino(b2a)": dino_score_b2a,
    }
    append_to_csv(csv_name, metrics_dict)
