import copy
import gc
import os
import shutil
import time
from glob import glob

import lpips
import numpy as np
import torch
import vision_aided_loss
from accelerate import Accelerator
from accelerate.utils import set_seed
from BosonSampler import BosonSampler
from cleanfid.fid import build_feature_extractor, frechet_distance, get_folder_features
from cyclegan_turbo import (
    CycleGAN_Turbo,
    VAE_decode,
    VAE_encode,
    initialize_unet,
    initialize_vae,
)
from diffusers.optimization import get_scheduler
from model import make_1step_sched
from my_utils.dino_struct import DinoStructureLoss
from my_utils.training_utils import (
    UnpairedDataset,
    build_transform,
    get_next_id,
    load_small_dataset,
    parse_args_unpaired_training,
)
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel


def main(args):
    set_seed(args.seed)
    if args.quantum_dynamic:
        print("--- qCycleGAN training ---")
        print(
            f"- Defining the quantum encoder with dims = {args.quantum_dims} and num_processes = {args.quantum_processes}"
        )
        # Set the random seeds
        torch.manual_seed(args.seed)
        print("boson sampler dims", (args.quantum_processes,) + args.quantum_dims)
        boson_sampler = BosonSampler((args.quantum_processes,) + args.quantum_dims)
        print("-- Boson Sampler defined --")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # set up id and name of experiment
    exp_id = get_next_id()
    if args.output_dir == "a":
        args.output_dir = os.path.join(
            "/training-models/all_outputs", f"exp-{str(exp_id)}"
        )
    args.exp_id = exp_id
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

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

    weight_dtype = torch.float32

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)

    if args.gan_disc_type == "vagan_clip":
        net_disc_a = vision_aided_loss.Discriminator(
            cv_type="clip", loss_type=args.gan_loss_type, device="cuda"
        )
        net_disc_a.cv_ensemble.requires_grad_(False)  # Freeze feature extractor
        net_disc_b = vision_aided_loss.Discriminator(
            cv_type="clip", loss_type=args.gan_loss_type, device="cuda"
        )
        net_disc_b.cv_ensemble.requires_grad_(False)  # Freeze feature extractor

    crit_cycle, crit_idt = torch.nn.L1Loss(), torch.nn.L1Loss()

    # LOAD PRETRAINED WEIGHTS #####
    if args.quantum_dynamic or args.cl_comp:
        print("- building the model")
        unet, l_modules_unet_encoder, l_modules_unet_decoder, l_modules_unet_others = (
            initialize_unet(args.lora_rank_unet, return_lora_module_names=True)
        )
        vae_a2b, vae_lora_target_modules = initialize_vae(
            args.lora_rank_vae,
            return_lora_module_names=True,
            dynamic=args.quantum_dynamic,
        )
        print("-- Loading from pretrained weights for quantum training")
        print(f"--- WEIGHTS from {args.quantum_start_path}")
        print(f"Defining cyclegan_d on process {accelerator.process_index}")
        cyclegan_d = CycleGAN_Turbo(
            accelerator=accelerator, pretrained_path=args.quantum_start_path
        )  # pretrained_path=args.quantum_start_path,
        print("--- CycleGAN defined")
        vae_enc = cyclegan_d.vae_enc
        vae_dec = cyclegan_d.vae_dec
        vae_a2b = cyclegan_d.vae
        vae_b2a = cyclegan_d.vae_b2a
        unet = cyclegan_d.unet

        unet.conv_in.requires_grad_(True)

        weight_dtype = torch.float32
        """vae_a2b.to(accelerator.device, dtype=weight_dtype)
        unet.to(accelerator.device, dtype=weight_dtype)
        vae_b2a = copy.deepcopy(vae_a2b)"""

        # freeze the VAE enc and detach it from the gradient
        vae_enc.requires_grad_(False)

        # unet.requires_grad_(True)
        # print("---- UNet is fully trained ----")

        # train the post_quant_conv layer at the
        vae_a2b.post_quant_conv.requires_grad_(True)
        vae_b2a.post_quant_conv.requires_grad_(True)

        # get the trainable parameters (vae_a2b.encoder and vae_b2a.encoder should have 0 trainable parameters)
        params_gen = cyclegan_d.get_traininable_params(
            unet, vae_a2b, vae_b2a, boson_sampler, dynamic=True, quantum_training=True
        )
        print(
            f"-- {vae_a2b.encoder.conv_in.__class__.__name__}: requires_grad={vae_a2b.encoder.conv_in.weight.requires_grad_}"
        )
        print(
            f"-- For vae_enc = {sum(p.numel() for p in vae_enc.parameters() if p.requires_grad)}, - a2b = {sum(p.numel() for p in vae_a2b.encoder.parameters() if p.requires_grad)} and -b2a = {sum(p.numel() for p in vae_b2a.encoder.parameters() if p.requires_grad)}"
        )
        print(
            f"-- For unet = {sum(p.numel() for p in unet.parameters() if p.requires_grad)}"
        )
        print(
            f"-- For unet.conv_in = {sum(p.numel() for p in unet.conv_in.parameters() if p.requires_grad)} - unet.conv_out = {sum(p.numel() for p in unet.conv_out.parameters() if p.requires_grad)} "
        )

    # CLASSICAL NETWORKS with no initialization
    else:
        print("--- Classical CycleGAN training ---")
        print("- building the model")
        unet, l_modules_unet_encoder, l_modules_unet_decoder, l_modules_unet_others = (
            initialize_unet(args.lora_rank_unet, return_lora_module_names=True)
        )
        vae_a2b, vae_lora_target_modules = initialize_vae(
            args.lora_rank_vae,
            return_lora_module_names=True,
            dynamic=args.quantum_dynamic,
        )
        vae_a2b.to(accelerator.device, dtype=weight_dtype)
        unet.to(accelerator.device, dtype=weight_dtype)
        unet.conv_in.requires_grad_(True)
        vae_b2a = copy.deepcopy(vae_a2b)
        params_gen = CycleGAN_Turbo.get_traininable_params(
            unet, vae_a2b, vae_b2a, boson_sampler
        )
        vae_enc = VAE_encode(vae_a2b, vae_b2a=vae_b2a)
        vae_dec = VAE_decode(vae_a2b, vae_b2a=vae_b2a)

        print(
            f"-- {vae_a2b.encoder.conv_in.__class__.__name__}: requires_grad={vae_a2b.encoder.conv_in.weight.requires_grad_}"
        )
        print(
            f"-- For vae_enc = {sum(p.numel() for p in vae_enc.parameters() if p.requires_grad)}, - a2b = {sum(p.numel() for p in vae_a2b.encoder.parameters() if p.requires_grad)} and -b2a = {sum(p.numel() for p in vae_b2a.encoder.parameters() if p.requires_grad)}"
        )
        print(
            f"-- For unet = {sum(p.numel() for p in unet.parameters() if p.requires_grad)}"
        )
        print(
            f"-- For unet.conv_in = {sum(p.numel() for p in unet.conv_in.parameters() if p.requires_grad)} - unet.conv_out = {sum(p.numel() for p in unet.conv_out.parameters() if p.requires_grad)} "
        )
    print(
        f"-- TOTAL parameters = {sum(p.numel() for p in unet.parameters()) + sum(p.numel() for p in vae_a2b.parameters()) + sum(p.numel() for p in vae_b2a.parameters())}"
    )
    print(
        f"-- TOTAL trainable parameters = {sum(p.numel() for p in unet.parameters() if p.requires_grad) + sum(p.numel() for p in vae_a2b.parameters() if p.requires_grad) + sum(p.numel() for p in vae_b2a.parameters() if p.requires_grad)}"
    )
    if args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # TEST
    # print(f"------------------------ After loading the weights, VAE conv in = {vae_enc.vae.encoder.conv_in.lora_A.vae_skip.weight}")
    optimizer_gen = torch.optim.AdamW(
        params_gen,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    params_disc = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
    optimizer_disc = torch.optim.AdamW(
        params_disc,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    _ = load_small_dataset(
        "quandelagl/small-bdd-100k", "/kaggle/img2img_turbo_annotations/dataset"
    )
    print(args.dataset_folder)
    dataset_train = UnpairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.train_img_prep,
        split="train",
        tokenizer=tokenizer,
        part=args.training_images,
    )
    # here, we could lower the number of images used
    print("- Dataset loaded")

    print(f"--> Length of dataset_train = {len(dataset_train)}")
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    print("- Data loaded")
    T_val = build_transform(args.val_img_prep)
    fixed_caption_src = dataset_train.fixed_caption_src
    fixed_caption_tgt = dataset_train.fixed_caption_tgt
    l_images_src_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_src_test.extend(glob(os.path.join(args.dataset_folder, "test_a", ext)))
    l_images_src_test_2 = [
        el for el in l_images_src_test if os.path.basename(el)
    ]  # not in list_failures]
    l_images_src_test = l_images_src_test_2

    l_images_tgt_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_tgt_test.extend(glob(os.path.join(args.dataset_folder, "test_b", ext)))

    l_images_src_test, l_images_tgt_test = (
        sorted(l_images_src_test),
        sorted(l_images_tgt_test),
    )
    # l_images_src_test, l_images_tgt_test = l_images_src_test[:100], l_images_tgt_test[:100]
    print(
        f"- FOR TEST PURPOSE: working with {len(l_images_src_test)} and {len(l_images_tgt_test)}"
    )
    print(unet.conv_in)
    # make the reference FID statistics
    print("-- make the reference FID statistics")
    if accelerator.is_main_process:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
        """
        FID reference statistics for A -> B translation
        """
        output_dir_ref = os.path.join(args.output_dir, "fid_reference_a2b")
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
        output_dir_ref = os.path.join(args.output_dir, "fid_reference_b2a")
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

    print("- Defining the scheduler for generators + discriminator")
    lr_scheduler_gen = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_gen,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    net_lpips = lpips.LPIPS(net="vgg")
    net_lpips.cuda()
    net_lpips.requires_grad_(False)

    print("- Defining the text embeddings")
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

    unet, vae_enc, vae_dec, net_disc_a, net_disc_b = accelerator.prepare(
        unet, vae_enc, vae_dec, net_disc_a, net_disc_b
    )
    (
        net_lpips,
        optimizer_gen,
        optimizer_disc,
        train_dataloader,
        lr_scheduler_gen,
        lr_scheduler_disc,
    ) = accelerator.prepare(
        net_lpips,
        optimizer_gen,
        optimizer_disc,
        train_dataloader,
        lr_scheduler_gen,
        lr_scheduler_disc,
    )
    if accelerator.is_main_process:
        config = dict(vars(args))
        config["run_name"] = f"qCycleGAN-{args.exp_id}"
    # checking if the VAE_encoder is frozen on the pretrained weights
    print(
        f"- For vae_enc = {sum(p.numel() for p in vae_enc.parameters() if p.requires_grad)}, - a2b = {sum(p.numel() for p in vae_a2b.encoder.parameters() if p.requires_grad)} and -b2a = {sum(p.numel() for p in vae_b2a.encoder.parameters() if p.requires_grad)}"
    )
    first_epoch = 0
    global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    # turn off eff. attn for the disc
    for name, module in net_disc_a.named_modules():
        if "attn" in name:
            module.fused_attn = False
    for name, module in net_disc_b.named_modules():
        if "attn" in name:
            module.fused_attn = False

    for _epoch in range(first_epoch, args.max_train_epochs):
        # print(f"-- adapt weights = {module_adaptation.weight} = adapt weights --")
        # print(f"-- UNET weights = {unet.conv_in.weight} = UNET weights --")
        for _step, batch in enumerate(train_dataloader):
            # print(f"In batch: image = {batch['pixel_values_src'].shape}")
            t_start = time.time()
            # if step > 10:
            #     breakRuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (
            l_acc = [unet, net_disc_a, net_disc_b, vae_enc, vae_dec]
            with accelerator.accumulate(*l_acc):
                img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)

                bsz = img_a.shape[0]
                fixed_a2b_emb = fixed_a2b_emb_base.repeat(bsz, 1, 1).to(
                    dtype=weight_dtype
                )
                fixed_b2a_emb = fixed_b2a_emb_base.repeat(bsz, 1, 1).to(
                    dtype=weight_dtype
                )
                timesteps = torch.tensor(
                    [noise_scheduler_1step.config.num_train_timesteps - 1] * bsz,
                    device=img_a.device,
                ).long()

                """
                Cycle Objective
                """
                # A -> fake B -> rec A
                if args.quantum_dynamic:
                    boson_sampler.unitaries = None
                    # print("Real A -> Fake B")
                    cyc_fake_b, q_emb_a = cyclegan_d.forward_with_networks_dynamic(
                        img_a,
                        "a2b",
                        vae_enc,
                        unet,
                        vae_dec,
                        noise_scheduler_1step,
                        timesteps,
                        fixed_a2b_emb,
                        bs=boson_sampler,
                        device=accelerator.device,
                        accelerator=accelerator,
                    )
                    if torch.isnan(cyc_fake_b).any():
                        print("!!!!! JUMPING TO NEXT ITER during Real A -> Fake B!!!!!")
                        global_step += 1
                        continue
                    # print("Fake B -> Rec A")
                    cyc_rec_a, q_emb_cyc_fake_b = (
                        cyclegan_d.forward_with_networks_dynamic(
                            cyc_fake_b,
                            "b2a",
                            vae_enc,
                            unet,
                            vae_dec,
                            noise_scheduler_1step,
                            timesteps,
                            text_emb=fixed_b2a_emb,
                            bs=boson_sampler,
                            device=accelerator.device,
                            accelerator=accelerator,
                        )
                    )
                    if torch.isnan(cyc_rec_a).any():
                        print("!!!!! JUMPING TO NEXT ITER during Fake B -> Rec A!!!!!")
                        global_step += 1
                        continue
                else:
                    cyc_fake_b = CycleGAN_Turbo.forward_with_networks(
                        img_a,
                        "a2b",
                        vae_enc,
                        unet,
                        vae_dec,
                        noise_scheduler_1step,
                        timesteps,
                        fixed_a2b_emb,
                    )
                    cyc_rec_a = CycleGAN_Turbo.forward_with_networks(
                        cyc_fake_b,
                        "b2a",
                        vae_enc,
                        unet,
                        vae_dec,
                        noise_scheduler_1step,
                        timesteps,
                        fixed_b2a_emb,
                    )
                # print(f"--loss cycle a-- (Process {accelerator.process_index})")
                loss_cycle_a = crit_cycle(cyc_rec_a, img_a) * args.lambda_cycle

                loss_cycle_a += (
                    net_lpips(cyc_rec_a, img_a).mean() * args.lambda_cycle_lpips
                )
                # B -> fake A -> rec B
                if args.quantum_dynamic:
                    # print("img B -> fake A")
                    cyc_fake_a, q_emb_b = cyclegan_d.forward_with_networks_dynamic(
                        img_b,
                        "b2a",
                        vae_enc,
                        unet,
                        vae_dec,
                        noise_scheduler_1step,
                        timesteps,
                        fixed_b2a_emb,
                        bs=boson_sampler,
                        device=accelerator.device,
                        accelerator=accelerator,
                    )
                    if torch.isnan(cyc_fake_a).any():
                        print("!!!!! JUMPING TO NEXT ITER during img B -> fake A !!!!!")
                        global_step += 1
                        continue
                    # print("fake A -> rec B")
                    cyc_rec_b, q_emb_cyc_fake_a = (
                        cyclegan_d.forward_with_networks_dynamic(
                            cyc_fake_a,
                            "a2b",
                            vae_enc,
                            unet,
                            vae_dec,
                            noise_scheduler_1step,
                            timesteps,
                            text_emb=fixed_a2b_emb,
                            bs=boson_sampler,
                            device=accelerator.device,
                            accelerator=accelerator,
                        )
                    )
                    if torch.isnan(cyc_rec_b).any():
                        print("!!!!! JUMPING TO NEXT ITER during fake A -> rec B!!!!!")
                        global_step += 1
                        continue
                else:
                    cyc_fake_a = CycleGAN_Turbo.forward_with_networks(
                        img_b,
                        "b2a",
                        vae_enc,
                        unet,
                        vae_dec,
                        noise_scheduler_1step,
                        timesteps,
                        fixed_b2a_emb,
                    )
                    cyc_rec_b = CycleGAN_Turbo.forward_with_networks(
                        cyc_fake_a,
                        "a2b",
                        vae_enc,
                        unet,
                        vae_dec,
                        noise_scheduler_1step,
                        timesteps,
                        fixed_a2b_emb,
                    )
                loss_cycle_b = crit_cycle(cyc_rec_b, img_b) * args.lambda_cycle

                loss_cycle_b += (
                    net_lpips(cyc_rec_b, img_b).mean() * args.lambda_cycle_lpips
                )
                accelerator.backward(loss_cycle_a + loss_cycle_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)

                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Generator Objective (GAN) for task a->b and b->a (fake inputs)
                """

                if args.quantum_dynamic:
                    fake_a = cyclegan_d.forward_with_networks_dynamic(
                        img_b,
                        "b2a",
                        vae_enc,
                        unet,
                        vae_dec,
                        noise_scheduler_1step,
                        timesteps,
                        fixed_b2a_emb,
                        q_emb=q_emb_b,
                        device=accelerator.device,
                    )
                    fake_b = cyclegan_d.forward_with_networks_dynamic(
                        img_a,
                        "a2b",
                        vae_enc,
                        unet,
                        vae_dec,
                        noise_scheduler_1step,
                        timesteps,
                        fixed_a2b_emb,
                        q_emb=q_emb_a,
                        device=accelerator.device,
                    )
                else:
                    fake_a = CycleGAN_Turbo.forward_with_networks(
                        img_b,
                        "b2a",
                        vae_enc,
                        unet,
                        vae_dec,
                        noise_scheduler_1step,
                        timesteps,
                        fixed_b2a_emb,
                    )
                    fake_b = CycleGAN_Turbo.forward_with_networks(
                        img_a,
                        "a2b",
                        vae_enc,
                        unet,
                        vae_dec,
                        noise_scheduler_1step,
                        timesteps,
                        fixed_a2b_emb,
                    )

                print("loss_gan")
                loss_gan_a = net_disc_a(fake_b, for_G=True).mean() * args.lambda_gan
                loss_gan_b = net_disc_b(fake_a, for_G=True).mean() * args.lambda_gan
                accelerator.backward(loss_gan_a + loss_gan_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                optimizer_disc.zero_grad()

                """
                Identity Objective
                """
                # print("Identity objective")
                if args.quantum_dynamic:
                    idt_a = cyclegan_d.forward_with_networks_dynamic(
                        img_b,
                        "a2b",
                        vae_enc,
                        unet,
                        vae_dec,
                        noise_scheduler_1step,
                        timesteps,
                        fixed_a2b_emb,
                        q_emb=q_emb_b,
                        device=accelerator.device,
                    )
                else:
                    idt_a = CycleGAN_Turbo.forward_with_networks(
                        img_b,
                        "a2b",
                        vae_enc,
                        unet,
                        vae_dec,
                        noise_scheduler_1step,
                        timesteps,
                        fixed_a2b_emb,
                    )
                print("loss_idt")
                loss_idt_a = crit_idt(idt_a, img_b) * args.lambda_idt

                loss_idt_a += net_lpips(idt_a, img_b).mean() * args.lambda_idt_lpips

                if args.quantum_dynamic:
                    idt_b = cyclegan_d.forward_with_networks_dynamic(
                        img_a,
                        "b2a",
                        vae_enc,
                        unet,
                        vae_dec,
                        noise_scheduler_1step,
                        timesteps,
                        fixed_b2a_emb,
                        q_emb=q_emb_a,
                        device=accelerator.device,
                    )
                else:
                    idt_b = CycleGAN_Turbo.forward_with_networks(
                        img_a,
                        "b2a",
                        vae_enc,
                        unet,
                        vae_dec,
                        noise_scheduler_1step,
                        timesteps,
                        fixed_b2a_emb,
                    )
                loss_idt_b = crit_idt(idt_b, img_a) * args.lambda_idt
                loss_idt_b += net_lpips(idt_b, img_a).mean() * args.lambda_idt_lpips
                loss_g_idt = loss_idt_a + loss_idt_b
                accelerator.backward(loss_g_idt, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Discriminator for task a->b and b->a (fake inputs)
                """
                loss_D_A_fake = (
                    net_disc_a(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                )
                loss_D_B_fake = (
                    net_disc_b(fake_a.detach(), for_real=False).mean() * args.lambda_gan
                )

                loss_D_fake = (loss_D_A_fake + loss_D_B_fake) * 0.5
                print("loss_D")
                accelerator.backward(loss_D_fake, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(
                        net_disc_b.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                """
                Discriminator for task a->b and b->a (real inputs)
                """
                loss_D_A_real = (
                    net_disc_a(img_b, for_real=True).mean() * args.lambda_gan
                )
                loss_D_B_real = (
                    net_disc_b(img_a, for_real=True).mean() * args.lambda_gan
                )
                loss_D_real = (loss_D_A_real + loss_D_B_real) * 0.5
                print("loss_D_real")
                accelerator.backward(loss_D_real, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(
                        net_disc_b.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()
            print(f"For this step {global_step}: time = {time.time() - t_start}")
            logs = {}
            logs["cycle_a"] = loss_cycle_a.detach().item()
            logs["cycle_b"] = loss_cycle_b.detach().item()
            logs["gan_a"] = loss_gan_a.detach().item()
            logs["gan_b"] = loss_gan_b.detach().item()
            logs["disc_a"] = (
                loss_D_A_fake.detach().item() + loss_D_A_real.detach().item()
            )
            logs["disc_b"] = (
                loss_D_B_fake.detach().item() + loss_D_B_real.detach().item()
            )
            logs["idt_a"] = loss_idt_a.detach().item()
            logs["idt_b"] = loss_idt_b.detach().item()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    eval_unet = accelerator.unwrap_model(unet)
                    eval_vae_enc = accelerator.unwrap_model(vae_enc)
                    eval_vae_dec = accelerator.unwrap_model(vae_dec)

                    if (
                        global_step % args.checkpointing_steps == 1
                        or global_step == args.max_train_steps
                    ):
                        outf = os.path.join(
                            args.output_dir, "checkpoints", f"model_{global_step}.pkl"
                        )
                        sd = {}
                        sd["l_target_modules_encoder"] = l_modules_unet_encoder
                        sd["l_target_modules_decoder"] = l_modules_unet_decoder
                        sd["l_modules_others"] = l_modules_unet_others
                        sd["rank_unet"] = args.lora_rank_unet
                        sd["sd_encoder"] = get_peft_model_state_dict(
                            eval_unet, adapter_name="default_encoder"
                        )
                        sd["sd_decoder"] = get_peft_model_state_dict(
                            eval_unet, adapter_name="default_decoder"
                        )
                        sd["sd_other"] = get_peft_model_state_dict(
                            eval_unet, adapter_name="default_others"
                        )
                        sd["rank_vae"] = args.lora_rank_vae
                        sd["vae_lora_target_modules"] = vae_lora_target_modules
                        sd["sd_vae_enc"] = eval_vae_enc.state_dict()
                        sd["sd_vae_dec"] = eval_vae_dec.state_dict()
                        sd["quantum_params"] = boson_sampler.model.state_dict()
                        torch.save(sd, outf, _use_new_zipfile_serialization=False)
                        gc.collect()
                        torch.cuda.empty_cache()

                    # compute val FID and DINO-Struct scores
                    if global_step % args.validation_steps == 1:
                        _timesteps = torch.tensor(
                            [noise_scheduler_1step.config.num_train_timesteps - 1] * 1,
                            device="cuda",
                        ).long()
                        net_dino = DinoStructureLoss()
                        """
                        Evaluate "A->B"
                        """
                        fid_output_dir = os.path.join(
                            args.output_dir, f"fid-{global_step}/samples_a2b"
                        )
                        os.makedirs(fid_output_dir, exist_ok=True)
                        l_dino_scores_a2b = []

                        # get val input images from domain a
                        for idx, input_img_path in enumerate(tqdm(l_images_src_test)):
                            if (
                                idx > args.validation_num_images
                                and args.validation_num_images > 0
                            ):
                                break
                            outf = os.path.join(fid_output_dir, f"{idx}.png")
                            with torch.no_grad():
                                # print(f"input_img_path = {input_img_path}")
                                input_img = T_val(
                                    Image.open(input_img_path).convert("RGB")
                                )
                                img_a = transforms.ToTensor()(input_img)
                                img_a = (
                                    transforms.Normalize([0.5], [0.5])(img_a)
                                    .unsqueeze(0)
                                    .cuda()
                                )
                                # src_name = os.path.basename(input_img_path)

                                if args.quantum_dynamic:
                                    eval_fake_b, q_emb_a_val = (
                                        cyclegan_d.forward_with_networks_dynamic(
                                            img_a,
                                            "a2b",
                                            eval_vae_enc,
                                            eval_unet,
                                            eval_vae_dec,
                                            noise_scheduler_1step,
                                            _timesteps,
                                            fixed_a2b_emb[0:1],
                                            bs=boson_sampler,
                                            device=accelerator.device,
                                        )
                                    )

                                else:
                                    eval_fake_b = CycleGAN_Turbo.forward_with_networks(
                                        img_a,
                                        "a2b",
                                        eval_vae_enc,
                                        eval_unet,
                                        eval_vae_dec,
                                        noise_scheduler_1step,
                                        _timesteps,
                                        fixed_a2b_emb[0:1],
                                    )

                                eval_fake_b_pil = transforms.ToPILImage()(
                                    eval_fake_b[0] * 0.5 + 0.5
                                )
                                # only save images for which idx is 0-100
                                # if idx < 100:
                                eval_fake_b_pil.save(outf)
                                a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                                b = (
                                    net_dino.preprocess(eval_fake_b_pil)
                                    .unsqueeze(0)
                                    .cuda()
                                )
                                dino_ssim = net_dino.calculate_global_ssim_loss(
                                    a, b
                                ).item()
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
                        ed_mu, ed_sigma = (
                            np.mean(gen_features, axis=0),
                            np.cov(gen_features, rowvar=False),
                        )
                        score_fid_a2b = frechet_distance(
                            a2b_ref_mu, a2b_ref_sigma, ed_mu, ed_sigma
                        )
                        print(
                            f"step={global_step}, fid(a2b)={score_fid_a2b:.2f}, dino(a2b)={dino_score_a2b:.3f}"
                        )
                        # remove folder
                        shutil.rmtree(fid_output_dir)
                        """
                        compute FID for "B->A"
                        """
                        fid_output_dir = os.path.join(
                            args.output_dir, f"fid-{global_step}/samples_b2a"
                        )
                        os.makedirs(fid_output_dir, exist_ok=True)
                        l_dino_scores_b2a = []
                        # get val input images from domain b
                        for idx, input_img_path in enumerate(tqdm(l_images_tgt_test)):
                            if (
                                idx > args.validation_num_images
                                and args.validation_num_images > 0
                            ):
                                break
                            outf = os.path.join(fid_output_dir, f"{idx}.png")
                            with torch.no_grad():
                                input_img = T_val(
                                    Image.open(input_img_path).convert("RGB")
                                )
                                img_b = transforms.ToTensor()(input_img)
                                img_b = (
                                    transforms.Normalize([0.5], [0.5])(img_b)
                                    .unsqueeze(0)
                                    .cuda()
                                )
                                # src_name = os.path.basename(input_img_path)
                                if args.quantum_dynamic:
                                    eval_fake_a, q_emb_b_val = (
                                        cyclegan_d.forward_with_networks_dynamic(
                                            img_b,
                                            "b2a",
                                            eval_vae_enc,
                                            eval_unet,
                                            eval_vae_dec,
                                            noise_scheduler_1step,
                                            _timesteps,
                                            fixed_b2a_emb[0:1],
                                            bs=boson_sampler,
                                            device=accelerator.device,
                                        )
                                    )

                                else:
                                    eval_fake_a = CycleGAN_Turbo.forward_with_networks(
                                        img_b,
                                        "b2a",
                                        eval_vae_enc,
                                        eval_unet,
                                        eval_vae_dec,
                                        noise_scheduler_1step,
                                        _timesteps,
                                        fixed_b2a_emb[0:1],
                                    )

                                eval_fake_a_pil = transforms.ToPILImage()(
                                    eval_fake_a[0] * 0.5 + 0.5
                                )

                                # only save images for which idx is 0-100
                                # if idx < 100:
                                eval_fake_a_pil.save(outf)
                                a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                                b = (
                                    net_dino.preprocess(eval_fake_a_pil)
                                    .unsqueeze(0)
                                    .cuda()
                                )
                                dino_ssim = net_dino.calculate_global_ssim_loss(
                                    a, b
                                ).item()
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
                        ed_mu, ed_sigma = (
                            np.mean(gen_features, axis=0),
                            np.cov(gen_features, rowvar=False),
                        )
                        score_fid_b2a = frechet_distance(
                            b2a_ref_mu, b2a_ref_sigma, ed_mu, ed_sigma
                        )
                        print(
                            f"step={global_step}, fid(b2a)={score_fid_b2a}, dino(b2a)={dino_score_b2a:.3f}"
                        )
                        # remove folder
                        shutil.rmtree(fid_output_dir)
                        logs["val/fid_a2b"], logs["val/fid_b2a"] = (
                            score_fid_a2b,
                            score_fid_b2a,
                        )
                        logs["val/dino_struct_a2b"], logs["val/dino_struct_b2a"] = (
                            dino_score_a2b,
                            dino_score_b2a,
                        )
                        del net_dino  # free up memory

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break


if __name__ == "__main__":
    args = parse_args_unpaired_training()
    main(args)
