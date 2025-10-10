import copy
import os
import sys

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel
from img2img_turbo_annotations.src_quantum.model import (
    download_url,
    make_1step_sched,
    my_vae_decoder_fwd,
    my_vae_encoder_fwd,
)
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel

p = "src/"
sys.path.append(p)


class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super().__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super().__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        # print(f"--- CONV_IN in vae {_vae.decoder.conv_in}")
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


def initialize_unet(rank, return_lora_module_names=False):
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/sd-turbo", subfolder="unet"
    )
    num_params = sum(p.numel() for p in unet.parameters())
    print(f"-----> Number of parameters in UNet: {num_params}")
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

    lora_conf_encoder = LoraConfig(
        r=rank,
        init_lora_weights="gaussian",
        target_modules=l_target_modules_encoder,
        lora_alpha=rank,
    )
    lora_conf_decoder = LoraConfig(
        r=rank,
        init_lora_weights="gaussian",
        target_modules=l_target_modules_decoder,
        lora_alpha=rank,
    )
    lora_conf_others = LoraConfig(
        r=rank,
        init_lora_weights="gaussian",
        target_modules=l_modules_others,
        lora_alpha=rank,
    )
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
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
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
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
    # conv_in + conv_out fully trained
    if dynamic:
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
    vae_lora_config = LoraConfig(
        r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules
    )
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae


class CycleGAN_Turbo(torch.nn.Module):
    def __init__(
        self,
        pretrained_name=None,
        pretrained_path=None,
        ckpt_folder="checkpoints",
        lora_rank_unet=8,
        lora_rank_vae=4,
        accelerator=None,
    ):
        super().__init__()

        # init components without device specification
        self.tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/sd-turbo", subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/sd-turbo", subfolder="text_encoder"
        ).cuda()
        self.sched = make_1step_sched()
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-turbo", subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/sd-turbo", subfolder="unet"
        )

        self.vae.encoder.forward = my_vae_encoder_fwd.__get__(
            self.vae.encoder, self.vae.encoder.__class__
        )
        self.vae.decoder.forward = my_vae_decoder_fwd.__get__(
            self.vae.decoder, self.vae.decoder.__class__
        )
        # add the skip connection convs
        self.vae.decoder.skip_conv_1 = torch.nn.Conv2d(
            512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        self.vae.decoder.skip_conv_2 = torch.nn.Conv2d(
            256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        self.vae.decoder.skip_conv_3 = torch.nn.Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        self.vae.decoder.skip_conv_4 = torch.nn.Conv2d(
            128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        self.vae.decoder.ignore_skip = False

        if pretrained_name == "day_to_night":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the night"
            self.direction = "a2b"
        elif pretrained_name == "night_to_day":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/night2day.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the day"
            self.direction = "b2a"
        elif pretrained_name == "clear_to_rainy":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/clear2rainy.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in heavy rain"
            self.direction = "a2b"
        elif pretrained_name == "rainy_to_clear":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/rainy2clear.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the day"
            self.direction = "b2a"

        elif pretrained_path is not None:
            print("-- Loading from pretrained path not None --")
            sd = torch.load(pretrained_path)
            self.load_ckpt_from_state_dict(sd)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = None
            self.direction = None

        # define object vae_enc and vae_dec
        """self.vae.decoder.gamma = 1
        self.vae_b2a = copy.deepcopy(self.vae)
        self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)"""

        """self.vae_enc.cuda()
        self.vae_dec.cuda()
        self.unet.cuda()"""

    def load_ckpt_from_state_dict(self, sd, dyn=False):
        print("--load_ckpt_from_state_dict--")

        lora_conf_encoder = LoraConfig(
            r=sd["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=sd["l_target_modules_encoder"],
            lora_alpha=sd["rank_unet"],
        )
        lora_conf_decoder = LoraConfig(
            r=sd["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=sd["l_target_modules_decoder"],
            lora_alpha=sd["rank_unet"],
        )
        lora_conf_others = LoraConfig(
            r=sd["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=sd["l_modules_others"],
            lora_alpha=sd["rank_unet"],
        )
        if not dyn:
            self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
            self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
            self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_encoder.weight", ".weight")
            if "lora" in n and "default_encoder" in n:
                p.data.copy_(sd["sd_encoder"][name_sd])
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_decoder.weight", ".weight")
            if "lora" in n and "default_decoder" in n:
                p.data.copy_(sd["sd_decoder"][name_sd])
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_others.weight", ".weight")
            if "lora" in n and "default_others" in n:
                p.data.copy_(sd["sd_other"][name_sd])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        vae_lora_config = LoraConfig(
            r=sd["rank_vae"],
            init_lora_weights="gaussian",
            target_modules=sd["vae_lora_target_modules"],
        )
        if not dyn:
            self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae.decoder.gamma = 1

        # Create copies using the correct device
        self.vae_b2a = copy.deepcopy(self.vae)

        # Create VAE components
        self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)

        # self.vae_b2a = copy.deepcopy(self.vae)
        # self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)

        self.vae_enc.load_state_dict(sd["sd_vae_enc"])
        # self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_dec.load_state_dict(sd["sd_vae_dec"])

    def load_ckpt_from_url(self, url, ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
        outf = os.path.join(ckpt_folder, os.path.basename(url))
        download_url(url, outf)
        sd = torch.load(outf)
        self.load_ckpt_from_state_dict(sd)

    @staticmethod
    def forward_with_networks(
        x, direction, vae_enc, unet, vae_dec, sched, timesteps, text_emb
    ):
        B = x.shape[0]
        assert direction in ["a2b", "b2a"]
        x_enc = vae_enc(x, direction=direction).to(x.dtype)
        model_pred = unet(
            x_enc,
            timesteps,
            encoder_hidden_states=text_emb,
        ).sample
        x_out = torch.stack([
            sched.step(
                model_pred[i], timesteps[i], x_enc[i], return_dict=True
            ).prev_sample
            for i in range(B)
        ])
        x_out_decoded = vae_dec(x_out, direction=direction)
        return x_out_decoded

    @staticmethod
    def forward_with_networks_dynamic(
        x,
        direction,
        vae_enc,
        unet,
        vae_dec,
        sched,
        timesteps,
        text_emb,
        bs=None,
        q_emb=None,
        device=None,
        accelerator=None,
    ):
        B = x.shape[0]
        starter_bool = q_emb is None  # will return q_emb if none was given as input
        assert direction in ["a2b", "b2a"]
        if bs is None and q_emb is None:
            raise AttributeError(
                "Cannot compute quantum embeddings if boson sampler not given"
            )

        # monitor NaN occurence
        if torch.isnan(x).any():
            print("?????? INPUT OF VAE ENCODER IS NAN ??????")

        # normal pass with x being the image
        x_enc = vae_enc(x, direction=direction).to(x.dtype)

        # monitor NaN occurence
        if torch.isnan(x_enc).any():
            print("?????? OUTPUT OF VAE ENCODER IS NAN ??????")
        if q_emb is None:
            with torch.no_grad():
                # need to compute the quantum embedding from the VAE encoder
                q_emb = torch.logit(
                    bs.compute(torch.sigmoid(x_enc), unitaries=bs.unitaries)
                )
            if torch.isnan(q_emb).any():
                print(
                    "!!!!!!!!!!!!!!!!!!!!! NEED TO REPLACE NAN EMB WITH RANDOM EMB !!!!!!!!!!!!!!!!!!!!!"
                )
                q_emb = torch.rand(q_emb.shape)

            if device is not None:
                # print("Sending q_emb to accelerator")
                q_emb = q_emb.to(device)
            else:
                # print("Sending q_emb to cuda")
                q_emb = q_emb.cuda()

        # concatenation of the quantum embeddings with the text embeddings
        text_emb = torch.cat((text_emb, q_emb.view(B, -1).unsqueeze(1)), dim=1)
        model_pred = unet(
            x_enc,
            timesteps,
            encoder_hidden_states=text_emb,
        ).sample

        x_out = torch.stack([
            sched.step(
                model_pred[i], timesteps[i], x_enc[i], return_dict=True
            ).prev_sample
            for i in range(B)
        ])
        x_out_decoded = vae_dec(x_out, direction=direction)
        if starter_bool:
            # print("Returning x_out_decoded and q_emb")
            # we return the q_emb in case it is reused at the same iteration
            return x_out_decoded, q_emb
        # print("Returning only x_out_decoded")
        return x_out_decoded

    @staticmethod
    def get_traininable_params(
        unet, vae_a2b, vae_b2a, boson_sampler, dynamic=False, quantum_training=False
    ):
        # add all unet parameters
        if quantum_training:
            dynamic = False
        if not dynamic:
            params_gen = list(unet.conv_in.parameters())
            unet.conv_in.requires_grad_(True)
        if quantum_training:
            params_gen += list(boson_sampler.model.parameters())
        if dynamic:
            # params_gen = list(unet.parameters())
            params_gen = list(unet.conv_in.parameters())
            unet.conv_in.requires_grad_(True)
            # unet.requires_grad_(True)
            params_gen.extend(list(vae_a2b.decoder.conv_in.parameters()))
            params_gen.extend(list(vae_b2a.decoder.conv_in.parameters()))
            params_gen.extend(list(vae_a2b.post_quant_conv.parameters()))
            params_gen.extend(list(vae_b2a.post_quant_conv.parameters()))
            # params_gen.extend(list(boson_sampler.model.parameters()))
            # #params_gen.extend(list(unet.conv_out.parameters()))
            # vae_a2b.decoder.conv_in.requires_grad_(True)
            # vae_b2a.decoder.conv_in.requires_grad_(True)
            # vae_a2b.post_quant_conv.requires_grad_(True)
            # vae_b2a.post_quant_conv.requires_grad_(True)
            # unet.conv_out.requires_grad_(True)
        unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
        if not dynamic:
            for n, p in unet.named_parameters():
                if "lora" in n and "default" in n:
                    assert p.requires_grad
                    params_gen.append(p)

        # add all vae_a2b parameters
        for n, p in vae_a2b.named_parameters():
            if "lora" in n and "vae_skip" in n:
                # assert p.requires_grad
                if p.requires_grad:
                    params_gen.append(p)

        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_4.parameters())

        # add all vae_b2a parameters
        for n, p in vae_b2a.named_parameters():
            if "lora" in n and "vae_skip" in n:
                # assert p.requires_grad
                if p.requires_grad:
                    params_gen.append(p)

        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_4.parameters())

        return params_gen

    def forward(self, x_t, direction=None, caption=None, caption_emb=None):
        if direction is None:
            assert self.direction is not None
            direction = self.direction
        if caption is None and caption_emb is None:
            assert self.caption is not None
            caption = self.caption
        if caption_emb is not None:
            caption_enc = caption_emb
        else:
            caption_tokens = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(x_t.device)
            caption_enc = self.text_encoder(caption_tokens)[0].detach().clone()
        return self.forward_with_networks(
            x_t,
            direction,
            self.vae_enc,
            self.unet,
            self.vae_dec,
            self.sched,
            self.timesteps,
            caption_enc,
        )
