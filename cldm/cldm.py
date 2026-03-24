import einops
import torch
import torch as th
import torch.nn as nn
import cv2
import numpy as np

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion, DDPM, default # Import DDPM for default function
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from cldm.modules.gan_components import LatentGenerator, LatentDiscriminator

from cldm.structural_consensus import StructuralSemanticConsensus
from cldm.semantic_attention import (
    SemanticBundledAttentionLoss,
    FashionCompatibilityLoss,
    CLIPSemanticLoss,
)

# For L_perc (assuming LPIPS is not directly available in current context)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class SimpleGarmentSegmentor(nn.Module):
    """Lightweight part segmentor used when no external segmentor is provided."""
    def __init__(self, num_parts: int = 4):
        super().__init__()
        self.num_parts = num_parts
        self.proj = nn.Conv2d(3, num_parts, kernel_size=1)

    def forward(self, x):
        # x in [0, 1], output soft part masks (B, K, H, W)
        return torch.softmax(self.proj(x), dim=1)


class SimpleMultimodalEncoder(nn.Module):
    """
    Minimal image-text encoder interface with encode_image/encode_text.
    It keeps the full pipeline trainable without heavy external dependencies.
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.img_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.img_proj = nn.Linear(64, embed_dim)

        self.txt_emb = nn.Embedding(256, embed_dim)
        self.txt_proj = nn.Linear(embed_dim, embed_dim)

    def encode_image(self, x):
        h = self.img_net(x).flatten(1)
        return self.img_proj(h)

    def encode_text(self, texts):
        device = self.txt_emb.weight.device
        batch_vecs = []
        for text in texts:
            if text is None:
                text = ""
            byte_ids = list(text.encode("utf-8", errors="ignore"))
            if len(byte_ids) == 0:
                byte_ids = [0]
            ids = torch.tensor(byte_ids, device=device, dtype=torch.long)
            vec = self.txt_emb(ids).mean(dim=0)
            batch_vecs.append(vec)
        text_feats = torch.stack(batch_vecs, dim=0)
        return self.txt_proj(text_feats)

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control,
                 generator_config=None, discriminator_config=None,
                 gan_loss_weight=1.0, mode_seeking_loss_weight=1.0,
                 robustness_loss_weight=0.1, perceptual_loss_weight=0.1,
                 consensus_loss_weight=0.5, bundle_loss_weight=0.5,
                 compatibility_loss_weight=0.5, clip_semantic_loss_weight=0.1,
                 alpha_gamma=1.0,  # gamma for alpha_t, Eq (20)
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key

        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        # Initialize GAN components
        if generator_config is not None:
            self.latent_generator = instantiate_from_config(generator_config)
        else:
            # Fallback: direct construction with reasonable defaults
            # in_channels/out_channels follow latent diffusion channels,
            # context_dim will be inferred from cond_stage_model if possible.
            context_dim = getattr(getattr(self, "cond_stage_model", None), "context_dim", 1024)
            self.latent_generator = LatentGenerator(
                in_channels=self.channels,
                out_channels=self.channels,
                context_dim=context_dim,
            )

        if discriminator_config is not None:
            self.latent_discriminator = instantiate_from_config(discriminator_config)
        else:
            context_dim = getattr(getattr(self, "cond_stage_model", None), "context_dim", 1024)
            self.latent_discriminator = LatentDiscriminator(
                in_channels=self.channels,
                context_dim=context_dim,
            )

        # GAN related loss weights and parameters
        self.gan_loss_weight = gan_loss_weight
        self.mode_seeking_loss_weight = mode_seeking_loss_weight
        self.robustness_loss_weight = robustness_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.consensus_loss_weight = consensus_loss_weight
        self.bundle_loss_weight = bundle_loss_weight
        self.compatibility_loss_weight = compatibility_loss_weight
        self.clip_semantic_loss_weight = clip_semantic_loss_weight
        self.alpha_gamma = alpha_gamma

        # Perceptual loss for L_perc (Eq. 19)
        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device)

        # Lightweight built-in modules so the full objective is trainable by default.
        self.semantic_encoder = SimpleMultimodalEncoder(embed_dim=256)
        self.garment_segmentor = SimpleGarmentSegmentor(num_parts=4)
        self.structural_consensus = StructuralSemanticConsensus(
            clip_model=self.semantic_encoder,
            garment_segmentor=self.garment_segmentor,
            lambda_cons=self.consensus_loss_weight,
            lambda_global=0.5,
        )
        self.semantic_bundle_loss = SemanticBundledAttentionLoss(
            lambda_bundle=self.bundle_loss_weight
        )
        self.fashion_compat_loss = FashionCompatibilityLoss(
            encoder=self.semantic_encoder,
            lambda_comp=self.compatibility_loss_weight,
        )
        self.clip_semantic_loss = CLIPSemanticLoss(
            clip_model=self.semantic_encoder,
            lambda_perc=self.clip_semantic_loss_weight,
        )


    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        # x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        x, c, t = super().get_input(batch, self.first_stage_key, *args, **kwargs) # receive t
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        # control_processed=[]
        # for i in range(len(t)):
        #   control = batch[self.control_key][i].unsqueeze(0)
        #   if 950 <= t[i].item() < 1000:
        #     control_shape = control.shape
        #     black_image = torch.zeros(control_shape, dtype=torch.float32, device=self.device)
        #     control = black_image
        #     if bs is not None:
        #         control = control[:bs] 
        #     control = control.to(self.device)
        #     control = einops.rearrange(control, 'b h w c -> b c h w')
        #     control = control.to(memory_format=torch.contiguous_format).float()
        #     control_processed.append(control)
        #   else:    
        #     if bs is not None:
        #         control = control[:bs]
        #     control = control.to(self.device)
        #     control = einops.rearrange(control, 'b h w c -> b c h w')
        #     control = control.to(memory_format=torch.contiguous_format).float()
        #     control_processed.append(control)


        return x, dict(c_crossattn=[c], c_concat=[control]), t

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:   
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        # z, c = self.get_input(batch, self.first_stage_key, bs=N)
        z, c, t = self.get_input(batch, self.first_stage_key, bs=N) # hierarchical t
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t_idx in range(self.num_timesteps):
                if t_idx % self.log_every_t == 0 or t_idx == self.num_timesteps - 1:
                    t_sample = repeat(torch.tensor([t_idx]), '1 -> b', b=n_row)
                    t_sample = t_sample.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t_sample, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def p_losses(self, x_start, cond, t, noise=None, texts=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Diffusion Refiner (F_theta) output
        model_output_denoise = self.apply_model(x_noisy, t, cond) # This is epsilon_theta(z_t, c, t) in methodology

        # Denoising loss (L_denoise) - This is the standard diffusion loss
        if self.parameterization == "x0":
            target_denoise = x_start
        elif self.parameterization == "eps":
            target_denoise = noise
        elif self.parameterization == "v":
            target_denoise = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()
        
        loss_denoise_simple = self.get_loss(model_output_denoise, target_denoise, mean=False).mean([1, 2, 3])
        logvar_t = self.logvar[t].to(self.device)
        loss_denoise = loss_denoise_simple / torch.exp(logvar_t) + logvar_t
        loss_denoise = self.l_simple_weight * loss_denoise.mean()

        # GAN Ideation (Generator G) at timestep t
        # u ~ N(0, I) (stochastic variation)
        u = torch.randn_like(x_noisy) # u has the same shape as z_t
        # c (multimodal conditioning) from cond dictionary
        c_crossattn = torch.cat(cond['c_crossattn'], 1) # Assuming c_crossattn is the text/pose conditioning

        # z_hat_(t-1) = G(z_t, u, c, t) - generator predicts a diverse latent candidate
        # Note: latent_generator expects t to be (batch_size, 1), not (batch_size,) if it's used in linear layer
        z_gen = self.latent_generator(x_noisy, u, c_crossattn, t) 

        # Discriminator D evaluates realism
        # To get z_(t-1)^real, we denoise x_noisy towards x_start and take the mean of the posterior distribution q(z_(t-1)|z_t, z_0)
        model_mean, _, _ = self.q_posterior(x_start=x_start, x_t=x_noisy, t=t)
        z_t_minus_1_real = model_mean 

        # Discriminator scores
        # D(z_(t-1)^real) and D(z_hat_(t-1))
        d_real = self.latent_discriminator(z_t_minus_1_real.detach(), c_crossattn, t) 
        d_fake = self.latent_discriminator(z_gen, c_crossattn, t) 

        # L_D: Hinge-based adversarial objective (Eq. 14)
        loss_D = torch.mean(torch.nn.functional.relu(1 - d_real)) + torch.mean(torch.nn.functional.relu(1 + d_fake))

        # L_G: Generator's adversarial loss (Eq. 15)
        loss_G_adv = -torch.mean(d_fake)

        # L_MS: Mode-seeking loss (Eq. 16)
        u1 = torch.randn_like(x_noisy)
        u2 = torch.randn_like(x_noisy)
        z_gen_u1 = self.latent_generator(x_noisy, u1, c_crossattn, t)
        z_gen_u2 = self.latent_generator(x_noisy, u2, c_crossattn, t)
        # To avoid division by zero, add a small epsilon (already handled in previous iteration)
        loss_MS = -torch.mean(torch.abs(z_gen_u1 - z_gen_u2).sum(dim=[1,2,3]) / (torch.norm(u1 - u2, p=1, dim=[1,2,3]) + 1e-6))
        # loss_MS = loss_MS.mean() # Already averaged over batch due to sum() and mean()

        # L_rob: Robustness loss (Eq. 18) - consistency against small perturbations delta
        # This requires re-running the diffusion model (F_theta) with a perturbed input
        delta = torch.randn_like(x_noisy) * 0.01 # Small adversarial noise
        model_output_denoise_perturbed = self.apply_model(x_noisy + delta, t, cond)
        loss_rob = torch.nn.functional.mse_loss(model_output_denoise, model_output_denoise_perturbed)

        # 1) z_tilde_denoise from diffusion refiner F_theta
        x_recon_denoise = self.predict_start_from_noise(
            x_noisy, t, model_output_denoise
        ) if self.parameterization == "eps" else model_output_denoise
        z_tilde_denoise = x_recon_denoise  # F_theta(z_t, c, t)
        # 2) alpha_t from Eq. (20)
        alpha_t_scalar = torch.pow(t.float() / self.num_timesteps, self.alpha_gamma)
        alpha_t = alpha_t_scalar.view(-1, 1, 1, 1)
        # 3) Fusion z_{t-1} = alpha_t * z_hat_{t-1} + (1-alpha_t) * z_tilde_{t-1}
        z_interpolated_t_minus_1 = alpha_t * z_gen + (1.0 - alpha_t) * z_tilde_denoise
        # 4) Decode fused latent for perception/semantic losses
        decoded_x_recon = self.decode_first_stage(z_interpolated_t_minus_1)
        decoded_x_start = self.decode_first_stage(x_start)
        decoded_x_recon = (decoded_x_recon + 1.0) / 2.0
        decoded_x_start = (decoded_x_start + 1.0) / 2.0
        loss_perc = self.lpips_loss(decoded_x_recon, decoded_x_start).mean()

        # EquiFashion structural / semantic losses
        if self.structural_consensus is not None and texts is not None:
            loss_cons = self.structural_consensus(decoded_x_recon, list(texts))
        else:
            loss_cons = decoded_x_recon.new_tensor(0.0)

        if self.semantic_bundle_loss is not None:
            # Proxy attention maps from reconstructed image activations.
            # Keeps bundle regularization active without requiring internal
            # transformer attention hooks.
            adj_map = decoded_x_recon[:, 0].abs()
            noun_map = decoded_x_recon[:, 1].abs() if decoded_x_recon.shape[1] > 1 else adj_map
            adj_map = adj_map / (adj_map.sum(dim=[1, 2], keepdim=True) + 1e-6)
            noun_map = noun_map / (noun_map.sum(dim=[1, 2], keepdim=True) + 1e-6)
            loss_bundle = self.semantic_bundle_loss([adj_map], [noun_map])
        else:
            loss_bundle = decoded_x_recon.new_tensor(0.0)

        if self.clip_semantic_loss is not None and texts is not None:
            loss_clip_sem = self.clip_semantic_loss(decoded_x_recon, list(texts))
        else:
            loss_clip_sem = decoded_x_recon.new_tensor(0.0)

        if self.fashion_compat_loss is not None and texts is not None:
            loss_comp = self.fashion_compat_loss(decoded_x_recon, list(texts))
        else:
            loss_comp = decoded_x_recon.new_tensor(0.0)

        # Loss for Generator and Diffusion Refiner (combined G and F_theta)
        loss_G_F = (
            self.gan_loss_weight * (loss_G_adv + self.mode_seeking_loss_weight * loss_MS)
            + self.perceptual_loss_weight * loss_perc
            + self.robustness_loss_weight * loss_rob
            + self.l_simple_weight * loss_denoise  # L_denoise is for F_theta
            + loss_cons
            + loss_bundle
            + loss_clip_sem
            + loss_comp
        )

        # Loss for Discriminator
        loss_D_total = self.gan_loss_weight * loss_D
        
        loss_dict = {
            f'train/loss_denoise_simple': loss_denoise_simple.mean(),
            f'train/loss_denoise': loss_denoise.mean(),
            f'train/loss_D': loss_D_total.mean(),
            f'train/loss_G_adv': loss_G_adv.mean(),
            f'train/loss_MS': loss_MS.mean(),
            f'train/loss_rob': loss_rob.mean(),
            f'train/loss_perc': loss_perc.mean(),
            f'train/loss_cons': loss_cons.mean(),
            f'train/loss_bundle': loss_bundle.mean(),
            f'train/loss_clip_sem': loss_clip_sem.mean(),
            f'train/loss_comp': loss_comp.mean(),
            f'train/loss_G_F_total': loss_G_F.mean(),
        }

        # Returning different losses for alternating optimization
        return loss_G_F, loss_D_total, loss_dict

    def forward(self, x, c, t, *args, **kwargs):
        # The forward method now calls the updated p_losses
        return self.p_losses(x, c, t, *args, **kwargs)

    def configure_optimizers(self):
        lr = self.learning_rate
        
        # Optimizers for Generator (G) and Diffusion Refiner (F_theta)
        params_G_F = list(self.control_model.parameters()) + \
                     list(self.model.diffusion_model.parameters()) + \
                     list(self.latent_generator.parameters()) + \
                     list(self.semantic_encoder.parameters()) + \
                     list(self.garment_segmentor.parameters())

        if not self.sd_locked:
            params_G_F += list(self.model.diffusion_model.output_blocks.parameters())
            params_G_F += list(self.model.diffusion_model.out.parameters())

        if self.learn_logvar:
            params_G_F.append(self.logvar)
        
        opt_G_F = torch.optim.AdamW(params_G_F, lr=lr)

        # Optimizer for Discriminator (D)
        opt_D = torch.optim.AdamW(list(self.latent_discriminator.parameters()), lr=lr)

        return [opt_G_F, opt_D]

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, c, t = self.get_input(batch, self.first_stage_key)
        texts = batch.get(self.cond_stage_key, None)

        # Train Generator and Diffusion Refiner (G_F)
        if optimizer_idx == 0:
            loss_G_F, _, loss_dict = self.p_losses(x, c, t, texts=texts)
            self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return loss_G_F

        # Train Discriminator (D)
        if optimizer_idx == 1:
            with torch.no_grad():  # Discriminator doesn't need gradients from G_F
                _, loss_D_total, loss_dict = self.p_losses(x, c, t, texts=texts)

            self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return loss_D_total

    @torch.no_grad()
    def sample_hybrid(self, cond, batch_size, ddim_steps, eta=0.0, **kwargs):
        """
        Hybrid GAN-Diffusion sampling that fuses z_hat_{t-1} from the latent
        generator and z_tilde_{t-1} from the diffusion refiner at every step
        (Eq. 20 in the EquiFashion pipeline).
        """
        device = self.device
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)

        # Start from z_T ~ N(0, I)
        z_t = torch.randn(batch_size, *shape, device=device)

        # DDIM timestep schedule (used only to define a sequence of t)
        ddim_sampler = DDIMSampler(self)
        _, timesteps = ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=False)

        c_crossattn = torch.cat(cond["c_crossattn"], 1)

        for t_scalar in timesteps:
            t_batch = torch.full((batch_size,), t_scalar, device=device, dtype=torch.long)

            # 1) Diffusion refiner F_theta: eps = epsilon_theta(z_t, c, t)
            eps = self.apply_model(z_t, t_batch, cond)

            # 2) x0 and z_tilde_{t-1} from DDPM posterior q(z_{t-1} | z_t, z_0)
            x0 = self.predict_start_from_noise(z_t, t_batch, eps)
            model_mean, _, _ = self.q_posterior(x_start=x0, x_t=z_t, t=t_batch)
            z_tilde_denoise = model_mean

            # 3) GAN ideation: z_hat_{t-1} = G(z_t, u, c, t)
            u = torch.randn_like(z_t)
            z_gen = self.latent_generator(z_t, u, c_crossattn, t_batch)

            # 4) Fusion Eq. (20)
            alpha_t_scalar = torch.pow(t_batch.float() / self.num_timesteps, self.alpha_gamma)
            alpha_t = alpha_t_scalar.view(-1, 1, 1, 1)
            z_t_minus_1 = alpha_t * z_gen + (1.0 - alpha_t) * z_tilde_denoise

            z_t = z_t_minus_1

        x_0 = self.decode_first_stage(z_t)
        return x_0
