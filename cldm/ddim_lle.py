"""Low-level attribute editing."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like
)

import cv2

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        # Return \sigma_t, \alpha_t, \alpha_{t-1}
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    # @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               init_img=None,english_attribute=None,
               **kwargs
               ):

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule,
                                                    init_img=init_img,english_attribute=english_attribute,
                                                    )
        return samples, intermediates

    # @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None,init_img=None,english_attribute=None):
        device = self.model.betas.device
        b = shape[0] # batch_size

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps) # [1000,991,...,0]
    
        #  into the latent space
        if init_img is not None:
            assert (
                x0 is None and x_T is None
            ), "Try to infer x0 and x_t from init_image, but they already provided"

            encoder_posterior = self.model.encode_first_stage(init_img)
            x0 = self.model.get_first_stage_encoding(encoder_posterior) # torch.Size([1, 4, 88, 64])

            if english_attribute == "A1":
                last_ts = torch.full((1,), time_range[8], device=device, dtype=torch.long) # Add noise to T time steps
            elif english_attribute == "A2":
                last_ts = torch.full((1,), time_range[16], device=device, dtype=torch.long) 
            elif english_attribute == "A3":
                last_ts = torch.full((1,), time_range[24], device=device, dtype=torch.long)
            elif english_attribute == "A4":
                last_ts = torch.full((1,), time_range[32], device=device, dtype=torch.long) 
            elif english_attribute == "A5":
                last_ts = torch.full((1,), time_range[40], device=device, dtype=torch.long) 
            
            x_T = torch.cat([self.model.q_sample(x0, last_ts) for _ in range(b)]) # concat

            img = x_T
        elif x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        """
        array([991, 981, 971, 961, 951, 941, 931, 921, 911, 901, 891, 881, 871,
        861, 851, 841, 831, 821, 811, 801, 791, 781, 771, 761, 751, 741,
        731, 721, 711, 701, 691, 681, 671, 661, 651, 641, 631, 621, 611,
        601, 591, 581, 571, 561, 551, 541, 531, 521, 511, 501, 491, 481,
        471, 461, 451, 441, 431, 421, 411, 401, 391, 381, 371, 361, 351,
        341, 331, 321, 311, 301, 291, 281, 271, 261, 251, 241, 231, 221,
        211, 201, 191, 181, 171, 161, 151, 141, 131, 121, 111, 101,  91,
            81,  71,  61,  51,  41,  31,  21,  11,   1])
        """
        
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        if mask is not None:
            mask = torch.nn.functional.interpolate(mask, size=x0.shape[-2:]) # Downsample the mask

        # Define the conditions as a list of tuples: (time_range, english_attribute)
        conditions = [
            (time_range[8], "A1"),
            (time_range[16], "A2"),
            (time_range[24], "A3"),
            (time_range[32], "A4"),
            (time_range[40], "A5")
        ]

        for i, step in enumerate(iterator):
            index = total_steps - i - 1

            # Check each condition and execute the block if it matches
            for max_time, attr in conditions:
                if 0 < step <= max_time and english_attribute == attr:
                    ts = torch.full((b,), step, device=device, dtype=torch.long)
                    
                    img_orig = self.model.q_sample(x0, ts)  # Add noise to input image
                    
                    outs = self.p_sample_ddim_edit(
                        img, cond, ts, index=index,
                        use_original_steps=ddim_use_original_steps,
                        quantize_denoised=quantize_denoised,
                        temperature=temperature,
                        noise_dropout=noise_dropout,
                        score_corrector=score_corrector,
                        corrector_kwargs=corrector_kwargs,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=unconditional_conditioning,
                        dynamic_threshold=dynamic_threshold,
                        x_h_latent=img_orig, mask=mask,
                    )
                    
                    img, pred_x0 = outs  # Extract outputs

                    if index % log_every_t == 0 or index == total_steps - 1:
                        intermediates['x_inter'].append(img)
                        intermediates['pred_x0'].append(pred_x0)
                    break  # Exit the loop once a condition is met

        return img, intermediates
    

    @torch.no_grad()
    def p_sample_ddim_edit(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None,x_h_latent=None,mask=None):
        b, *_, device = *x.shape, x.device

        self.model.requires_grad_(True) 
        x.requires_grad_(True)       

        # \alpha_t
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        # \alpha_{t-1}
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        # \sigma_t
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device) 
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            # apply_model
            model_t = self.model.apply_model(x, t, c) # pred noise；e_t
            model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)   # e_t
        
        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output) 
        
        # eps
        else:
            e_t = model_output
        
        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        # current prediction for x_0
        if self.model.parameterization != "v":
            # pred x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt() # \frac{{x}_{t}-\sqrt{1-\alpha_{t}}\epsilon_{\theta}^{(t)}({x}_{t})}{\sqrt{\alpha_{t}}}
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()  
        
        # Calculation direction
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t # \sqrt{1-\alpha_{t-1}-\sigma_{t}^{2}}\cdot\epsilon_{\theta}^{(t)}({x}_{t})
        
        # random noise
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature #\sigma_{t}\epsilon_{t}
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        
        # x_t−1​ 
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise # x_{t-1}=\sqrt{\alpha_{t-1}}\left(\frac{{x}_{t}-\sqrt{1-\alpha_{t}}\epsilon_{\theta}^{(t)}({x}_{t})}{\sqrt{\alpha_{t}}}\right)+{\sqrt{1-\alpha_{t-1}-\sigma_{t}^{2}}\cdot\epsilon_{\theta}^{(t)}({x}_{t})}+{\sigma_{t}\epsilon_{t}}
        
        x_prev = (1. - mask) * x_h_latent + mask * x_prev # optimize the unedited area

        return x_prev, pred_x0


