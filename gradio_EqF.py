import cv2
import einops 
import gradio as gr
import numpy as np
import torch
import random


from pytorch_lightning import seed_everything
from utils.resize import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_lle import DDIMSampler as DDIMSampler_LLE
from cldm.ddim_hlg import DDIMSampler as DDIMSampler_HLG
from automation_pose_mask.openpose import OpenposeDetector
from automation_pose_mask.auto_mask import MaskDetector
from PIL import Image
from rembg import remove

from utils.config import *

# import debugpy; debugpy.listen(('127.0.0.1', 56789)); debugpy.wait_for_client()

apply_openpose = OpenposeDetector(body_model_path=openpose_body_model_path,
                                  hand_model_path=openpose_hand_model_path)
apply_mask = MaskDetector(sam_model_path=sam_model_path)

model = create_model(model_yaml).cpu()
model.load_state_dict(load_state_dict(my_model_path, location='cuda'))

hlg_sampler = DDIMSampler_HLG(model)
lle_sampler = DDIMSampler_LLE(model)

example_path = os.path.join(os.path.dirname(__file__), "preselected_images")
example_image_list = [os.path.join(example_path, x)
                      for x in os.listdir(example_path)]

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8) 
    output_mask = Image.fromarray(mask)
    
    return output_mask

def add_white_background(image):
    # Ensure image.size is a tuple
    if not isinstance(image.size, tuple):
        raise ValueError("Size must be a tuple containing width and height of the image")
    
    # Create a new image with a white background
    white_bg = Image.new("RGBA", image.size, "WHITE")
    white_bg.paste(image, (0, 0), image)
    return white_bg.convert("RGB")

def hlg_process(hlg_prompt, input_image, category, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    device = "cuda"
    model.to(device)
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(detected_map).save("./mask_result/pose.jpg")

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 4294967294)
        seed_everything(seed)


        print(hlg_prompt)
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([hlg_prompt + ', ' + a_prompt] * num_samples)]}
        # cond = {"c_concat": [control], "c_crossattn": []} 
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = hlg_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)


        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        # results = [x_samples[i] for i in range(num_samples)]
        num_samples = x_samples.shape[0]

        results = [Image.fromarray(x_samples[i]) for i in range(num_samples)]

        rmbg_results = [remove(img) for img in results]

        wbg_results = [add_white_background(img) for img in rmbg_results]

        device = "cpu"
        model.to(device)
        torch.cuda.empty_cache()
        # return [detected_map] + results, ch_prompt_out
        return wbg_results


def lle_process(lle_prompt, dict_img_mask, category, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, attribute, selection_mode):
    # grad update
    device = "cuda"
    model.to(device)
    
    input_image = dict_img_mask["background"].convert("RGB")  
    # convert NumPy data
    input_image = np.array(input_image)
    # print(input_image.shape)
    input_image = HWC3(input_image)
    # detect pose
    detected_map, keypoints = apply_openpose(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)
    Image.fromarray(detected_map).save("./mask_result/pose.jpg")

    english_category = category
    print(english_category)

    english_attribute = attribute
    print(english_attribute)

    sam_mode = False
    sketch_mode = False
    # "Automatically recognize", "User interface"
    if selection_mode == "Automatically recognize":
        sam_mode = True
        sketch_mode = False
    elif selection_mode == "User interface":
        sam_mode = False
        sketch_mode = True



    if sketch_mode:
        mask = pil_to_binary_mask(dict_img_mask['layers'][0].convert("RGB")) # PIL.Image
        mask.save("./mask_result/mask_user.jpg")
    elif sam_mode:
        mask = apply_mask(resize_image(input_image, detect_resolution), keypoints, category=english_category, attribute=english_attribute, sam_mode=sam_mode)
    else : 
        mask = None   

    if mask is not None:
        mask = mask.convert("L")
        mask = np.array(mask)
        mask = torch.from_numpy(mask.copy()).float().cuda() / 255.0
        mask = mask[None,None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1


    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape

    init_img = torch.from_numpy(img.copy()).float().cuda() / 127.0 - 1.0 # normalization
    init_img = torch.stack([init_img for _ in range(num_samples)], dim=0)
    init_img = einops.rearrange(init_img, 'b h w c -> b c h w').clone()

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    if seed == -1:
        seed = random.randint(0, 4294967294)
    seed_everything(seed)

    print(lle_prompt)
    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([lle_prompt + ', ' + a_prompt] * num_samples)]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
    shape = (4, H // 8, W // 8)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

    samples, intermediates = lle_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond,init_img=init_img,mask=mask,english_attribute=english_attribute)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().detach().numpy().clip(0, 255).astype(np.uint8)
    
    results = [x_samples[i] for i in range(num_samples)]
    device = "cpu"
    model.to(device)
    torch.cuda.empty_cache()
    # return [detected_map] + [mask] + results, ch_prompt_out
    return results

def result2input(images):
    # print(images)
    res_dict = {
        "background": images[-1][0],
        "layers":None,
        "composite": None
    }
    return res_dict

sam_dict = dict(sam_masks=None, mask_image=None, cnet=None, orig_image=None, pad_mask=None)


def create_hfddm():
    with gr.Blocks().queue() as app:

        category = gr.Radio(list(category_dict.values()),
                                value=list(category_dict.values())[0],
                                label="Select the category of clothing you want to design")
                    
        with gr.Row():       
            with gr.Column():                
                with gr.Tab("Draft design"):
                    with gr.Row():
                        hlg_prompt = gr.Textbox(label="Please enter the high-level design concept", interactive=True)
                        
                    hlg_input_image = gr.Image(sources=("upload", "webcam"), type="numpy", value=example_image_list[0], label="Reference pose")
                    example = gr.Examples(
                        inputs=hlg_input_image,
                        examples_per_page=20,
                        examples=example_image_list)
                        
                    hlg_run = gr.Button("Generate")


                with gr.Tab("Attribute editing"):
                # with gr.Column():
                    lle_prompt = gr.Textbox(label="Please upload the attribute for editing", interactive=True)

                    lle_input_image = gr.ImageEditor(sources='upload', type="pil", label='Please enter the edited image', interactive=True, value=example_image_list[0])
                    lle_example = gr.Examples(
                        inputs=lle_input_image,
                        # examples_per_page=14,
                        examples=example_image_list)
                    
                    # radio button
                    selection_mode = gr.Radio(["Automatically recognize", "User interface"], label="Edit the region selection", value="Automatically recognize")
                    

                    current_tab = {}
                    lle_run = {}
                    for tab_elem in list(attribute_dict.values()):
                        with gr.Tab(tab_elem):
                            current_tab[tab_elem] = gr.Label(value=tab_elem, visible=False)

                            lle_run[tab_elem] = gr.Button("Generate")

                    
            with gr.Column():
                with gr.Row():
                    result_gallery = gr.Gallery(label=' ', show_label=False, elem_id="gallery", selected_index=0, interactive=False)
                    

                with gr.Row():
                    with gr.Column():
                        pass
                        
                    with gr.Column():
                        send2llg = gr.Button("Send to attribute editing")

            

        with gr.Accordion("Advanced Options", open=False, visible=True):
            num_samples = gr.Slider(label="Images", minimum=1, maximum=1, value=1, step=1)
            image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
            strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
            guess_mode = gr.Checkbox(label='Guess Mode', value=False)
            detect_resolution = gr.Slider(label="OpenPose Resolution", minimum=128, maximum=1024, value=512, step=1)
            ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=100, step=1, visible=False)
            scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=4294967294, value=11, step=1)
            eta = gr.Number(label="eta (DDIM)", value=0.0)
            a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed, masterpiece, 8k, canon, film still, white background,')
            n_prompt = gr.Textbox(label="Negative Prompt",
                                    value='breast,naked,(teeth:1.3),headscarf, hat, sketch by Bad_Artist, (worst quality, low quality:1.4), (bad anatomy), watermark, signature, text, logo,contact, (extra limbs),Six fingers,Low quality fingers,monochrome,(((missing arms))),(((missing legs))), (((extra arms))),(((extra legs))),less fingers,lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, (depth of field, bokeh, blurry:1.4),blurry background,bandages,')

        hlg_ips = [hlg_prompt, hlg_input_image, category, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
        hlg_run.click(fn=hlg_process, inputs=hlg_ips, outputs=[result_gallery])

        for tab_elem in list(attribute_dict.values()):
            lle_ips = [lle_prompt, lle_input_image, category, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, current_tab[tab_elem], selection_mode]
            lle_run[tab_elem].click(fn=lle_process, inputs=lle_ips, outputs=[result_gallery])
        send2llg.click(fn=result2input, inputs=result_gallery, outputs=lle_input_image)
    return app


hfddm_block = create_hfddm()



demo = gr.Blocks(
        title="AI Fashion Design",
        theme=gr.themes.Monochrome(secondary_hue="orange", neutral_hue="gray"),
    ).queue()

with demo:    
    gr.Markdown(
        """
        # <div style="color: white">AI Fashion Design</div>
        """)

    with gr.Tab("Fashion Design👗"):
        hfddm_block.render()
        
demo.launch(
    server_name="0.0.0.0",
    server_port=7860)