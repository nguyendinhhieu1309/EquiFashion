import torch
import sys
import argparse
import numpy as np
from pathlib import Path


from .segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from .utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point
import torch.nn.functional as F
import torch.nn as nn

from matplotlib import pyplot as plt
import cv2
import json
from PIL import Image


class MaskDetector:
    def __init__(self,
                 sam_model_path="/model/sam_vit_h_4b8939.pth",
                 sam_model_type="vit_h",
                 device="cuda"):

        self.sam_model_path = sam_model_path
        self.sam_model_type = sam_model_type
        self.device = device
        self.dilate_kernel_size = 15
        self.predictor = None
            
    def onload(self):
        
        self.predictor = SamPredictor(sam_model_registry[self.sam_model_type](checkpoint=self.sam_model_path).to(self.device))

    def offload(self):
        del self.predictor
        torch.cuda.empty_cache()
        
        self.predictor = None

    def __call__(self, input_img, keypoints, category, attribute, draw_masks=False,sam_mode=True):
        if self.predictor is None:
            self.onload()
        # img = load_img_to_array(input_img)
        img = input_img # np.array
        self.predictor.set_image(img)
        candidate = keypoints['candidate']
        subset = keypoints['subset']

        # save mask
        out_dir = "./mask_result/"
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)

        if attribute == "A1" or attribute == "A5":
            if category in ["Dress", "Jumpsuit", "Coat", "Pant", "Skirt", "Shirt"]:
                x1, y3, x2, y4 = 128, 285, 378, 638
                # x-axis should be based on left/right elbows
                for i in range(18):
                    for n in range(len(subset)):
                        if i == 3:
                            index = int(subset[n][i]) # Determine coordinates by index i
                            if index == -1:
                                continue
                            x1, y1 = candidate[index][0:2] # left elbow
                        if i == 6:
                            index = int(subset[n][i]) # Determine coordinates by index i
                            if index == -1:
                                continue
                            x2, y2 = candidate[index][0:2] # right elbow
                        if i == 8:
                            index = int(subset[n][i]) # Determine coordinates by index i
                            if index == -1:
                                continue
                            x3, y3 = candidate[index][0:2] # left hip
                        if i == 13:
                            index = int(subset[n][i]) # Determine coordinates by index i
                            if index == -1:
                                continue
                            x4, y4 = candidate[index][0:2] # right ankle

                boxes =[[int(x1)-100,int(y3),int(x2)+100,int(y4)+60]]  # Proposed bbox; tune by dataset characteristics
                # boxes =[[int(x1)-100,int(y3)+120,int(x2)+100,int(y4)+40]] 
            elif category in ["Blouse", "Sweater", "Shirt"]:
                x1, y1,x2, y2=188, 205, 317, 322
                for i in range(18):
                    for n in range(len(subset)):
                        if i == 8:
                            index = int(subset[n][i]) # Determine coordinates by index i
                            if index == -1:
                                continue
                            x1, y1 = candidate[index][0:2] # left elbow
                        if i == 11:
                            index = int(subset[n][i]) # Determine coordinates by index i
                            if index == -1:
                                continue
                            x2, y2 = candidate[index][0:2] # right elbow
                boxes =[[int(x1)-20,int(y1)-60,int(x2)+20,int(y2)+20]]  # Proposed bbox
                # boxes =[[int(x1)-30,int(y1)-60,int(x2)+40,int(y2)+20]]
            

        elif attribute in ["A2", "A3"]:
            x1, y1,x2, y2 = 110, 42, 187, 338
            x3, y3,x4, y4=307, 44, 387, 325
            for i in range(18):
                for n in range(len(subset)):
                    if i == 2:
                        index = int(subset[n][i]) # Determine coordinates by index i
                        if index == -1:
                            continue
                        x1, y1 = candidate[index][0:2] # left elbow
                    if i == 4:
                        index = int(subset[n][i]) # Determine coordinates by index i
                        if index == -1:
                            continue
                        x2, y2 = candidate[index][0:2] # right elbow
                    if i == 5:
                        index = int(subset[n][i]) # Determine coordinates by index i
                        if index == -1:
                            continue
                        x3, y3 = candidate[index][0:2] # left hip
                    if i == 7:
                        index = int(subset[n][i]) # Determine coordinates by index i
                        if index == -1:
                            continue
                        x4, y4 = candidate[index][0:2] #  
            boxes =[[int(x1)-80,int(y1)-30,int(x2)+20,int(y2)+40],[int(x3)-15,int(y3)-30,int(x4)+40,int(y4)+40]] # boxes = [[110, 42, 187, 338], [307, 44, 387, 325]]
            # boxes =[[int(x1)-80,int(y1)+70,int(x2),int(y2)-50],[int(x3)+20,int(y3)+60,int(x4)+40,int(y4)-50]]
            
        elif attribute == "A4":
            x1, y1 = 190, 35
            for i in range(18):
                for n in range(len(subset)):
                    if i == 1:
                        index = int(subset[n][i]) # Determine coordinates by index i
                        if index == -1:
                            continue
                        x1, y1 = candidate[index][0:2] # left elbow
            boxes =[[int(x1)-60,int(y1)-40,int(x1)+60,int(y1)+40]]  # Expand around neck position
            # boxes =[[int(x1)-80,int(y1)-40,int(x1)+80,int(y1)+40]]
       
        if sam_mode: 
            first_mask = True # Initialize flag to mark first mask
            for idx in range(len(boxes)):
                masks, _, _ = self.predictor.predict(box=np.array([boxes[idx]]))  # Prediction call
                masks = masks.astype(np.uint8) * 255
                if self.dilate_kernel_size is not None:
                    masks = [dilate_mask(mask, self.dilate_kernel_size) for mask in masks]
                    if draw_masks:
                        for ii, m in enumerate(masks):
                            mask_p = out_dir / f"maskss_{idx}_{ii}.jpg"
                            save_array_to_img(m, mask_p)

                    mask = masks[-1]  # Use last mask
                    # If this is the first mask, initialize merged_mask directly
                    if first_mask:
                        merged_mask = mask.copy()
                        first_mask = False
                    else:
                        # Otherwise merge current mask into merged_mask
                        merged_mask = np.maximum(merged_mask, mask)
        else:
            # Use coordinate rectangles directly for all regions
            mask = np.zeros_like(img[:,:,0])
            for box in boxes:
                x1, y1, x2, y2 = box
                mask[y1:y2, x1:x2] = 255
            merged_mask = mask
       
        # Draw all bounding boxes on the original image
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle indicates bbox

        # merged_mask_path = out_dir + "merged_mask.jpg"
        # save_array_to_img(merged_mask, merged_mask_path)

        # Save image with bounding boxes
        # bbox_image_path = out_dir + "bbox_image.jpg"
        # save_array_to_img(img, bbox_image_path)

        alpha_mask_img = Image.fromarray(merged_mask, mode='L')
        
        self.offload()

        return alpha_mask_img
