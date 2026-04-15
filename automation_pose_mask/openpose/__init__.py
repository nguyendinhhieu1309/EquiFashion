# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import model
from . import util
from .body import Body
from .hand import Hand
from tqdm import tqdm



class OpenposeDetector:
    def __init__(self,
                 body_model_path="EquiFashion/utils/checkpoints/body_pose_model.pth",
                 hand_model_path="EquiFashion/utils/checkpoints/hand_pose_model.pth"):
        self.body_model_path = body_model_path
        self.hand_model_path = hand_model_path
        self.body_estimation = None
        self.hand_estimation = None
        
    def onload(self):
        self.body_estimation = Body(self.body_model_path)
        self.hand_estimation = Hand(self.hand_model_path)

    def offload(self):
        del self.body_estimation
        del self.hand_estimation
        torch.cuda.empty_cache()
        self.body_estimation = None
        self.hand_estimation = None
    
    def __call__(self, oriImg, hand=True):
        if self.body_estimation is None:
            self.onload()
        oriImg = oriImg[:, :, ::-1].copy()
        with torch.no_grad():
            # Body pose estimation
            candidate, subset = self.body_estimation(oriImg)
            canvas = np.zeros_like(oriImg)
            # canvas = copy.deepcopy(oriImg)
            canvas = util.draw_bodypose(canvas, candidate, subset)

            # Hand pose estimation
            hands_list = util.handDetect(candidate, subset, oriImg)
            all_hand_peaks = []
            for x, y, w, is_left in hands_list:
                peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :])
                peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                all_hand_peaks.append(peaks.tolist())

            canvas = util.draw_handpose(canvas, [np.array(hand_peaks) for hand_peaks in all_hand_peaks])
            self.offload()
            return canvas, dict(candidate=candidate.tolist(), subset=subset.tolist())
