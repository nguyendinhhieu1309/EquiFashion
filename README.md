# EquiFashion

**EquiFashion: Hybrid GAN-Diffusion Balancing Diversity-Fidelity for Fashion Design Generation**

Nguyen Dinh Hieu (ORCID: 0009-0002-6683-8036), Tran Minh Khuong, Ngo Dinh Hoang Minh, Ta Uyen Nhi, Phan Duy Hung (ORCID: 0000-0002-6033-6484)  
FPT University, Hanoi, Vietnam  
`hieundhe180318@fpt.edu.vn`, `khuongtmhe180089@fpt.edu.vn`,`minhndhhe182227@fpt.edu.vn`, `nhi.nichole.1907@gmail.com`, `hungpd2@fe.edu.vn`

Code and model assets: [Hugging Face - NguyenDinhHieu/EquiFashionModel](https://huggingface.co/NguyenDinhHieu/EquiFashionModel)

---

## Overview

EquiFashion is a hybrid GAN-Diffusion framework for controllable fashion image generation.  
The core goal is to balance:

- **Diversity** (stylized exploration via GAN-based latent ideation)
- **Fidelity** (photorealistic and stable refinement via latent diffusion)
- **Semantic precision** (attribute-to-region consistency)
- **Physical plausibility** (pose-conditioned generation)

EquiFashion is designed for fashion design workflows where outputs must be both creative and structurally coherent under complex prompts.

---

## Abstract

This project presents EquiFashion, a hybrid GAN-Diffusion framework that bridges the long-standing gap between stylistic diversity and photorealistic fidelity in generative fashion design. Unlike prior approaches that favor either creativity or realism, EquiFashion unifies both through a dual-stage process: GAN-driven ideation for multimodal exploration and diffusion-based refinement for detail reconstruction. A semantic-bundled attention and structural consensus mechanism improves localized attribute alignment, while pose-conditioned latent diffusion preserves human-centric geometry.  

To support robust evaluation, the project introduces EquiFashion-DB, a multimodal dataset (image, text, sketch, pose, fabric) with controlled text noise. Across CM-Fashion, HieraFashion, and EquiFashion-DB, EquiFashion reports strong improvements in fidelity and semantic alignment.

---

## Key Contributions

- **Hybrid GAN-Diffusion architecture** for diversity-fidelity balancing.
- **Semantic-Bundled Cross-Attention** to reduce attribute leakage.
- **Structural Semantic Consensus** for part-level text-image alignment.
- **Pose-conditioned latent refinement** for realistic body-aware synthesis.
- **EquiFashion-DB benchmark** for robust multimodal evaluation.

---

## Pipeline

**EquiFashion pipeline overview.** The framework includes:
1) multimodal conditioning parsing (text, pose, sketch/fabric cues),
2) GAN-based latent ideation for style exploration,
3) diffusion-based refinement for high-fidelity reconstruction,
4) structural semantic consensus for part-level alignment, and
5) semantic-bundled attention for localized attribute consistency.

<img width="612" height="747" alt="image" src="https://github.com/user-attachments/assets/87ee94df-1a91-47a3-b77f-63522b3c6dda" />


---

## ⚙️ Framework and Environment Setup

This project uses the following core libraries:

| Framework | Version |
|-----------|---------|
| ![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1%2Bcu113-ee4c2c?logo=pytorch&logoColor=white) | 1.12.1 + cu113 |
| ![TorchVision](https://img.shields.io/badge/TorchVision-0.13.1-5C3EE8?logo=pytorch&logoColor=white) | 0.13.1 |
| ![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-1.5.0-792EE5?logo=lightning&logoColor=white) | 1.5.0 |
| ![Python](https://img.shields.io/badge/Python-3.8.5-3776ab?logo=python&logoColor=white) | 3.8.5 |
| ![CUDA](https://img.shields.io/badge/CUDA-11.3-76B900?logo=nvidia&logoColor=white) | 11.3 |
| ![Transformers](https://img.shields.io/badge/Transformers-4.29.2-FFCA28?logo=huggingface&logoColor=black) | 4.29.2 |
| ![OpenCV](https://img.shields.io/badge/OpenCV-4.6.0-5C3EE8?logo=opencv&logoColor=white) | 4.6.0 |
| ![Gradio](https://img.shields.io/badge/Gradio-4.29.0-F97316?logo=gradio&logoColor=white) | 4.29.0 |

---

## Repository Structure

```text
EquiFashion/
├─ app.py                             # Gradio app entry (HF/serving-oriented)
├─ gradio_EqF.py                      # Alternative Gradio demo entry
├─ cldm/                              # Main model logic (ControlLDM, losses, samplers)
│  ├─ cldm.py
│  ├─ semantic_attention.py
│  ├─ structural_consensus.py
│  └─ modules/gan_components.py
├─ scripts/train/
│  ├─ train_cldm.py                   # Training script
│  └─ my_dataset.py                   # Dataset loader
├─ utils/
│  ├─ config.py                       # Paths and dictionaries
│  └─ configs/cldm_v2.yaml            # Model config
├─ automation_pose_mask/              # OpenPose + mask utilities
├─ requirements.txt
└─ environment.yaml
```

---

## Environment Setup

### Option A: Conda (recommended for reproducibility)

```bash
conda env create -f environment.yaml
conda activate EquiFashion
```

### Option B: pip

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## Model Weights

All checkpoints and pretrained assets are hosted on Hugging Face:

- [https://huggingface.co/NguyenDinhHieu/EquiFashionModel](https://huggingface.co/NguyenDinhHieu/EquiFashionModel)

After downloading, place required files under your checkpoint directory (default pattern used in code):

- `checkpoints/body_pose_model.pth`
- `checkpoints/hand_pose_model.pth`
- `checkpoints/sam_vit_h_4b8939.pth`
- `checkpoints/EqF_100epochs.ckpt` (or target training/inference checkpoint)

If needed, adjust paths in `utils/config.py`.

---

## Data Preparation

Default dataset root in code:

```python
dataset_root = "dataset/EquiFashion_DB/"
```

Expected training annotations include prompt text and control hints (pose, etc.) as used by `scripts/train/my_dataset.py`.

Suggested layout:

```text
dataset/
└─ EquiFashion_DB/
   ├─ train.json (or train_pose.json)
   ├─ train/ or images/ or gt/
   └─ train_pose/...
```

---

## Training

Run:

```bash
python scripts/train/train_cldm.py
```

Main training script:

- `scripts/train/train_cldm.py`

Important notes:

- Update `CUDA_VISIBLE_DEVICES`, batch size, precision, and checkpoint path for your GPU setup.
- For lower VRAM, consider mixed precision and memory-saving settings.

---

## Inference / Demo

### App entry

```bash
python app.py
```

### Alternative demo entry

```bash
python gradio_EqF.py
```

The app provides draft generation and attribute editing workflows with pose/mask guidance.

---

## Citation

If you use this code, models, or dataset, please cite:

```bibtex
@misc{equifashion2025,
  title={EquiFashion: Hybrid GAN–Diffusion Balancing Diversity–Fidelity for Fashion Design Generation},
  author={Tran Minh Khuong and Nguyen Dinh Hieu and Phan Duy Hung},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  year={2026},
  organization={FPT University, Hanoi}
}
}
```

---

## Contact

For questions and collaboration:

- `hieundhe180318@fpt.edu.vn`
- `khuongtmhe180089@fpt.edu.vn`
- `hungpd2@fe.edu.vn`

