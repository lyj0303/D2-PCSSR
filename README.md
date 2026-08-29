# D2-PCSSR
Data-Efficient PolSAR Crop Parameter Retrieval via Decoupled Deep Collaborative Regression with Spectral Sequence Guidance
---

## 👀Introduction

This repository contains the code for our paper `D2-PCSSR: Data-Efficient PolSAR Crop Parameter Retrieval via Decoupled Deep Collaborative Regression with Spectral Sequence Guidance`. 

![](figure/方法总体流程图-改1.png)
## 📂 Dataset Preparation

We evaluated our method on two crop datasets collected in Hengshui, China:
*   **Winter Wheat:** LAI retrieval across three key growth stages (Booting, Heading, Grain-Filling).
*   **Summer Maize:** Plant Height retrieval across three key growth stages (Jointing, Tasseling, Grain-Filling).

*Note: Due to data sharing policies, the original PolSAR images are not publicly available. However, we provide the pre-processed feature vectors and the pseudo-label generation pipeline for full reproducibility.*

Please organize your data as follows:

## 🚀 Quick Start
1. Create environment.
'''
conda create -n sigma python=3.9
conda activate sigma
'''

2. Install all dependencies. Install pytorch, cuda and cudnn, then install other dependencies via:
'''
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt
'''

3. Install Mamba
'''
cd models/encoders/selective_scan && pip install . && cd ../../..
'''
