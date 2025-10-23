

<div align="center">
<h1>CRACK-Net: Hierarchical Dual-Pyramid FeatureFusion with Position-Aware Attention for PavementDefect Detection</h1>
<div>

<img src="Code\fig\fig1.png" alt="fig1"  />

# The core innovation points of the paper

1.A double-layer feature pyramid architecture that decomposes road defect detection into multi-scale subtasks. 

2.A novel pluggable position-aware attention module that integrates coordinate-spatial modules. 

3.A multi-branch deep network that integrates the local feature extraction capabilities of CNN and the global contextual modeling of Transformer.



# Updates

**2025-07-26** Code released!



# TODO List
- [x] Code and datasets
- [x] Figure of results
- [x] Release Code
- [ ] Inference code and checkpoint



# Setup
Download GRDDC2020 datasets

Prepare Conda environment

```python
conda create -n DualPyramid python=3.10 -y && conda activate DualPyramid
```

Install remaining dependencies
```
pip install -r requirements.txt
```
Preprocess

```
python tools/data_preprocess.py --input_dir ./raw_data --output_dir ./processed_data
```



# Train

```
python train.py --models/Dualpyramid-Net.yaml --weights yolov5s.pt
```



# Inference
```
python detect.py --source test_images/ --runs/train/Dualpyramid-Net/weights/best.pt
```







# Core Module Description

##  Dual Pyramid Architecture

```
├── models/Dualpyramid-Net.yaml
│ ├── Neck-1 # Feature Pyramid Implementation
│ ├── C3CASM # Coordinate-Spatial Attention
│ ├── Neck-2
│ └── cat # CNN-Transformer Fusion
```

**Neck 1** : Processes features at the 20 - 40 scale
**Neck 2** : Performs reverse fusion for features at the 40 - 20 scale

**C3CASM Mechanism**

![fig2](Code\fig\fig2.png)





**CAT Module**

![fig3](Code\fig\fig3.png)

# Result

![fig4](Code\fig\fig4.png)



#  Citation
If you find this codebase useful for your research, please cite as follows:
```
@article{wu2025dualpyramid,
  title={CRACK-Net: Hierarchical Dual-Pyramid FeatureFusion with Position-Aware Attention for PavementDefect Detection},
  author={Wu, Hao and Zhang, Kaibing and Li, Xiaoyan et al.},
  journal={Computer Vision and Image Understanding},
  year={2025}
}
```
