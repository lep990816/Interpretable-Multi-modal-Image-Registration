# Interpretable Multi-modal Image Registration Network Based on Disentangled Convolutional Sparse Coding
[paper link](https://ieeexplore.ieee.org/abstract/document/10034541)

## 1.Requirements
- Python >= 3.7
- Pytorch >= 0.4.1
- opencv-python
- korina
- matplotlib

## 2.Dataset
For homography estimation, the training data and testing data is from the DPDN, CAVE, RGB-NIR Sence datasets. For deformation registration, the training data and testing data is from Atlas and RIRE datasets. Please download the above data before running the code.

## 3.Training
### Pre-train for AG-Net
AG-Net is trained with fully registered multi-modal images for each dataset. We provide the pre-trained model for RIRE dataset. To train for other dataset, run the following command.
```
python train_AGNet.py
```
### Train for InMIR-Net
For rigid registration task, run the following command.
```
python main_homography.py
```

For non-rigid registration task, run the following command.
```
python main_deformation.py
```

## 4.Testing
### Test for homography estimation
Run the following command.
```
python test.py
```


## 5.Citation
If you find our paper or code useful for your research, please cite:
```
@article{deng2023interpretable,
  title={Interpretable Multi-modal Image Registration Network Based on Disentangled Convolutional Sparse Coding},
  author={Deng, Xin and Liu, Enpeng and Li, Shengxi and Duan, Yiping and Xu, Mai},
  journal={IEEE Transactions on Image Processing},
  volume={32},
  pages={1078--1091},
  year={2023},
  publisher={IEEE}
}
```
