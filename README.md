# Interpretable Multi-modal Image Registration Network Based on Disentangled Convolutional Sparse Coding

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
python train_single.py
```
### Train for InMIR-Net
For rigid registration task, run the following command.
```
python train_homography.py
```

For non-rigid registration task, run the following command.
```
python train_deformation.py
```
