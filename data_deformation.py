from torch.utils.data import Dataset
from pathlib import Path
import pdb,numpy
import os,glob
import torch
from torchvision import transforms
from pathlib import Path
import random,cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
import shutil
import os.path as osp
import matplotlib.pyplot as plt
import seaborn as sns
from Model import losses
import torch.nn as nn

def B_spline_function(flag,t):
    if (flag == 0):
        return (1 - t*t*t + 3 * t*t - 3 * t) / 6.0
    elif (flag == 1):
        return (4 + 3 * t*t*t - 6 * t*t) / 6.0
    elif (flag == 2):
        return (1 - 3 * t*t*t + 3 * t*t + 3 * t) / 6.0
    elif (flag == 3):
        return (t*t*t / 6.0)
    else:
        return 0.0

def B_spline_transform(srcImg):
    delta_x = 32
    delta_y = 32
    H,W,C = srcImg.shape
    dsrImg =  np.zeros((H,W,C),dtype='uint8')
    grid_rows = (int)(H / delta_x) + 1 + 3
    grid_cols = (int)(W / delta_y) + 1 + 3
    noiseMat = np.zeros((grid_rows,grid_cols,2))
    offset = np.zeros((H,W,2))

    for i in range(grid_rows):
        for j in range(grid_cols):
            for k in range(2):
                noiseMat[i,j,k] = random.randint(-10,10)

    #B_spline 变形
    for x in range(H):
        for y in range(W):

            i = int(x / delta_x)  #int
            j = int(y / delta_y)

            u = float(x / delta_x - i)  #float
            v = float(y / delta_y - j)

            px = [0 for n in range(4)]
            py = [0 for n in range(4)]

            for k in range(4):
                px[k] = float(B_spline_function(k,u))  #float
                py[k] = float(B_spline_function(k,v))
            
            Tx = 0
            Ty = 0
            for m in range(4):
                for n in range(4):
                    control_point_x = int(i + m)  #int
                    control_point_y = int(j + n)
                    temp = float(py[n] * px[m])

                    Tx += temp * noiseMat[control_point_x,control_point_y,0]
                    Ty += temp * noiseMat[control_point_x,control_point_y,1]
            
            offset[x,y,0] = Tx
            offset[x,y,1] = Ty
    
    #反向映射，双线性插值
    for row in range(H):
        for col in range(W):

            src_x = row + offset[row,col,0]  #float
            src_y = col + offset[row,col,1]
            x1 = int(src_x)
            y1 = int(src_y)
            x2 = int(x1 + 1)
            y2 = int(y1 + 1)

            if (x1<0 or x1>(H - 2) or y1<0 or y1>(W - 2)):
                dsrImg[row,col,0] = 0
                dsrImg[row,col,1] = 0
                dsrImg[row,col,2] = 0
            else:
                pointa = []
                pointb = []
                pointc = []
                pointd = []

                pointa = srcImg[x1,y1,:]
                pointb = srcImg[x2,y1,:]
                pointc = srcImg[x1,y2,:]
                pointd = srcImg[x2,y2,:]

                B = (int)((x2 - src_x)*(y2 - src_y)*pointa[0] - (x1 - src_x)*(y2 - src_y)*pointb[0] - (x2 - src_x)*(y1 - src_y)*pointc[0] + (x1 - src_x)*(y1 - src_y)*pointd[0])
                G = (int)((x2 - src_x)*(y2 - src_y)*pointa[1] - (x1 - src_x)*(y2 - src_y)*pointb[1] - (x2 - src_x)*(y1 - src_y)*pointc[1] + (x1 - src_x)*(y1 - src_y)*pointd[1])
                R = (int)((x2 - src_x)*(y2 - src_y)*pointa[2] - (x1 - src_x)*(y2 - src_y)*pointb[2] - (x2 - src_x)*(y1 - src_y)*pointc[2] + (x1 - src_x)*(y1 - src_y)*pointd[2])

                dsrImg[row,col,0] = B
                dsrImg[row,col,1] = G
                dsrImg[row,col,2] = R

                # dsrImg.dtype = np.

    return dsrImg
    
def generator_affine_param(random_t=0.3,random_s=0.3,random_alpha = 1/8,random_tps=0.4,to_dict = False):
    alpha = (np.random.rand(1)-0.5) * 2 * np.pi * random_alpha
    theta = np.random.rand(6)

    theta[[2, 5]] = (theta[[2, 5]] - 0.5) * 2 * random_t
    theta[0] = (1 + (theta[0] - 0.5) * 2 * random_s) * np.cos(alpha)
    theta[1] = (1 + (theta[1] - 0.5) * 2 * random_s) * (-np.sin(alpha))
    theta[3] = (1 + (theta[3] - 0.5) * 2 * random_s) * np.sin(alpha)
    theta[4] = (1 + (theta[4] - 0.5) * 2 * random_s) * np.cos(alpha)
    theta = theta.reshape(2, 3)

    if to_dict:
        temp = theta.reshape(6)
        theta = {}
        theta['p0'] = temp[0]
        theta['p1'] = temp[1]
        theta['p2'] = temp[2]
        theta['p4'] = temp[4]
        theta['p5'] = temp[5]

    return theta

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

def noisy(noise_typ, img):
    if noise_typ == "gauss":
        mean = 0
        var = 10
        
        sigma = var ** 0.5 * 10
        gaussian = np.random.normal(mean, sigma, (512,512)) #  np.zeros((224, 224), np.float32)

        noisy_image = np.zeros(img.shape, np.float32)

        if len(img.shape) == 2:
            noisy_image = img + gaussian
        else:
            noisy_image[:, :, 0] = img[:, :, 0] + gaussian
            noisy_image[:, :, 1] = img[:, :, 1] + gaussian
            noisy_image[:, :, 2] = img[:, :, 2] + gaussian

        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        noisy_image = noisy_image.astype(np.uint8)
        return noisy_image

    elif noise_typ == "s&p":
        prob=0.1
        output = np.zeros(img.shape,np.uint8)
        thres = 1 - prob 
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = img[i][j]
        return output

    elif noise_typ == "compress":
        # img = cv2.imread(img,0)
        cv2.imwrite('tmp.jpg', img,  encode_param)
        output = cv2.imread('tmp.jpg')
        os.remove('tmp.jpg')
        return output

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)

class ImageFolder(Dataset):
    def __init__(self, root, transform=None,patch_size=(256,256), split='train'):
        splitdir = Path(root) / split 

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        splitdir_rgb = splitdir / "modal1"
        splitdir_mutispectral = splitdir / "modal2"
        splitdir_npy = splitdir / "npy"

        self.split = split
        self.rgb_list = sorted(glob.glob(os.path.join(splitdir_rgb,"*")))
        self.mutispectral_list = sorted(glob.glob(os.path.join(splitdir_mutispectral, "*")))
        self.npy_list = sorted(glob.glob(os.path.join(splitdir_npy, "*")))
        self.STN = SpatialTransformer(size=[256,256],mode='bilinear')
        self.patch_size = patch_size
        self.transform = transform
    
    def __getitem__(self, index):

        img1 = cv2.imread(self.rgb_list[index])
        img1_ = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1_, cv2.COLOR_RGB2GRAY)
        img2 = cv2.imread(self.mutispectral_list[index])
        img2_ = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2_, cv2.COLOR_RGB2GRAY)

        H, W = img1.shape
        if self.patch_size[0]==H:
            startH = 0
            startW = 0
        else:
            startH = random.randint(0,H-self.patch_size[0]-1)
            startW = random.randint(0,W-self.patch_size[1]-1)
        if self.split == 'train':
            number = random.randint(0,99)
        else:
            number = index
        fai = numpy.load(self.npy_list[number], mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        # fai = fai * 3
        img1_input = self.STN(self.transform(img1).unsqueeze(0).float(),fai)

        smooth_loss_gt = losses.gradient_loss(torch.tensor(fai))
       
        return img1_input , self.transform(img2), self.transform(img1)
        
    def __len__(self):
        return len(self.rgb_list)
