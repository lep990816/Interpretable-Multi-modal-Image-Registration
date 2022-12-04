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
# import kornia
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
    # theta[0] = 1
    theta[0] = (1 + (theta[0] - 0.5) * 2 * random_s) * np.cos(alpha)
    theta[1] = (1 + (theta[1] - 0.5) * 2 * random_s) * (-np.sin(alpha))
    # theta[1] = 0
    # theta[3] = 0
    # theta[4] = 1
    theta[3] = (1 + (theta[3] - 0.5) * 2 * random_s) * np.sin(alpha)
    theta[4] = (1 + (theta[4] - 0.5) * 2 * random_s) * np.cos(alpha)
    theta = theta.reshape(2, 3)

    if to_dict:
        temp = theta.reshape(6)
        theta = {}
        theta['p0'] = temp[0]
        theta['p1'] = temp[1]
        # theta['p1'] = 0
        theta['p2'] = temp[2]
        # # theta['p3'] = temp[3]
        # theta['p3'] = 0
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
        # pdb.set_trace()

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        # offset2 = np.zeros((256,256,1))
        # pdb.set_trace()
        # offset2[:,:,0] = (new_locs[0,:,:,0].cpu().numpy()**2+new_locs[0,:,:,1].cpu().numpy()**2)**0.5

        return F.grid_sample(src, new_locs, mode=self.mode)

class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories: ::
        - rootdir/
            - train/
                -left/
                    - 0.png
                    - 1.png
                -right/
            - test/
                -left/
                    - 0.png
                    - 1.png
                -right/
    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """
    def __init__(self, root, transform=None,patch_size=(256,256), split='train'):
        splitdir = Path(root) / split  # 相当于osp.join

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        splitdir_rgb = splitdir / "modal1"
        splitdir_mutispectral = splitdir / "modal2"
        splitdir_npy = splitdir / "npy"
        # splitdir_rgb_origin = splitdir / "RGB_origin"
        
        # print(splitdir_right_disp)
        self.split = split

        self.rgb_list = sorted(glob.glob(os.path.join(splitdir_rgb,"*")))
        self.mutispectral_list = sorted(glob.glob(os.path.join(splitdir_mutispectral, "*")))
        self.npy_list = sorted(glob.glob(os.path.join(splitdir_npy, "*")))
        self.STN = SpatialTransformer(size=[256,256],mode='bilinear')
        # pdb.set_trace()
        # self.rgb_origin_list = sorted(glob.glob(os.path.join(splitdir_rgb_origin,"*")))
        # self.right_disp_list = sorted(glob.glob(os.path.join(splitdir_right_disp,"*")))

        self.patch_size = patch_size
        #只保留了ToTensor
        self.transform = transform
    

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        # img1 = Image.open(self.left_list[index]).convert('RGB')
        # img2 = Image.open(self.right_list[index]).convert('RGB')
        # if os.path.basename(self.left_list[index]) != os.path.basename(self.right_list[index]):
        #     print(self.left_list[index])
        #     raise ValueError("cannot compare pictures.")
        ##
        # img1 = cv2.imread(self.left_list[index])
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # img2 = cv2.imread(self.right_list[index])
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # img3 = cv2.imread(self.left_disp_list[index])
        # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        # img4 = cv2.imread(self.right_disp_list[index])
        # img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
        # #random cut for pair
        # H, W, _ = img1.shape
        # # randint是闭区间
        # print(H)
        # print(W)
        # print(self.patch_size)
        img1 = cv2.imread(self.rgb_list[index])
        # img1 = noisy('compress',img1)
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

        # img1 = img1[startH:(startH + self.patch_size[0]), startW:(startW + self.patch_size[1])]
        # img1_ = img1_[startH:(startH + self.patch_size[0]), startW:(startW + self.patch_size[1])]
        # img2 = img2[startH:(startH + self.patch_size[0]), startW:(startW + self.patch_size[1])]
        # img2_ = img2_[startH:(startH + self.patch_size[0]), startW:(startW + self.patch_size[1])]
        if self.split == 'train':
            number = random.randint(0,99)
        else:
            number = index
        fai = numpy.load(self.npy_list[number], mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        # fai = fai * 2.5
        img1_input = self.STN(self.transform(img1).unsqueeze(0).float(),fai)

        smooth_loss_gt = losses.gradient_loss(torch.tensor(fai))
        # print(smooth_loss_gt)
        # pdb.set_trace()

        # img3 = cv2.imread(self.rgb_origin_list[index])
        # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        # img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)
        # img4 = cv2.imread(self.right_disp_list[index])
        # img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
        # img4 = cv2.cvtColor(img4, cv2.COLOR_RGB2GRAY)
        # print(img1.shape)
        # img1 = img1.reshape(832,1024,1)
        # img2 = img2.reshape(832, 1024, 1)
        # img3 = img3.reshape(832, 1024, 1)
        # img4 = img4.reshape(832, 1024, 1)
        # img1 = img1.reshape(1,-1,-1)
        # img2 = img2.unsqueeze(0)
        # img3 = img3.unsqueeze(0)
        # img4 = img4.unsqueeze(0)
        #random cut for pair
        
        # img1 = img1.reshape(H,W,1)
        # img2 = img2.reshape(H,W,1)
        # img1_input = B_spline_transform(img1_)
        # img1_input_grey = cv2.cvtColor(img1_input, cv2.COLOR_RGB2GRAY)
        return img1_input , self.transform(img2), self.transform(img1)
        # img3 = img3.reshape(H,W,1)
        
        # print(H)
        # print(W)
        # print(self.patch_size)


        # # pdb.set_trace()
        img2 = img2[startH:(startH + self.patch_size[0]), startW:(startW + self.patch_size[1])]
        img2_ = img2_[startH:(startH + self.patch_size[0]), startW:(startW + self.patch_size[1])]
        # img3 = img3[startH:(startH + self.patch_size[0]), startW:(startW + self.patch_size[1])]
        # img4 = img4[startH:(startH + self.patch_size[0]), startW:(startW + self.patch_size[1])]

        # corners = torch.tensor([
        #     [x, y],
        #     [x + patch_size, y],
        #     [x + patch_size, y + patch_size],
        #     [x, y + patch_size],
        # ], dtype=torch.float)

        # delta = torch.tensor([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        # corners = corners.unsqueeze(0)
        # delta = delta.unsqueeze(0)
        # # print(corners.type())
        # # print(delta.type())
        # corners_hat = corners + delta
        # h = kornia.get_perspective_transform(corners, corners_hat)
        # h_inv = torch.inverse(h).numpy().squeeze(0)
        # # pdb.set_trace()
        # h = h.numpy().squeeze()
        # # print(delta)
        # # print(h_inv)
        # # pdb.set_trace()
        # output = cv2.warpPerspective(img1_, h_inv, (480,480))

        # GT = cv2.warpPerspective(output, h, (480,480))
        # output_gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
        # GT_gray = cv2.cvtColor(GT, cv2.COLOR_RGB2GRAY)

        # # img5 = Image.fromarray(img1_)
        # # img5 = img5.convert('L')
        # # img5 = transforms.ToTensor()(img5)
        # # theta = generator_affine_param()
        # # theta = torch.from_numpy(theta.astype(np.float32))
        # # N, C, W, H = img5.unsqueeze(0).size()
        # # grid = F.affine_grid(theta.unsqueeze(0), img5.unsqueeze(0).size())
        # # output = F.grid_sample(img5.unsqueeze(0), grid)
        # # new_img5 = output[0]

        # ###
        # # H_list = get_H(img1,img2)
        # # ##
        # # if H_list[0]==None:
        # #     print(self.left_list[index])
        # #     print(self.right_list[index])
        # #     #raise ValueError("None!!H_matrix")
        # #     # 只有ToTensor
        # #     if self.transform:
        # #         return self.transform(img1), self.transform(img2),self.transform(img3),self.transform(img4) # ,H_list[1],H_list[2],H_list[3]
        # #     return img1, img2, img3, img4 # ,H_list[1],H_list[2],H_list[3]

        # # #只有ToTensor
        # # print(self.transform)
        # # print(img1.size)
        # # pdb.set_trace()
        # return self.transform(img2),self.transform(output_gray),self.transform(GT_gray),output,img1_,self.rgb_list[index]#,self.transform(img4) #,H_list[1],H_list[2],H_list[3]
        # return img1,img2, img3, img4 #,H_list[1],H_list[2],H_list[3]
        # print(self.transform)
       
        
    def __len__(self):
        return len(self.rgb_list)
