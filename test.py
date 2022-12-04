import os,glob,imageio
import time
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import os.path as osp
import torch.nn.functional as F
from torch.nn import MSELoss
from network_deformation import InMIR,ransformer
from network_single import AGNet
from torchvision import transforms
from torch.utils.data import DataLoader
from data_deformation import ImageFolder
# from data_affine import ImageFolder
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pdb
from Model import losses
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
# import kornia

out_root_path = "pic_RIRE_big"
if not os.path.exists(out_root_path):
    print("not ex")
    os.system("mkdir "+out_root_path)

plt_delta_path = osp.join(out_root_path,"heat_delta")
plt_path = osp.join(out_root_path,"heat_gt")
plt_gt_path = osp.join(out_root_path,"heat")
plt_heat_path = osp.join(out_root_path,"heat_pre")
field_save_path = osp.join(out_root_path,"fai")
rgb_yuan_save_path = osp.join(out_root_path,"rgb_yuan")
rgb_save_path = osp.join(out_root_path,"rgb")
depth_save_path = osp.join(out_root_path,"depth")
rgb_origin_save_path = osp.join(out_root_path,"GT")
x_hat_save_path = osp.join(out_root_path,"x_hat")
y_hat_save_path = osp.join(out_root_path,"y_hat")
common_save_path = osp.join(out_root_path,"y_c")
common_warp_save_path = osp.join(out_root_path,"y_c_warp")
p_x_save_path = osp.join(out_root_path,"p_x")
p_y_save_path = osp.join(out_root_path,"p_y")
p_x_c_save_path = osp.join(out_root_path,"p_x_u")
p_y_c_save_path = osp.join(out_root_path,"p_y_v")
y_v_save_path = osp.join(out_root_path,"y_v")
x_u_save_path = osp.join(out_root_path,"x_u")
x_c_warp_save_path = osp.join(out_root_path,"x_c_warp")
x_c_single_save_path = osp.join(out_root_path,"x_c_single")
y_c_single_save_path = osp.join(out_root_path,"y_c_single")
patch_x_save_path = osp.join(out_root_path,"patch_x_save_path")
patch_y_save_path = osp.join(out_root_path,"patch_y_save_path")
use_in_gif_save_path = osp.join(out_root_path,"rgb_origin")
gif_out_save_path = osp.join(out_root_path,"output_gif")
gif_in_save_path = osp.join(out_root_path,"input_gif")
output_rgb_save_path = osp.join(out_root_path,"out_rgb")
GT_rgb_save_path = osp.join(out_root_path,"GT_rgb")
input_rgb_save_path = osp.join(out_root_path,"input_rgb")
gt_field_save_path = osp.join(out_root_path,"gt_field")
delta_field_save_path = osp.join(out_root_path,"delta_field")

out_root_path_file = open(osp.join(out_root_path,"details1.txt"),'w')

if not os.path.exists(delta_field_save_path):
    os.system("mkdir " + delta_field_save_path)
if not os.path.exists(plt_delta_path):
    os.system("mkdir " + plt_delta_path)
if not os.path.exists(field_save_path):
    os.system("mkdir " + field_save_path)
if not os.path.exists(plt_gt_path):
    os.system("mkdir " + plt_gt_path)
if not os.path.exists(plt_heat_path):
    os.system("mkdir " + plt_heat_path)
if not os.path.exists(plt_path):
    os.system("mkdir " + plt_path)
if not os.path.exists(gt_field_save_path):
    os.system("mkdir " + gt_field_save_path)
if not os.path.exists(rgb_save_path):
    os.system("mkdir " + rgb_save_path)
if not os.path.exists(rgb_origin_save_path):
    os.system("mkdir " + rgb_origin_save_path)
if not os.path.exists(depth_save_path):
    os.system("mkdir " + depth_save_path)
if not os.path.exists(common_save_path):
    os.system("mkdir " + common_save_path)
if not os.path.exists(common_warp_save_path):
    os.system("mkdir " + common_warp_save_path)   
if not os.path.exists(rgb_yuan_save_path):
    os.system("mkdir " + rgb_yuan_save_path)
if not os.path.exists(x_hat_save_path):
    os.system("mkdir " + x_hat_save_path)
if not os.path.exists(y_hat_save_path):
    os.system("mkdir " + y_hat_save_path)
if not os.path.exists(p_x_save_path):
    os.system("mkdir " + p_x_save_path)
if not os.path.exists(p_y_save_path):
    os.system("mkdir " + p_y_save_path)
if not os.path.exists(p_x_c_save_path):
    os.system("mkdir " + p_x_c_save_path)
if not os.path.exists(p_y_c_save_path):
    os.system("mkdir " + p_y_c_save_path)
if not os.path.exists(y_v_save_path):
    os.system("mkdir " + y_v_save_path)
if not os.path.exists(x_u_save_path):
    os.system("mkdir " + x_u_save_path)
if not os.path.exists(x_c_warp_save_path):
    os.system("mkdir " + x_c_warp_save_path)
if not os.path.exists(x_c_single_save_path):
    os.system("mkdir " + x_c_single_save_path)
if not os.path.exists(y_c_single_save_path):
    os.system("mkdir " + y_c_single_save_path)
if not os.path.exists(patch_x_save_path):
    os.system("mkdir " + patch_x_save_path)
if not os.path.exists(patch_y_save_path):
    os.system("mkdir " + patch_y_save_path)
if not os.path.exists(use_in_gif_save_path):
    os.system("mkdir " + use_in_gif_save_path)
if not os.path.exists(gif_out_save_path):
    os.system("mkdir " + gif_out_save_path)
if not os.path.exists(gif_in_save_path):
    os.system("mkdir " + gif_in_save_path)
if not os.path.exists(output_rgb_save_path):
    os.system("mkdir " + output_rgb_save_path)
if not os.path.exists(GT_rgb_save_path):
    os.system("mkdir " + GT_rgb_save_path)
if not os.path.exists(input_rgb_save_path):
    os.system("mkdir " + input_rgb_save_path)


def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=1)#duration间隔(改这里)
    return


def save_pic(data,path):
    if osp.exists(path):
        os.system("rm "+path)
        print("rm "+path)
    reimage = data.cpu().clone()
    # reimage[reimage > 1.0] = 1.0
    reimage[reimage > 1.0] = 1.0
    reimage[reimage < 0.0] = 0.0
    

    reimage = reimage.squeeze(0)
    # print(reimage.shape)
    reimage = transforms.ToPILImage()(reimage)  # PIL格式
    # print(reimage.size)
    # print(path)
    reimage.save(osp.join(path+'.png'))


def NCC(img1,img2):
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    r = np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2))
    return r

def Dice(img1, img2):
    img1 = img1.cpu()
    img2 = img2.cpu()
    dice = 2 *(img1 * img2).sum() / (img1.sum() + img2.sum())
    return dice

def Dice3(img1,img2):
    _,_,H,W= img1.shape
    fenmu1 = 0
    fenmu2 = 0
    fenzi = 0
    for i in range(H):
        for j in range(W):
            if (img1[0,0,i,j] != 0):
                fenmu1 += 1
            if (img2[0,0,i,j] != 0):
                fenmu2 += 1
            if (img1[0,0,i,j] != 0)&(img2[0,0,i,j] != 0):
                fenzi += 1

    dice = 2*fenzi/(fenmu1+fenmu2)
    return dice



nf_enc = [16, 32, 32, 32]
nf_dec = [32, 32, 32, 32, 32, 16, 16]

class Test_save:
    def __init__(self):
        self.epoch = 1000
        self.batch_size = 1
        self.patch_size = [256,256]
        self.criterion = MSELoss(reduction='mean')
        self.lr = 0.0001
        # self.path = '/temp_disk2/lep/dataset/brain_nobone/'
        self.path = '/temp_disk2/lep/dataset/RIRE/'

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.train_set =  ImageFolder(self.path,
                                split='train',
                                patch_size=self.patch_size,
                                transform=self.transform)
        # self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size,
                                    #    shuffle=True, num_workers=0)
        self.test_set =  ImageFolder(self.path,
                                split='test',
                                patch_size=self.patch_size,
                                transform=self.transform)
        self.test_loader = DataLoader(self.test_set, batch_size=1,
                                       shuffle=False, num_workers=0)
        self.model = InMIR(2,nf_enc,nf_dec).cuda()
        self.single_model = AGNet().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.9)
        self.train_loss = []
        self.test_loss = []
        self.val_psnr = []
        self.test_DICE_loss = []
        self.test_ssim_loss = []
        self.test_MSE_loss = []
        self.test_NCC_loss = []
        self.test_smooth_loss = []
        self.test_MIND_1_loss = []
        self.test_MIND_2_loss = []
        self.STN = SpatialTransformer([256,256]).cuda()
        self.grad_loss_fn = losses.gradient_loss
        self.STN .eval()
        self.best_loss = 10000000

    def test_save(self):
        global out_root_path_file

        state = torch.load('model_RIRE_small/best.pth')
        
        self.model.load_state_dict(state['model'])
        self.model.eval()
        with torch.no_grad():
            testepoch_loss = []
            testepoch_MSE_loss = []
            testepoch_dice_loss = []
            testepoch_ncc_loss = []
            testepoch_ssim_loss = []
            testepoch_MIND_1_loss = []
            testepoch_MIND_2_loss = []
            testepoch_smooth_loss = []
            i=1
            for batch, (rgb,depth,GT,grid,gt_grid,gt_fai) in enumerate(self.test_loader):
                rgb = rgb.cuda().squeeze(1)
                depth = depth.cuda()
                GT = GT.cuda()
                grid = grid.cuda()
                state_single = torch.load('RIRE_AGNet.pth')
                self.single_model.load_state_dict(state_single['model'])
                start = time.time()
                x_hat,y_hat,warp_y_c,x_c,x_warp,y_c,p_x,p_y,p_x_warp,fai,x_c_warp = self.model(rgb,depth)
                x_u_single,y_v_single,p_x_single,p_y_single,x_c_single,y_c_single,x_hat_single,y_hat_single = self.single_model(x_warp, depth)
                grid_warp = self.STN(grid,fai)
                grad_loss = self.grad_loss_fn(fai)
                testepoch_smooth_loss.append(grad_loss.item())

                end = time.time()
                
                ncc = NCC(x_warp,GT)

                testepoch_ncc_loss.append(ncc.item())
                loss11 = self.criterion(x_warp,GT)
                testepoch_MSE_loss.append(loss11.item())
                GT_ = GT.squeeze(0).squeeze(0)
                fig0 = plt.figure()
                plt.figure(dpi=120)

                offset2 = np.zeros((256,256))
                offset2[:,:] = ((fai[0,0,:,:].cpu().numpy()**2+fai[0,1,:,:].cpu().numpy()**2)**0.5)
                fig2 = plt.figure()

                cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
                sns.heatmap(offset2.reshape((256,256)),vmax=8,vmin=0,cmap="YlGnBu")
                plt.xticks([])
                plt.yticks([])
                plt.savefig(osp.join(plt_heat_path,str(i)+'.png'))
                

                offset1 = np.zeros((256,256))
                offset1[:,:] = ((gt_fai[0,0,0,:,:].cpu().numpy()**2+gt_fai[0,0,1,:,:].cpu().numpy()**2)**0.5)
                fig1 = plt.figure()

                cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
                sns.heatmap(offset1,vmax=8,vmin=0,cmap="YlGnBu")
                plt.xticks([])
                plt.yticks([])
                plt.savefig(osp.join(plt_gt_path,str(i)+'.png'))

                fig3 = plt.figure()
                sns.heatmap(offset1-offset2,vmax=5,vmin=-5,cmap="YlGnBu")
                plt.xticks([])
                plt.yticks([])
                plt.savefig(osp.join(plt_delta_path,str(i)+'.png'))

                plt.close('all')
                save_pic(gt_grid.squeeze(0),osp.join(gt_field_save_path,str(i)))
                save_pic(grid_warp, osp.join(field_save_path,str(i)))

                save_pic(x_warp, osp.join(rgb_save_path,str(i)))
                save_pic(GT, osp.join(rgb_origin_save_path,str(i)))
                save_pic(y_c, osp.join(common_save_path,str(i)))
                save_pic(warp_y_c, osp.join(common_warp_save_path,str(i)))
                save_pic(rgb, osp.join(rgb_yuan_save_path,str(i)))
                save_pic(depth, osp.join(depth_save_path,str(i)))
                save_pic(x_hat, osp.join(x_hat_save_path,str(i)))
                save_pic(y_hat, osp.join(y_hat_save_path,str(i)))
                save_pic(p_x, osp.join(p_x_save_path,str(i)))
                save_pic(p_y, osp.join(p_y_save_path,str(i)))
                save_pic(x_c, osp.join(p_x_c_save_path,str(i)))
                save_pic(x_c_warp,osp.join(x_c_warp_save_path,str(i)))

                save_pic(x_hat-x_c, osp.join(x_u_save_path,str(i)))
                save_pic(y_hat-y_c, osp.join(y_v_save_path,str(i)))
                save_pic(x_c_single,osp.join(x_c_single_save_path,str(i)))
                save_pic(y_c_single,osp.join(y_c_single_save_path,str(i)))
                
                i = i+1

            self.test_MSE_loss.append(np.mean(testepoch_MSE_loss))
            self.test_NCC_loss.append(np.mean(testepoch_ncc_loss))
            self.test_smooth_loss.append(np.mean(testepoch_smooth_loss))
            self.test_ssim_loss.append(np.mean(testepoch_ssim_loss))
            mse_std = np.var(testepoch_MSE_loss)
            ncc_std = np.var(testepoch_ncc_loss)
            smooth_std = np.var(testepoch_smooth_loss)
            ssim_std = np.var(testepoch_ssim_loss)
            print("MSE:",self.test_MSE_loss,mse_std)
            # print("dice:",self.test_DICE_loss,dice_std)
            print("ncc:",self.test_NCC_loss,ncc_std)
            print("smooth:",self.test_smooth_loss,smooth_std)
            print("dice:",self.test_ssim_loss,ssim_std)


