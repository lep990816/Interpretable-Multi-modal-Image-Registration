import os,glob
import cv2
import time
import torch
import random
import matplotlib
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import MSELoss
from network_homography import InMIRNet
from network_AGnet import AGNet
from torchvision import transforms
from torch.utils.data import DataLoader
from data_homography import ImageFolder
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import os.path as osp

def save_pic(data,path):
    if osp.exists(path):
        os.system("rm "+path)
        print("rm "+path)
    reimage = data.cpu().clone()
    reimage = reimage.squeeze(0)
    reimage = transforms.ToPILImage()(reimage)  # PIL格式
    reimage.save(osp.join(path+'.png'))


def NCC(img1,img2):
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    r = np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2))
    return r

def h_adjust(orishapea,orishapeb,resizeshapea,resizeshapeb, h_1): #->h_ori
    a =  resizeshapea / orishapea 
    b = resizeshapeb / orishapeb 
    # the shape of H matrix should be (1, 3, 3)
    h_1[:, 0, :] = a*h_1[:, 0, :]
    h_1[:, :, 0] = (1./a)*h_1[:, :, 0]
    h_1[:, 1, :] = b * h_1[:, 1, :]
    h_1[:, :, 1] = (1. / b) * h_1[:, :, 1]
    return h_1

class Trainer:
    def __init__(self):
        self.epoch = 1000
        self.batch_size = 1
        self.patch_size = [256,256]
        self.criterion = MSELoss(reduction='mean')
        self.lr = 0.00001
        self.path = '/temp_disk2/lep/dataset/DPDN/'
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.train_set =  ImageFolder(self.path,
                                split='train',
                                patch_size=self.patch_size,
                                transform=self.transform)
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size,
                                       shuffle=True, num_workers=0)
        self.test_set =  ImageFolder(self.path,
                                split='test',
                                patch_size=self.patch_size,
                                transform=self.transform)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size,
                                       shuffle=False, num_workers=0)
        self.model = InMIRNet().cuda()
        self.single_model = AGNet().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.9)
        self.train_loss = []
        self.test_loss = []
        self.val_psnr = []
        self.best_loss = 10000000        
        self.lambda1 = 10
        self.lambda2 = 2.5
        self.lambda3 = 50
        self.model_path = os.path.join('model_'+str(self.lambda1)+'_'+str(self.lambda2)+'_'+str(self.lambda3))

    def train(self):
        seed = random.randint(1, 1000)
        print("===> Random Seed: [%d]" % seed)
        random.seed(seed)
        torch.manual_seed(seed)
        for ep in range(1, self.epoch+1):
            self.model.train()
            epoch_loss = []
            for batch, (depth,rgb,GT,input_rgb,GT_rgb,filename,delta) in enumerate(self.train_loader):
                rgb = rgb.cuda()
                depth = depth.cuda()
                delta = delta.cuda()
                GT = GT.cuda()

                state_single = torch.load('AGNet_DPDN.pth')
                self.single_model.load_state_dict(state_single['model'])
                self.single_model.eval()

                self.optimizer.zero_grad()
                torch.cuda.synchronize()
                start_time = time.time()
                
                x_hat,y_hat,warp_y_c,x_c,x_warp,y_c,p_x,p_y,patch_b_hat,p_x_warp,delta_pre,h_inv = self.model(rgb, depth)
                x_u_single,y_v_single,p_x_single,p_y_single,x_c_single,y_c_single,x_hat_single,y_hat_single = self.single_model(x_warp, depth)

                loss = (
                        200 * self.lambda1 * self.criterion(x_warp,GT) 
                        # +200 * self.lambda1 * self.criterion(delta,delta_pre) #使用坐标损失效果更好
                        + 200 * self.lambda2 * self.criterion(p_y,p_y_single) 
                        + 200 * self.lambda2 * self.criterion(p_x_warp,p_x_single) 
                        + 200 * self.lambda3 * self.criterion(x_hat,rgb) + 200 * self.lambda3 * self.criterion(y_hat,depth) 
                        )
                loss1 = self.criterion(x_warp,GT)
                loss2 = self.criterion(x_hat,rgb) + self.criterion(y_hat,depth)
                delta = delta.squeeze(0)
                delta_loss = self.criterion(delta,delta_pre.squeeze(0))
                loss.backward()
                chonggou2 = self.criterion(x_hat_single,GT) + self.criterion(y_hat_single,depth)
                ssim1 = ms_ssim(x_warp,GT, data_range=1, size_average=False)[0]
                ssim2 = ms_ssim(p_x_warp,p_x_single, data_range=1, size_average=False)[0]
                epoch_loss.append(loss.item())

                self.optimizer.step()
                torch.cuda.synchronize()
                end_time = time.time()
                if batch % 100 == 0 and batch != 0:
                    print('Epoch:{}\tcur/all:{}/{}\tAvg Loss:{:.4f}\tTime:{:.2f}\tMSE:{:.4f}\tdelta_loss:{:.4f}\tchonggou1:{:.4f}\tchonggou2:{:.4f}\tssim1:{:.4f}\tssim_common:{:.4f}\tfilename:{}'\
                    .format(ep, batch, len(self.train_loader), loss.item(), end_time-start_time,loss1.item(),delta_loss.item(),loss2.item(),chonggou2.item(),ssim1.item(),ssim2.item(),filename))

            self.scheduler.step()
            self.train_loss.append(np.mean(epoch_loss))
            print('Epoch:{}\tavg loss{:.4f}'.format(ep,np.mean(epoch_loss)))
            state = {
                'model': self.model.state_dict(),
                'train_loss': self.train_loss
            }
            if not os.path.exists(self.model_path):
                os.system("mkdir "+self.model_path)
            torch.save(state, os.path.join(self.model_path, 'last.pth'))
            if ep % 100 == 0:
                torch.save(state, os.path.join(self.model_path, str(ep)+'.pth'))
        print('===> Finished Training!')

    def test(self):
        
        save_path = 'test_result/'
        state = torch.load(os.path.join(self.model_path, 'last.pth'))
        self.model.load_state_dict(state['model'])
        self.model.eval()
        state_single = torch.load('AGNet_DPDN.pth')
        self.single_model.load_state_dict(state_single['model'])
        self.single_model.eval()
        with torch.no_grad():
            
            testepoch_loss = []
            
            for batch, (depth,rgb,GT,input_rgb,GT_rgb,filename,delta) in enumerate(self.test_loader):
                rgb = rgb.cuda()
                depth = depth.cuda()
                delta = delta.cuda()
                GT = GT.cuda()
                x_hat,y_hat,warp_y_c,x_c,x_warp,y_c,p_x,p_y,patch_b_hat,p_x_warp,delta_pre,h_inv = self.model(rgb, depth)
                x_u_single,y_v_single,p_x_single,p_y_single,x_c_single,y_c_single,x_hat_single,y_hat_single = self.single_model(x_warp, depth)
                loss = (
                        200 * self.lambda1 * self.criterion(x_warp,GT) 
                        # +200 * self.lambda1 * self.criterion(delta,delta_pre) #使用坐标损失效果更好
                        + 200 * self.lambda2 * self.criterion(p_y,p_y_single) 
                        + 200 * self.lambda2 * self.criterion(p_x_warp,p_x_single) 
                        + 200 * self.lambda3 * self.criterion(x_hat,rgb) + 200 * self.lambda3 * self.criterion(y_hat,depth) 
                        )
                delta_loss = self.criterion(delta,delta_pre)
                testepoch_loss.append(delta_loss.item())

            if np.mean(testepoch_loss) < self.best_loss:
                torch.save(state, os.path.join(self.model_path, 'best.pth'))
                print(np.mean(testepoch_loss))
                print('Covered!')
                self.best_loss = np.mean(testepoch_loss)

            self.test_loss.append(np.mean(testepoch_loss))

        return self.test_loss
