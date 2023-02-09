import os,glob
import cv2
import time
import torch
import random
import matplotlib
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import MSELoss
from Model import losses
from network_deformation import InMIR
from network_AGnet import AGNet
from torchvision import transforms
from torch.utils.data import DataLoader
from data_deformation import ImageFolder
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import os.path as osp

model_path = 'model_RIRE_small'
if not os.path.exists(model_path):
    os.system("mkdir "+model_path)

def save_pic(data,path):
    if osp.exists(path):
        os.system("rm "+path)
        print("rm "+path)
    reimage = data.cpu().clone()
    reimage[reimage > 1.0] = 1.0
    reimage[reimage < 0.0] = 0.0
    reimage = reimage.squeeze(0)
    reimage = transforms.ToPILImage()(reimage) 
    reimage.save(osp.join(path+'.png'))

def Dice(img1, img2):
    
    dice = 2 *(img1 * img2).sum() / (img1.sum() + img2.sum())
    return dice

def NCC(img1,img2):
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    r = np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2))
    return r

class Gradient_loss(nn.Module):
    def __init__(self):
        super(Gradient_loss, self).__init__()

    def forward(s, penalty='l2'):
        dx = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dz = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
        if (penalty == 'l2'):
            dx = dx * dx
            dz = dz * dz
        d = torch.mean(dx)  + torch.mean(dz)       
        return d / 2.0

class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
		intersection = input_flat * target_flat
		loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N
		return loss

class smooth_Loss(nn.Module):
    def __init__(self):
        super(smooth_Loss, self).__init__()
    def	forward(self, fai,batch):
        tidu = torch.tensor([0]).float().cuda()
        for k in range(fai.size(0)):
            for i in range(62):
                for j in range(62):                  
                    tidu += torch.abs(fai[k][0][i+1][j+1] - fai[k][0][i][j+1]) + torch.abs(fai[k][0][i+1][j+1] - fai[k][0][i+2][j+1] ) + torch.abs(fai[k][0][i+1][j+1]  - fai[k][0][i+1][j] ) + torch.abs(fai[k][0][i+1][j+1] - fai[k][0][i+1][j+2])
                    tidu += torch.abs(fai[k][1][i+1][j+1] - fai[k][1][i][j+1]) + torch.abs(fai[k][1][i+1][j+1] - fai[k][1][i+2][j+1]) + torch.abs(fai[k][1][i+1][j+1] - fai[k][1][i+1][j]) + torch.abs(fai[k][1][i+1][j+1] - fai[k][1][i+1][j+2])
        return tidu

nf_enc = [16, 32, 32, 32]
nf_dec = [32, 32, 32, 32, 32, 16, 16]

class Trainer:
    def __init__(self):
        self.epoch = 1000
        self.batch_size = 4
        self.patch_size = [256,256]
        self.criterion = MSELoss(reduction='mean')
        self.lr = 0.001
        self.path = '/temp_disk2/lep/dataset/RIRE/'
        # self.path = '/home/yangwenzhe/lep_code/dataset/RGB_Depth_training_dataset/'
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
        self.test_loader = DataLoader(self.test_set, batch_size=1,
                                       shuffle=False, num_workers=0)
        self.model = InMIR(2,nf_enc,nf_dec).cuda()
        self.diceloss = DiceLoss()
        self.smooth_Loss = smooth_Loss()
        self.single_model = AGNet().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.75)
        self.train_loss = []
        self.test_loss = []
        self.test_loss2 = []
        self.test_grad_loss = []
        self.val_psnr = []
        self.sim_loss_fn = losses.mse_loss
        self.grad_loss_fn = losses.gradient_loss
        self.best_loss = 10000

    def train(self):
        seed = random.randint(1, 1000)
        print("===> Random Seed: [%d]" % seed)
        random.seed(seed)
        torch.manual_seed(seed)

        for ep in range(1, self.epoch+1):
            self.model.train()
            epoch_loss = []
            for batch, (modal1,modal2,GT) in enumerate(self.train_loader):
                modal1 = modal1.cuda().squeeze(1)
                modal2 = modal2.cuda()
                GT = GT.cuda()

                state_single = torch.load('RIRE_AGNet.pth')

                self.single_model.load_state_dict(state_single['model'])
                self.single_model.eval()

                self.optimizer.zero_grad()
                torch.cuda.synchronize()
                start_time = time.time()
                
                x_hat,y_hat,warp_y_c,x_c,x_warp,y_c,p_x,p_y,p_x_warp,fai,x_c_warp = self.model(modal1, modal2)
                x_u_single,y_v_single,p_x_single,p_y_single,x_c_single,y_c_single,x_hat_single,y_hat_single = self.single_model(x_warp, modal2)

                
                dice_loss = self.diceloss(x_warp,GT)
                grad_loss = self.grad_loss_fn(fai)
                loss = (grad_loss * 100 + 1000 * self.criterion(x_warp,GT)
                        + 250 * self.criterion(p_y,y_c_single)
                        + 250 * self.criterion(x_c_warp,x_c_single) 
                        + 100 * self.criterion(x_hat,modal1) + 100 * self.criterion(y_hat,modal2))

                loss1 = self.criterion(x_warp,GT)
                loss2 = self.criterion(x_hat,modal1) + self.criterion(y_hat,modal2)
                chonggou2 = self.criterion(x_hat_single,GT) + self.criterion(y_hat_single,modal2)
                ssim1 = ms_ssim(x_warp,GT, data_range=1, size_average=False)[0]
                ssim2 = ms_ssim(x_c_warp,x_c_single, data_range=1, size_average=False)[0]

                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
                torch.cuda.synchronize()
                end_time = time.time()
                if batch % 5 == 0 and batch != 0:
                    print('Epoch:{}\tcur/all:{}/{}\tAvg Loss:{:.4f}\tTime:{:.2f}\tDICE:{:.4f}\tMSE:{:.4f}\tchonggou1:{:.4f}\tchonggou2:{:.4f}\tssim1:{:.4f}\tssim_common:{:.4f}\loss_smooth:{:.9f}'\
                    .format(ep, batch, len(self.train_loader), loss.item(), end_time-start_time,(1-dice_loss).item(),loss1.item(),loss2.item(),chonggou2.item(),ssim1.item(),ssim2.item(),grad_loss.item()))

            self.scheduler.step()
            self.train_loss.append(np.mean(epoch_loss))
            print('Epoch:{}\tavg loss{:.4f}'.format(ep,np.mean(epoch_loss)))
            state = {
                'model': self.model.state_dict(),
                'train_loss': self.train_loss
            }
            torch.save(state, os.path.join('model_RIRE_small', 'last.pth'))
            if ep % 100 == 0:
                torch.save(state, os.path.join('model_RIRE_small', str(ep)+'.pth'))

        print('===> Finished Training!')

    def test(self):
        
        save_path = 'test_result/'
        state = torch.load('model_RIRE_small/last.pth')
        self.model.load_state_dict(state['model'])
        self.model.eval()
        state_single = torch.load('RIRE_AGNet.pth')
        self.single_model.load_state_dict(state_single['model'])
        self.single_model.eval()

        with torch.no_grad():
            testepoch_MSE_loss = []
            testepoch_loss = []
            testepoch_grad_loss = []

            for batch, (modal1,modal2,GT) in enumerate(self.test_loader):
                modal1 = modal1.cuda().squeeze(1)
                modal2 = modal2.cuda()
                GT = GT.cuda()
                x_hat,y_hat,warp_y_c,x_c,x_warp,y_c,p_x,p_y,p_x_warp,fai,x_c_warp = self.model(modal1, modal2)
                x_u_single,y_v_single,p_x_single,p_y_single,x_c_single,y_c_single,x_hat_single,y_hat_single = self.single_model(x_warp, modal2)
                dice_loss = self.diceloss(x_warp,GT)
                loss = self.criterion(x_warp,GT)
                grad_loss = self.grad_loss_fn(fai)
                loss = (grad_loss * 100 + 1000 * self.criterion(x_warp,GT)
                        + 250 * self.criterion(p_y,y_c_single)
                        + 250 * self.criterion(x_c_warp,x_c_single) 
                        + 100 * self.criterion(x_hat,modal1) + 100 * self.criterion(y_hat,modal2))
                testepoch_MSE_loss.append(loss.item())
                testepoch_loss.append(loss2.item())
                testepoch_grad_loss.append(grad_loss.item())

            if np.mean(testepoch_MSE_loss) < self.best_loss:
                torch.save(state, os.path.join('model_RIRE_small', 'best.pth'))
                print(np.mean(testepoch_MSE_loss))
                print('Covered!')
                self.best_loss = np.mean(testepoch_MSE_loss)
            self.test_loss.append(np.mean(testepoch_MSE_loss))
            self.test_loss2.append(np.mean(testepoch_loss))
            self.test_grad_loss.append(np.mean(testepoch_grad_loss))

        return self.test_loss,self.test_loss2,self.test_grad_loss
