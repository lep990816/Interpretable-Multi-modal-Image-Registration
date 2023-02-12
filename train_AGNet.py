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
from network_AGnet import AGNet
from torchvision import transforms
from torch.utils.data import DataLoader
from data_affine import ImageFolder
import os.path as osp

model_path = 'AGNet_RIRE'
if not os.path.exists(model_path):
    os.system("mkdir "+model_path)
model_latest = os.path.join(model_path,'last.pth')
model_best = os.path.join(model_path,'best.pth')

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

class Trainer:
    def __init__(self):
        self.epoch = 100
        self.batch_size = 2
        self.patch_size = [480,480]
        self.criterion = MSELoss(reduction='mean')
        self.lr = 0.0001
        self.path = '/temp_disk2/lep/dataset/RIRE/'
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.train_set =  ImageFolder(self.path,
                                split='train',
                                patch_size=self.patch_size,
                                transform=self.transform)
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size,
                                       shuffle=True, num_workers=0)
        self.test_set =  ImageFolder(self.path,
                                split='train',
                                patch_size=self.patch_size,
                                transform=self.transform)
        self.test_loader = DataLoader(self.test_set, batch_size=1,
                                       shuffle=False, num_workers=0)
        self.model = AGNet().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.9)
        self.train_loss = []
        self.test_loss = []
        self.val_psnr = []
        self.best_loss = 10000000

    def train(self):
        seed = random.randint(1, 1000)
        print("===> Random Seed: [%d]" % seed)
        random.seed(seed)
        torch.manual_seed(seed)
        best = 1000000
        for ep in range(1, self.epoch+1):
            self.model.train()
            epoch_loss = []
            for batch, (input1,input2) in enumerate(self.train_loader):
                input1 = input1.cuda()
                input2 = input2.cuda()
                self.optimizer.zero_grad()
                torch.cuda.synchronize()
                start_time = time.time()
                p_x,p_y,x_hat,y_hat = self.model(input1, input2) 
                loss1 = 1000 * self.criterion(input1,x_hat)  + 1000 * self.criterion(input2,y_hat)
                epoch_loss.append(loss1.item())
                loss1.backward()
                self.optimizer.step()
                torch.cuda.synchronize()
                end_time = time.time()
                if batch % 10 == 0 and batch != 0:
                    print('Epoch:{}\tcur/all:{}/{}\tAvg Loss:{:.4f}\tTime:{:.2f}'.format(ep, batch, len(self.train_loader), loss1.item(), end_time-start_time))
            self.scheduler.step()
            self.train_loss.append(np.mean(epoch_loss))
            state = {
                'model': self.model.state_dict(),
                'train_loss': self.train_loss
            }
            torch.save(state, os.path.join(model_path, 'last.pth'))
            if ep % 1 == 0:
                torch.save(state, os.path.join(model_path, str(ep)+'.pth'))
        print('===> Finished Training!')

    def test(self,ep):
        state = torch.load(os.path.join(model_path,str(ep)+'.pth'))
        self.model.load_state_dict(state['model'])
        self.model.eval()
        with torch.no_grad():
            
            testepoch_loss = []
            for batch, (input1,input2) in enumerate(self.test_loader):
                input1 = input1.cuda()
                input2 = input2.cuda()
                p_x,p_y,x_hat,y_hat = self.model(input1, input2)
                loss1 = 1000 * self.criterion(input1,x_hat)  + 1000 * self.criterion(input2,y_hat)
                testepoch_loss.append(loss1.item())

            if np.mean(testepoch_loss) < self.best_loss:
                torch.save(state, os.path.join(model_path, 'best.pth'))
                print('Covered!')
                self.best_loss = np.mean(testepoch_loss)
            self.test_loss.append(np.mean(testepoch_loss))
        return self.test_loss
