from train_deformation import Trainer
import os
# from test_deformation import Test_save

os.environ['CUDA_VISIBLE_DEVICES']='3'

if __name__ == '__main__':
    t = Trainer()
    t.train()
