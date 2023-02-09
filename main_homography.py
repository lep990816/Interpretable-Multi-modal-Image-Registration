from train_homography import Trainer
import os

os.environ['CUDA_VISIBLE_DEVICES']='3'

if __name__ == '__main__':
    t = Trainer()
    t.train()
