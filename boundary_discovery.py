
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from utils.config import cfg

from models import WBDNet
from step import train_wbd
from dataloaders.datasets import WBD_Data, pad_collate


random.seed(cfg.manualSeed)
np.random.seed(cfg.manualSeed)
torch.manual_seed(cfg.manualSeed)
torch.cuda.manual_seed(cfg.manualSeed)
#torch.set_num_threads(4)

if cfg.CUDA:    
    torch.cuda.manual_seed(cfg.manualSeed)  
    torch.cuda.manual_seed_all(cfg.manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):   
    np.random.seed(cfg.manualSeed + worker_id)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training the model for word boundary discovery')

    parser.add_argument('--data_path', type=str, default= '/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/coco/audio/',
                        help='directory of database')   
    parser.add_argument('--save_root',type=str,default='outputs',
                        help='path for saving model and results')    
    # parameters
    parser.add_argument("--start_epoch",type=int,default=0,
                        help='resume the pertra')
    parser.add_argument("--epoch", type=int, default=100,
                        help="max epoch")
    parser.add_argument("--optim", type=str, default="adam",
                        help="training optimizer", choices=["sgd", "adam"])
    parser.add_argument('--batch_size', default=64, type=int, 
                        help='mini-batch size')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='initial learning rate')
    parser.add_argument('--lr_decay', default=50, type=int, metavar='LRDECAY',
                        help='Divide the learning rate by 10 every lr_decay epochs')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')  
    parser.add_argument('--bce-weight', '--bw', default=50, type=float,
                        help='weight to face the data unblance')  

    # evaluation
    parser.add_argument('--only-val',default=False,type=bool,
                        help='True for evaluation with pre-trained model')

    parser.add_argument('--BK_train',default=1,type=int,
                        help='neighbor number of the ground-truth boundary that considered as the boundary during training ')
    parser.add_argument('--BK',default=2,type=int,
                        help='predicted boundary is considered to be correct is it is whiin BK frames of the ground-truth boundary')

    args = parser.parse_args()
    
    dataset = WBD_Data(args.data_path,args,'train')
    dataset_val = WBD_Data(args.data_path,args,'val')
    
    
    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.WBDNet.batch_size,
            drop_last=True, shuffle=True,num_workers=cfg.workers,collate_fn=pad_collate,worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=cfg.WBDNet.batch_size,
        drop_last=False, shuffle=False,num_workers=cfg.workers,collate_fn=pad_collate,worker_init_fn=worker_init_fn) 
    

    model = WBDNet.RNN(cfg.WBDNet.input_size, cfg.WBDNet.batch_size, cfg.WBDNet.hidden_size)
    if not args.only_val:
        train_wbd.train(model,train_loader,val_loader,args)
    else:
        for i in range(4):
            args.BK=i
            print('bk=%d'%i)
            train_wbd.evaluation(model,val_loader,args)


    