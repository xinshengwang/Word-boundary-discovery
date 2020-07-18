import torch
import torch.nn as nn
import os
import numpy as np
from utils.config import cfg
from utils.util import adjust_learning_rate, AverageMeter
import pdb

def train(model,train_loader,val_loader,args):
    if cfg.CUDA:
        model = model.cuda()

    exp_dir = args.save_root    
    save_model_path = os.path.join(exp_dir,'models')
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    epoch = args.start_epoch
    if epoch != 0:
        model.load_state_dict(torch.load("%s/models/WBDNet_%d.pth" % (exp_dir,epoch)))
        print('loaded parametres from epoch %d' % epoch)

    trainables = [p for p in model.parameters() if p.requires_grad]

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(trainables, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
                                weight_decay=args.weight_decay,
                                betas=(0.95, 0.999))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)

    loss_meter = AverageMeter()
    criterion_mse = nn.MSELoss() 
    criterion_bce = nn.BCELoss()
    
    save_file = os.path.join(exp_dir,'results.txt')
    while epoch <= args.epoch:
        epoch += 1
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)

        model.train()
        
        for i, (audio,target,mask,length,cap_ID) in enumerate(train_loader):
            loss = 0
            
            audio = audio.float().cuda()
            target = target.float().cuda()
            mask = mask.cuda()
            optimizer.zero_grad()

            predict = model(audio)
            criterion_bce_log = nn.BCEWithLogitsLoss(pos_weight=(args.bce_weight*target + 1.0)*mask)
            loss = criterion_bce_log(predict,target)

            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(),args.batch_size)

            if i%100 == 0:
                print('iteration = %d | loss = %f '%(i,loss))
        if epoch % 1 == 0:
            torch.save(model.state_dict(),
                "%s/models/WBDNet_%d.pth" % (exp_dir,epoch))
            metrics = evaluation(model,val_loader,args)
            Recall = metrics['recall']
            P = metrics['precision']
            F1 = metrics['F1']
            info = "epoch {} | loss {:.2f} | Recall {:.2%} | Precision {:.2%} | F1 {:.2%}\n".format(epoch,loss,Recall,P,F1)
            print(info)
            with open(save_file,'a') as f:
                f.write(info)

def evaluation(model,val_loader,args):
    model.eval()
    
    if args.only_val:
        exp_dir = args.save_root 
        model.cuda()
        model.load_state_dict(torch.load("%s/models/WBDNet_%d.pth" % (exp_dir,args.start_epoch)))
        print('loaded parametres from epoch %d' % args.start_epoch)
    
    
    total_retrieved = 0.0
    total_gt = 0.0
    total = 0.0
    TP = 0.0
    

    for i, (audio,target,mask,length,cap_ID) in enumerate(val_loader):
        audio = audio.float().cuda()
        target = target.int()
        mask = mask.int()
        output = model(audio)
        # if i==0 :
        #     pdb.set_trace()
        predict = get_predicted_boundary(output, args.BK)
        predict_label = (predict > 0.5) * mask
        # pdb.set_trace()
        total_retrieved += predict_label.sum()
        total_gt += ((target.sum()) + args.batch_size * 2 * args.BK) / (args.BK*2 + 1)
       
        total += mask.sum()
        # pdb.set_trace()
        TP += ((predict_label == target) * target).sum()
    Recall = TP / total_gt
    P = TP / total_retrieved
    F1 = 2*Recall*P/(Recall + P)

    metrics={}
    metrics['recall'] = Recall
    metrics['precision'] = P
    metrics['F1'] = F1
    if args.only_val:
        info = "Recall {:.2%} | Precision {:.2%} | F1 {:.2%}\n".format(Recall,P,F1)
        print(info)
    else:
        return metrics

def get_predicted_boundary(predict,k):
    stride = 2*k+1
    pading = int((stride-1)/2)
    # pdb.set_trace()
    max_pool = nn.MaxPool1d(stride,1,padding=pading).cuda()
    ext_predict = predict.unsqueeze(1)
    max_predict = max_pool(ext_predict).squeeze()
    if max_predict.shape[-1]<ext_predict.shape[-1]:
        mid = torch.zeros(ext_predict.shape).squeeze()
        mid[:,:max_predict.shape[-1]] = max_predict
        max_predict = mid
    max_mask = ((predict.to('cpu') - max_predict.to('cpu')) >= 0).int()
    sub_mask = max_mask -1
    predict = predict.to('cpu').detach() 
    predict = predict * max_mask + sub_mask
    return predict

