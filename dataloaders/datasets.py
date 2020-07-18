from utils.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate

import os
import sys
import json
import numpy as np
import numpy.random as random


def pad_collate(batch):
    max_input_len = float('-inf')  
    
    for elem in batch:
        mel, target, length,cap_ID = elem
        max_input_len = max_input_len if max_input_len > length else length       

    for i, elem in enumerate(batch):
        mel, target, length,cap_ID = elem
        input_length = mel.shape[1]
        input_dim = mel.shape[0]

        pad_mel = np.zeros((input_dim,max_input_len), dtype=np.float)
        pad_mel[:input_dim, :input_length] = mel       

        pad_target = np.zeros(max_input_len,dtype=np.float)
        mask = np.zeros(max_input_len,dtype=np.float)
        pad_target[:input_length] = target
        mask[:input_length] = 1.0
        pad_mel = pad_mel.transpose(1,0)
        batch[i] = (pad_mel, pad_target, mask, length,cap_ID)
        # print('feature.shape: ' + str(feature.shape))
        # print('trn.shape: ' + str(trn.shape))

    batch.sort(key=lambda x: x[-2], reverse=True)

    return default_collate(batch)


class WBD_Data(data.DataLoader):
    def __init__(self, data_path,args,split='train'):
        self.args = args
        self.split = split
        if split=='train':
            self.data_dir = os.path.join(data_path,'train2014')
        else:
            self.data_dir = os.path.join(data_path,'val2014')
        self.filenames = self.load_filelnames(self.data_dir)

    def load_filelnames(self,data_dir):
        if self.split == 'train':
            path = data_dir + '/filenames/' + 'All_1.0_None.json'
        else:
            path = data_dir + '/filenames/' + 'All_1.0_None_uniqueImgID_2000.json'
        with open(path,'rb') as f:
            data = json.load(f)
        return data

    def load_json(self,path):
        with open(path,'rb') as f:
            data = json.load(f)
        return data

    def load_target(self,wav_duration,mel_len,timecode):
        target = np.zeros(mel_len)
        times = []
        for item in timecode:
            if 'WORD' in item:
                time = item[0]
                times.append(time)
        times = np.array(times)

        positions = times * mel_len / (wav_duration*1000)
        positions = np.around(positions).astype(np.int32)
        if self.split == 'train':
            if self.args.BK_train == 0:
                target[positions] = 1
            else:
                pad_positions = list(positions)
                
                for i in range(self.args.BK_train):
                    k = i+1
                    positions_right = positions[:-1] + k
                    positions_left = positions[1:] - k
                    pad_positions = pad_positions + list(positions_left) + list(positions_right)
                target[pad_positions] = 1
        else:
            if self.args.BK == 0:
                target[positions] = 1
            else:
                pad_positions = list(positions)
                for i in range(self.args.BK):
                    k = i+1
                    positions_right = positions[:-1] + k
                    positions_left = positions[1:] - k
                    pad_positions = pad_positions + list(positions_left) + list(positions_right)
                target[pad_positions] = 1
        return target

    def __getitem__(self,index):
        data_dict = self.filenames[index]
        wav_name = data_dict['wavFilename']
        mel_name = wav_name.replace('.wav','.npy')
        mel_path = self.data_dir + '/mel/' + mel_name 
        mel =  np.load(mel_path,allow_pickle=True)
        json_path = self.data_dir + '/json/' + wav_name.replace('.wav','.json')
        json_dict = self.load_json(json_path)

        wav_duration = json_dict['duration']
        mel_len = mel.shape[1]
        timecode = json_dict['timecode']
        cap_ID = json_dict['captionID']

        target = self.load_target(wav_duration,mel_len,timecode)
        
        return mel, target, mel_len, cap_ID


    def __len__(self):
        return len(self.filenames)