# Wrod boundary discovery

This is a supervised method with BLSTM to detect the speech word boundary.

## Database

* Databse: Speech COCO
* Speech: The speech is represented as filterbank.
* Target: The target is represented as a sequence of 0 (when the filterbank frame is not word boundary) or 1 (when the filterbanke frame is a word boundary). 

## Objective function

This word boundary detection task is treated as a binary classification problem for frame level of filterbank features.

## Train the model
```
python boundary_discovery.py    --data_path $data_path --lr 0.001 --save_root $save_root  --weight-decay 1e-4    --batch_size 64 
                                --epoch 100 --lr_decay 50  --bce-weight 10  --BK 2
           
```