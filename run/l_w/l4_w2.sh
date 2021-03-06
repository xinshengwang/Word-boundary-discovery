data_path=/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/coco/audio/
save_root=outputs/l4_w2
lr=0.0001
wd=1e-2
batch_size=64
n_epochs=100
start_epoch=0
lr_decay=50
bce_weight=10
BK_train=0
BK=2

python boundary_discovery.py --data_path $data_path \
--lr $lr \
--save_root $save_root \
--weight-decay $wd \
--batch_size $batch_size \
--epoch $n_epochs \
--lr_decay $lr_decay \
--bce-weight $bce_weight \
--BK_train $BK_train \
--BK $BK
                        
                               
