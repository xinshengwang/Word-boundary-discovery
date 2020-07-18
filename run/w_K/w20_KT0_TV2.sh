data_path=/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/coco/audio/
save_root=outputs/w30_KT0_TV2
lr=0.0001
wd=1e-3
batch_size=64
n_epochs=100
start_epoch=100
lr_decay=50
bce_weight=20
BK_train=0
BK=1
val=True

python boundary_discovery.py --data_path $data_path \
--lr $lr \
--save_root $save_root \
--weight-decay $wd \
--batch_size $batch_size \
--start_epoch $start_epoch \
--epoch $n_epochs \
--lr_decay $lr_decay \
--bce-weight $bce_weight \
--BK_train $BK_train \
--BK $BK \
--only-val $val \

                        
                               
