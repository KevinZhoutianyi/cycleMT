cd /base_vol/TIANYIZHOU/cycleMT

pip install torch_optimizer
pip install transformers==4.18.0
pip install datasets
pip install sacrebleu==1.5.1
pip install wandb
mkdir log
mkdir model
mkdir checkpoint
rm -f ./checkpoint/*
rm -f ./log/*.txt
rm -f ./tensorboard/* 
python main.py --num_workers 2 --batch_size 2 --max_length 128 --train_num_points 400000 --valid_num_points 3500 \
        --poolsize 1 --D_lr 5e-6 --G_lr 5e-6 --D_gamma 1 --G_gamma 1  --num_beam 4 --rep_iter 1000 --D_pretrain_iter 5000 --test_iter 10000\
        --lambda_once 1 --lambda_B 50 --lambda_A 50 --lambda_GP 50 --load_D 0 --load_G 0 --valid_begin 1 --train_D 1 --DperG 15 \
        --exp_name 1once.50cycle,800kdata.4beam.2bs

