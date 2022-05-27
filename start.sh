cd /tianyi-vol/cycleMT

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
python main.py --num_workers 4 --batch_size 32 --max_length 256 --train_num_points 800000 --valid_num_points 3500 \
                --poolsize 1 --D_lr 5e-6 --G_lr 5e-6 --D_gamma 0.95 --G_gamma 0.95 --rep_iter 1000 --D_pretrain_iter 5000 --test_iter 10000 \
                --lambda_once 1 --lambda_B 100 --lambda_A 100 --lambda_GP 50 \
                --load_D 1 --load_G 0 --valid_begin 1 --train_D 1 --DperG 15 \
                --exp_name 1once.100cycle,800kdata