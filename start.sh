cd /tianyi-vol/cycleMT

pip install torch_optimizer
pip install transformers==4.18.0
pip install datasets
pip install sacrebleu==1.5.1
pip install wandb
mkdir log
mkdir model
mkdir checkpoint
rm -f ./log/*.txt
rm -f ./tensorboard/*
python main.py  --num_workers 2             --batch_size 32             --max_length 256    \
                --train_num_points 50000    --valid_num_points 1000                         \
                --D_lr 1e-5                 --D_gamma 0.5                                   \
                --rep_iter 1000             --D_pretrain_iter 5000      --test_iter 10000   \
                --lambda_once 1             --lambda_B 0                --lambda_A 0        \
                --load_D 0                  --valid_begin 1             --train_D 1         \
                --exp_name  fixgenerate256
