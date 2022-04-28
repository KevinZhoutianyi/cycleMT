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
python main.py --train_num_points 50000 \
    --valid_num_points 1000 --test_iter 10000 \
    --rep_iter 1000 --batch_size 40 \
    --exp_name 100cycle2beam --D_pretrain_iter 5000 --lambda_once 1 --lambda_B 100 --lambda_A 100\
    --load_D 0 --valid_begin 1

