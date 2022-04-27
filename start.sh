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
python main.py --train_num_points 100000 \
    --valid_num_points 2000 --test_iter 25000 \
    --rep_iter 2000 --batch_size 44 \
    --exp_name nocycleloss --D_pretrain_iter 5000 --lambda_once 1 --lambda_B 0 --lambda_A 0

