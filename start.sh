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
python main.py --train_num_points 10000 \
    --valid_num_points 1000 --test_iter 2500 \
    --rep_iter 500 --batch_size 32 \
    --exp_name nocyclelosstrainDTrainG --D_pretrain_iter 5000 --lambda_once 1 --lambda_B 1 --lambda_A 1\
    --load_D 0 --valid_begin 1

