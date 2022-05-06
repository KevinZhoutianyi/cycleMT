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
python main.py  --num_workers 4             --batch_size 32             --max_length 256    \
                --train_num_points 300000   --valid_num_points 3000     --poolsize 1                    \
                --D_lr 5e-5                 --G_lr 5e-6                 --D_gamma 1       --G_gamma 1      \
                --rep_iter 1000             --D_pretrain_iter 0      --test_iter 10000    \
                --lambda_once 1             --lambda_B 20                --lambda_A 20         --lambda_GP 0.001    \
                --load_D 0                  --load_G 0                  --valid_begin 1             --train_D 1         \
                --exp_name 0.001GP20GP
