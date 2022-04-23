cd /tianyi-vol/cycleMT

pip install transformers
pip install datasets
pip install sacrebleu==1.5.1
pip install torch_optimizer
pip install wandb
mkdir log
mkdir model
rm -f ./log/*.txt
rm -f ./tensorboard/*
python main.py --train_num_points 100000 --valid_num_points 2000 --test_iter 25000 --rep_iter 2000 --batch_size 48 --exp_name servertest

