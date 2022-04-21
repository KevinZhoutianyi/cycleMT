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
python pretrain.py

