
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
python pretrain.py