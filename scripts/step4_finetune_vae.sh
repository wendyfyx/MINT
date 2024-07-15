#!/bin/bash
#$ -S /bin/bash

export OMP_NUM_THREADS=1
export PYTHONBIN=/ifs/loni/faculty/thompson/four_d/wfeng/miniconda3/envs/dl/bin/python

script_path=../src/scripts/finetune_vae.py
log_folder=../outputs/model_logs

# Data used for finetuning
config_path=../outputs/configs/config_train.json

# Pretrained model file
pretrain_ckpt_path=../assets/tractoinferno_pretrain_24000steps.ckpt

# Finetune settings
finetune_model_name=vae_finetune
finetune_model_version=tractoinferno
n_step=2000 # number of stpes to fine tune

# Run finetune script
cmd="${PYTHONBIN} ${script_path} "
cmd+="--seed 2023 "
cmd+="--device cpu "
cmd+="--config_path ${config_path} "

cmd+="--pretrain_ckpt_path ${pretrain_ckpt_path} "
cmd+="--output_log_folder ${log_folder} "
cmd+="--finetune_model_name ${finetune_model_name} "
cmd+="--finetune_model_version ${finetune_model_version} "

cmd+="--log_every_n_steps 10 "
cmd+="--checkpoint_every_n_steps 1000 "
cmd+="--n_step ${n_step} "

echo "FINE-TUNING COMMAND: "
echo "${cmd}"
eval ${cmd}