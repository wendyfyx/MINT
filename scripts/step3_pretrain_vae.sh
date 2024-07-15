#!/bin/bash
#$ -S /bin/bash

export OMP_NUM_THREADS=1
export PYTHONBIN=/ifs/loni/faculty/thompson/four_d/wfeng/miniconda3/envs/dl/bin/python

script_path=../src/scripts/pretrain_vae.py
log_folder=../outputs/model_logs

# Settings
config_path=../outputs/configs/config_train.json
model_name=vae_pretrain
model_version=ppmi_test

# Pretrain VAE model
cmd="${PYTHONBIN} ${script_path} "
cmd+="--seed 2024 "
cmd+="--device cpu "
cmd+="--config_path ${config_path} "
cmd+="--batch_size 128 "
cmd+="--n_sample 4 "
cmd+="--n_loader 100 " 
cmd+="--n_worker 4 "
cmd+="--feature_idx 0 1 2 3 4 5 6 "
cmd+="--centering "
cmd+="--scaling "

cmd+="--dim_latent 64 "
cmd+="--encoder_type Encoder "
cmd+="--encoder_kernels 15 15 15 15 "
cmd+="--encoder_dims 32 64 128 256 "
cmd+="--encoder_block SepConvBlock "
cmd+="--decoder_type Decoder "
cmd+="--decoder_kernels 15 15 15 "
cmd+="--decoder_dims 128 96 64 "
cmd+="--decoder_block SepConvBlock "
cmd+="--norm_layer_type bn "
cmd+="--no-norm_first "
cmd+="--norm_eps 1e-5 "

cmd+="--loss_wt 10 1 "
cmd+="--recon_wt 1.5 1 "
cmd+="--recon_mode likelihood likelihood "
cmd+="--anneal_mode sigmoid "
cmd+="--anneal_cycle 8 "
cmd+="--anneal_ratio 0.5 "

cmd+="--learning_rate 0.0005 "
cmd+="--weight_decay 0.001 "
cmd+="--gradient_clip_val 2 "
cmd+="--gradient_clip_algorithm norm "
cmd+="--n_epoch 30 "
cmd+="--log_folder ${log_folder} "
cmd+="--model_name ${model_name} "
cmd+="--model_version ${model_version} "
cmd+="--log_every_n_steps 10 "

echo "TRAINING COMMAND: "
echo "${cmd}"
eval ${cmd}