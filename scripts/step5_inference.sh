#!/bin/bash
#$ -S /bin/bash

export PYTHONBIN=/ifs/loni/faculty/thompson/four_d/wfeng/miniconda3/envs/dl/bin/python

output_folder=../outputs

model_ckpt=${output_folder}/model_logs/vae_finetune/tractoinferno/checkpoints/last.ckpt
config_path=${output_folder}/configs/config_test.json

## Run inference and save z, reconstruction and evaluation metrics to .h5 files
mkdir -p ${output_folder}/inference

cmd="${PYTHONBIN} ../src/scripts/inference_and_eval_vae.py "
cmd+="--seed 2023 "
cmd+="--device cpu "
cmd+="--config_path ${config_path} "
cmd+="--model_ckpt ${model_ckpt} "
cmd+="--n_segments 100 "
# cmd+="--output_z_path ${output_folder}/inference/z_test.h5 "
# cmd+="--output_recon_path ${output_folder}/inference/recon_test.h5 "
cmd+="--output_eval_path ${output_folder}/inference/eval_test.h5 "
cmd+="--no-overwrite"

echo "RUN INFERECE: "
echo "${cmd}"
eval ${cmd}

## Aggregate evaluation metrics into .csv files
mkdir -p ${output_folder}/evaluation

cmd="${PYTHONBIN} ../src/scripts/save_eval_metrics_to_csv.py "
cmd+="--i ${output_folder}/inference/eval_test.h5 "
cmd+="--o ${output_folder}/evaluation/eval_test "

echo "AGGREGATE METRICS: "
echo "${cmd}"
eval ${cmd}