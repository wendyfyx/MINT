#!/bin/bash
#$ -S /bin/bash

export PYTHONBIN=python # change as needed


output_folder=../outputs
mkdir -p ${output_folder}/configs

script_path=../src/scripts/make_json_configs.py
${PYTHONBIN} ${script_path} -i ${output_folder}/data/data.h5 \
                            -metadata ../example_data/metadata.csv \
                            -test 0.2 \
                            -o ${output_folder}/configs/