#!/bin/bash
#$ -S /bin/bash

export PYTHONBIN=python # change as needed


# Example PPMI data
input_data_folder=../example_data/subjects_small

output_folder=../outputs
mkdir -p ${output_folder}/data

h5_path=${output_folder}/data/data.h5

# Save data for control group to .h5
script_path=../src/scripts/prepare_h5_data.py
${PYTHONBIN} ${script_path} -i ${input_data_folder} \
                            -o ${h5_path} \
                            -features fa md rd ad \
                            -split control \
                            -mode w

# Save data for patient group to the same .h5 (append mode)
${PYTHONBIN} ${script_path} -i ${input_data_folder} \
                            -o ${h5_path} \
                            -features fa md rd ad \
                            -split patient \
                            -mode a

# Save streamline count for each bundle in .h5
script_path=../src/scripts/compute_h5_sample_weight.py
${PYTHONBIN} ${script_path} -i ${h5_path} \
                            --no-save_prob