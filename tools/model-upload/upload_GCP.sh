#!/bin/bash
#SBATCH --job-name=huggingface_job
#SBATCH --output=huggingface_job_output.txt
#SBATCH --error=huggingface_job_error.txt

set -e

export HUGGINGFACE_HUB_TOKEN='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'

start=50
end=50
increment=50
saved_model_directory="Mixtral-8x7b-GENIAC-eric-gcp-single-node-v0.2"
ucllm_nedo_dev="${HOME}/Mixtral"
tokenizer_dir=${ucllm_nedo_dev}/tokenizer_model_directory
upload_base_dir=${ucllm_nedo_dev}/${saved_model_directory}

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)
  cp -r $tokenizer_dir/tokenizer* $upload_dir

  python ${ucllm_nedo_dev}/tools/model-upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name Eric2333/Mixtral-GCP-upload-test2
done
