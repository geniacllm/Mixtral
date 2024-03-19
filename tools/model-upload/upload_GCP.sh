#!/bin/bash

set -e

start=1000
end=1000
increment=250
ucllm_nedo_dev="${HOME}/moe-recipes"
tokenizer_dir=${ucllm_nedo_dev}/tokenizer_model_directory
upload_base_dir=${ucllm_nedo_dev}/huggingface

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)
  cp -r $tokenizer_dir/tokenizer* $upload_dir

  python ${ucllm_nedo_dev}/tools/model-upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name ks5531/testGCP
done
