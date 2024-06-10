#!/bin/bash
export TRANSFORMERS_CACHE=/mnt/petrelfs/huangsiyuan/.cache/huggingface
export HF_HOME=/mnt/petrelfs/huangsiyuan/.cache/huggingface
export HF_HUB_OFFLINE=1
export TORCH_HOME=/mnt/petrelfs/huangsiyuan/.cache/torch
export LD_LIBRARY_PATH=/mnt/petrelfs/huangsiyuan/anaconda3/envs/accessory/lib

export PATH=/mnt/lustre/share/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-11.8/lib64:$LD_LIBRARY_PATH

# setting up GCC environment
export PATH=/mnt/lustre/share/gcc/gcc-7.3.0/bin:/mnt/lustre/share/gcc/gcc-7.3.0/lib64:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/gcc/gcc-7.3.0/lib/:$LD_LIBRARY_PATH

pretrained_path=PATH_TO_PRETRAIN
pretrained_type=consolidated

llama_path=PATH_TO_LLAMA
llama_config="$llama_path"/13B/params.json

tokenizer_path="$llama_path"/13B/tokenizer.model
data_config=configs/a3vlm.yaml

data_parallel=sdp
model_parallel=2

exp=a3vlm
exp_name=finetune/mm/"$exp"
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"
mkdir -p output_dir/"$exp"


while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT
export MASTER_PORT=$PORT

srun -p test_s2 --gres=gpu:8 --cpus-per-task 12 -n16  --nodes=2 --ntasks-per-node=8 --quotatype=auto  --job-name="$exp" \
python -u main_finetune.py \
--output_dir output/"$exp_name" --epochs 3 --warmup_epochs 0.03 \
--batch_size 2 --accum_iter 8 --num_workers 4 \
--max_words 2048 \
--lr 0.00002 --min_lr 0 --clip_grad 8 --weight_decay 0 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type llama_ens5 --llama_config $llama_config --tokenizer_path "$tokenizer_path" \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--data_config $data_config --dialog \
--image_transform padded_resize --cache_ann_on_disk \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name done: $exp_name"