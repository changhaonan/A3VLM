remove_space=True
pretrained_path=PATH_TO_PRETRAIN
addition_flag=affordance_v6_rgb_8points_wo_unseen_7B

export TRANSFORMERS_CACHE=/mnt/petrelfs/XXXXX/.cache/huggingface
export HF_HOME=/mnt/petrelfs/XXXXX/.cache/huggingface
export HF_HUB_OFFLINE=1
export TORCH_HOME=/mnt/petrelfs/XXXXX/.cache/torch
export LD_LIBRARY_PATH=/mnt/petrelfs/XXXXX/anaconda3/envs/accessory/lib


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

MODEL=internlm_ems5_light 
config=resample_args_query.json

llama_path="/mnt/petrelfs/XXXXX/data/llama2"
llama_config="$llama_path"/7B/params.json
tokenizer_path=$PATH_TO_PRETRAIN/tokenizer.model

export OMP_NUM_THREADS=8
export NCCL_LL_THRESHOLD=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

vqa_data=demo
torchrun --nproc-per-node=1 --master_port=$PORT eval_affordance_v2.py \
    --llama_type ${MODEL} \
    --llama_config ${llama_config} \
    --tokenizer_path ${tokenizer_path} \
    --pretrained_path ${pretrained_path} \
    --dataset ${vqa_data} \
    --batch_size 8 --input_size 448 \
    --model_parallel_size 2 --addition_flag ${addition_flag} --remove_space 