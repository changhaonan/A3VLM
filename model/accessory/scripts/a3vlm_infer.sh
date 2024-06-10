remove_space=True
pretrained_path=PATH_TO_PRETRAIN
addition_flag=affordance_v6_rgb_8points_wo_unseen

export TRANSFORMERS_CACHE=/mnt/petrelfs/huangsiyuan/.cache/huggingface
export HF_HOME=/mnt/petrelfs/huangsiyuan/.cache/huggingface
export HF_HUB_OFFLINE=1
export TORCH_HOME=/mnt/petrelfs/huangsiyuan/.cache/torch
export LD_LIBRARY_PATH=/mnt/petrelfs/huangsiyuan/anaconda3/envs/accessory/lib


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

MODEL=llama_ens5 
config=resample_args_query.json

llama_path="/mnt/petrelfs/huangsiyuan/data/llama2"
llama_config="$llama_path"/13B/params.json
tokenizer_path=$PATH_TO_PRETRAIN/tokenizer.model

export OMP_NUM_THREADS=8
export NCCL_LL_THRESHOLD=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

vqa_data=demo
srun -p  Gveval-S1 -N 1 --gres=gpu:2 --quotatype=auto  --job-name=${vqa_data} \
torchrun --nproc-per-node=2 --master_port=$PORT eval_affordance_v2.py \
    --llama_type ${MODEL} \
    --llama_config ${llama_config} \
    --tokenizer_path ${tokenizer_path} \
    --pretrained_path ${pretrained_path} \
    --dataset ${vqa_data} \
    --batch_size 8 --input_size 448 \
    --model_parallel_size 2 --addition_flag ${addition_flag} --remove_space 