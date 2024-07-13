remove_space=True
pretrained_path=/mnt/lustre/huangsiyuan/LLaMA2-Accessory/accessory/output/finetune/mm/affordance_v7_rgb_8points_full_7B/A3VLM7B
addition_flag=affordance_v6_rgb_8points_internlm7b_test

export TRANSFORMERS_CACHE=/mnt/petrelfs/huangsiyuan/.cache/huggingface
export HF_HOME=/mnt/petrelfs/huangsiyuan/.cache/huggingface
# NOTE the HF_HUB_OFFLINE is only for my personal usage
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

MODEL=internlm_ems5_light 
tokenizer_path="/mnt/lustre/huangsiyuan/LLaMA2-Accessory/accessory/output/finetune/mm/affordance_v7_rgb_8points_full_7B/A3VLM7B/tokenizer.model"

export OMP_NUM_THREADS=8
export NCCL_LL_THRESHOLD=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

sample_num=2011
vqa_data=/mnt/lustre/huangsiyuan/A3VLM/model/accessory/demo_data/demo.json
srun -p  test_s2 -N 1 --gres=gpu:1 --quotatype=auto  --job-name=${vqa_data} \
torchrun --nproc-per-node=1 --master_port=$PORT eval_affordance_v2.py \
    --llama_type ${MODEL} \
    --tokenizer_path ${tokenizer_path} \
    --pretrained_path ${pretrained_path} \
    --dataset ${vqa_data} \
    --batch_size 1 --input_size 1024 \
    --model_parallel_size 1 --addition_flag ${addition_flag} --remove_space --sampled_num ${sample_num}