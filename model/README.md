# Model Training

This folder is greatly adoptered from the [LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory), please follow the original repository to setup the environment, see [Installation](https://llama2-accessory.readthedocs.io/en/latest/install.html).

## Finetuning

If you want to reproduce the reported performance, you need:

1. Obtained the pretrianed SPHINX-1K from the [Huggingface](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/tree/main/finetune/mm/SPHINX/SPHINX-1k)
2. Generated the training samples with our `data_gen` and update the config file content in `configs/a3vlm.yaml`
3. Modify the cooresponding parameter in the `scripts/a3vlm_train.sh`
4. Run!

## Inference

1. Obtained our provided Ckpt in Huggingface
    1. If you want to get the number reported in our paper, obtained the Ckpt from [A3VLM-Eval](https://huggingface.co/SiyuanH/A3VLM-Eval/settings), where we keep unseen classes out to conduct the evaluation.
    2. If you want to deploy or test it with more diverse images, obtained the Ckpt from [A3VLM](https://huggingface.co/SiyuanH/A3VLM).
2. Modify the parameter in 'scripts/a3vlm_infer.sh'
3. Create a VQA Json to include your tasks and update the json path in 'eval_affordance_v2.py'
4. Run the shell!

## Inference with Quant

If you have limited GPU memory, please check the 'eval_affordance_with_quant.py'

## Inference with 7B
When you have limited GPU (even with quant, still cannot afford), you can check our 7B model with InternLM as the language backbone [Huggingface](https://huggingface.co/SiyuanH/A3VLM7B). Remember set the "model_type" to be "internlm_ems5_light" in the script. And modify the lines 41 with 47 in accessory/model/LLM/inmternlm_ems5_light.py yp point to the actual path.
