#!/bin/bash

# 这是一个标准的启动脚本
# 作用：调用 LLaMA-Factory 开始微调 Qwen2.5 模型

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /gemini/code/model/Qwen2___5-7B-Instruct \
    --dataset mooc_agent \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \
    --output_dir saves/Qwen2.5-7B/mooc_agent_v1 \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --fp16 \
    --lora_rank 32 \
    --lora_alpha 64