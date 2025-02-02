MODEL=t5-base
MAX_LENGTH=256
MAX_STEPS=30000
PREFIX_LENGTH=100
R=2
lr=6e-1
batch_size=16
TASK_NAME=sst2

CUDA_VISIBLE_DEVICES=0 python train.py \
    --peft_type PROMPT_TUNING_LODIM \
    --learning_rate ${lr} \
    --prefix_length ${PREFIX_LENGTH} \
    --task_name ${TASK_NAME} \
    --dataset_config_name en \
    --model_name_or_path ${MODEL} \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --max_seq_length ${MAX_LENGTH} \
    --save_strategy steps \
    --evaluation_strategy steps \
    --max_steps ${MAX_STEPS} \
    --eval_steps 1000 \
    --save_steps 1000 \
    --warmup_steps 500 \
    --logging_steps 100 \
    --weight_decay 1e-5 \
    --load_best_model_at_end \
    --save_total_limit 1 \
    --output_dir saved_${MODEL}/${TASK_NAME}/lodim_r${R}_lr${lr}_pl${PREFIX_LENGTH}_st${MAX_STEPS}_bs${batch_size}_ml${MAX_LENGTH} \
    --overwrite_output_dir \
    --lodim_r ${R}
