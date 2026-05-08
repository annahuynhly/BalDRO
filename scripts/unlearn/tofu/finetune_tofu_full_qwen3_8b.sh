export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen3-8B"
BASE_MODEL="Qwen/Qwen3-8B"
SPLIT="full"

LR="1e-5"
BSZ=1
GRAD_ACC=32
EPOCHS=5

TASK_NAME="tofu_${MODEL}_${SPLIT}"
OUTPUT_DIR="./saves/train/${TASK_NAME}"

python src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/default \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=${BASE_MODEL} \
    model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL} \
    ++model.model_args.attn_implementation=eager \
    "data.train.TOFU_QA_full.args.hf_args.name=${SPLIT}" \
    task_name=${TASK_NAME} \
    paths.output_dir="${OUTPUT_DIR}" \
    do_save=True \
    forget_split=forget01 \
    holdout_split=holdout01 \
    eval.tofu.retain_logs_path=null \
    trainer.args.report_to=none \
    trainer.args.learning_rate=${LR} \
    trainer.args.per_device_train_batch_size=${BSZ} \
    trainer.args.gradient_accumulation_steps=${GRAD_ACC} \
    trainer.args.num_train_epochs=${EPOCHS} \
    trainer.args.gradient_checkpointing=true \
    trainer.args.optim=paged_adamw_8bit \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=False
