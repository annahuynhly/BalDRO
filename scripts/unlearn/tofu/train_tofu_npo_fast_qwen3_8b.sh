export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPORTTO="none"

MODEL="Qwen3-8B"
TRAINER="NPO"
PRETRAINED_PATH="$(pwd)/saves/train/tofu_Qwen3-8B_full"

forget_split="forget01"
holdout_split="holdout01"
retain_split="retain99"

LR="5e-5"
BSZ=1
GRAD_ACC=16
EPOCHS=2

SUFFIX="lr${LR}_b${BSZ}_ga${GRAD_ACC}_e${EPOCHS}_fast"
TASK_NAME="unlearn_tofu_${MODEL}_${forget_split}_${TRAINER}_${SUFFIX}"
OUTPUT_DIR="./saves/unlearn/tofu/${forget_split}/${MODEL}/${TRAINER}/${SUFFIX}"

# Requires: saves/eval/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json
# Generate with: bash scripts/unlearn/tofu/gen_retain_eval_qwen3_8b.sh

python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=${TRAINER} \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=${PRETRAINED_PATH} \
    model.tokenizer_args.pretrained_model_name_or_path=${PRETRAINED_PATH} \
    ++model.model_args.attn_implementation=eager \
    forget_split=${forget_split} \
    holdout_split=${holdout_split} \
    retain_split=${retain_split} \
    task_name=${TASK_NAME} \
    paths.output_dir="${OUTPUT_DIR}" \
    do_save=True \
    eval.tofu.retain_logs_path=./saves/eval/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json \
    trainer.args.ddp_find_unused_parameters=false \
    trainer.args.gradient_checkpointing=true \
    trainer.args.report_to=${REPORTTO} \
    trainer.args.logging_steps=1 \
    trainer.args.learning_rate=${LR} \
    trainer.args.per_device_train_batch_size=${BSZ} \
    trainer.args.gradient_accumulation_steps=${GRAD_ACC} \
    trainer.args.num_train_epochs=${EPOCHS} \
    trainer.args.eval_strategy=epoch \
    trainer.args.eval_on_start=False \
    trainer.args.optim=paged_adamw_8bit
