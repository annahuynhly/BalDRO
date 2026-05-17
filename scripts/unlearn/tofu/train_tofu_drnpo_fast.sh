export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPORTTO="none"

MODEL="Llama-2-7b-chat-hf"
TRAINER="DrNPO"
PRETRAINED_PATH="open-unlearning/tofu_Llama-2-7b-chat-hf_full"

forget_split="forget01"
holdout_split="holdout01"
retain_split="retain99"

LR="5e-5"
BSZ=2
GRAD_ACC=8
EPOCHS=2
BETA_DV=2.0

SUFFIX="lr${LR}_b${BSZ}_ga${GRAD_ACC}_betaDV${BETA_DV}_e${EPOCHS}_fast"
TASK_NAME="unlearn_tofu_${MODEL}_${forget_split}_${TRAINER}_${SUFFIX}"
OUTPUT_DIR="./saves/unlearn/tofu/${forget_split}/${MODEL}/${TRAINER}/${SUFFIX}"

# Requires: saves/eval/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json
# This should already exist from previous NPO runs on Llama-2-7b-chat-hf.

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
    trainer.args.eval_strategy=no \
    trainer.method_args.beta_dv_forget=${BETA_DV} \
    trainer.method_args.beta_dv_retain=1.0 \
    trainer.method_args.forget_dro=True \
    trainer.method_args.retain_dro=False
