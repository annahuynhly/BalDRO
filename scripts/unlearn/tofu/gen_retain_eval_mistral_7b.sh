export CUDA_VISIBLE_DEVICES=0

MODEL="Mistral-7B-Instruct-v0.3"
retain_split="retain99"
RETAIN_PATH="./saves/train/tofu_Mistral-7B-Instruct-v0.3_retain99"

# Generates: saves/eval/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json
# Required before running NPO/DrNPO training scripts.

python src/eval.py --config-name=eval.yaml \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=${RETAIN_PATH} \
    model.tokenizer_args.pretrained_model_name_or_path=${RETAIN_PATH} \
    ++model.model_args.attn_implementation=eager \
    eval.tofu.forget_split=forget01 \
    eval.tofu.holdout_split=holdout01 \
    eval.tofu.retain_logs_path=null \
    task_name=tofu_${MODEL}_${retain_split}
