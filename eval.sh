model_name=llama3.1
task=cnndm
control_mode=off
prefix=none

if echo "$task" | grep -q "cnndm"; then
    python rouge_utils.py --model_name $model_name --task $task --control_mode $control_mode --prefix $prefix
    python bertscore_utils.py --model_name $model_name --task $task --control_mode $control_mode --prefix $prefix
else
    CUDA_VISIBLE_DEVICES=0 python llmeval_utils.py --model_name $model_name --task $task --control_mode $control_mode --prefix $prefix
fi

python length_utils.py --model_name $model_name --task $task --control_mode $control_mode --prefix $prefix
