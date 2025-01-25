#--trials 5
#--beam 16
CUDA_VISIBLE_DEVICES=1 python MetropolisHastings.py\
    --model_name llama3.1 \
    --task cnndm \
    --control_mode off \
    --batch_size 10 


CUDA_VISIBLE_DEVICES=1 python MetropolisHastings.py\
    --model_name llama3.1 \
    --task cnndm \
    --control_mode on \
    --batch_size 10 \
    --trials 5 \
    --beam 16
