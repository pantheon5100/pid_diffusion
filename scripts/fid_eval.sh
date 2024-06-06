
##################################################################
# For CIFAR model evaluation
##################################################################

EXP_PATH="./model_zoo/pid_cifar"

mpirun -np 1 python ./scripts/fid_evaluation.py \
    --training_mode one_shot_pinn_edm_edm_one_shot \
    --fid_dataset cifar10 \
    --exp_dir $EXP_PATH\
    --batch_size 125 \
    --sigma_max 80 \
    --sigma_min 0.002 \
    --s_churn 0 \
    --steps 35 \
    --sampler oneshot \
    --attention_resolutions "2"  \
    --class_cond False \
    --dropout 0.0 \
    --image_size 32 \
    --num_channels 128 \
    --num_res_blocks 4 \
    --num_samples 50000 \
    --resblock_updown True \
    --use_fp16 False \
    --use_scale_shift_norm True \
    --weight_schedule uniform \
    --seed 0

##################################################################


##################################################################
# For ImageNet model evaluation
##################################################################

# EXP_PATH="./experiment/pid_imagenet"

# mpirun -np 1 python ./scripts/fid_evaluation.py \
#     --training_mode one_shot_pinn_edm_edm_one_shot \
#     --fid_dataset imagenet \
#     --exp_dir $EXP_PATH\
#     --batch_size 250 \
#     --sigma_max 80 \
#     --sigma_min 0.002 \
#     --s_churn 0 \
#     --sampler oneshot \
#     --attention_resolutions 32,16,8  \
#     --class_cond True \
#     --dropout 0.0 \
#     --image_size 64 \
#     --num_channels 192 \
#     --num_head_channels 64 \
#     --num_res_blocks 3 \
#     --num_samples 50000 \
#     --resblock_updown True \
#     --use_fp16 True \
#     --use_scale_shift_norm True \
#     --weight_schedule uniform

##################################################################
