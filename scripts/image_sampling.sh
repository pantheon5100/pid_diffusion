#########################################################################
# Image generation for ImageNet edmedm model
#########################################################################

# OPENAI_LOGDIR=./experiment/image_sampling/ImageNetEDM mpirun -np 1 python ./scripts/image_sample.py \
#     --training_mode one_shot_pinn_edm_edm_teacher \
#     --batch_size 128 \
#     --sigma_max 80 \
#     --sigma_min 0.002 \
#     --s_churn 0 \
#     --steps 79 \
#     --sampler heun_deter \
#     --model_path ./model_zoo/edm-imagenet-64x64-cond-adm.ckpt \
#     --attention_resolutions 32,16,8  \
#     --class_cond True \
#     --dropout 0.1 \
#     --image_size 64 \
#     --num_channels 192 \
#     --num_head_channels 64 \
#     --num_res_blocks 3 \
#     --num_samples 128 \
#     --resblock_updown True \
#     --use_fp16 False \
#     --use_scale_shift_norm True \
#     --weight_schedule uniform
#########################################################################

#########################################################################
# Image generation for ImageNet oneshot model
#########################################################################

# OPENAI_LOGDIR=./experiment/image_sampling/ImageNetPID mpirun -np 1 python ./scripts/image_sample.py \
#     --training_mode one_shot_pinn_edm_edm_one_shot \
#     --batch_size 128 \
#     --sigma_max 80 \
#     --sigma_min 0.002 \
#     --s_churn 0 \
#     --steps 79 \
#     --sampler oneshot \
#     --model_path ./model_zoo/edm-imagenet-64x64-cond-adm.ckpt \
#     --attention_resolutions 32,16,8  \
#     --class_cond True \
#     --dropout 0.1 \
#     --image_size 64 \
#     --num_channels 192 \
#     --num_head_channels 64 \
#     --num_res_blocks 3 \
#     --num_samples 128 \
#     --resblock_updown True \
#     --use_fp16 False \
#     --use_scale_shift_norm True \
#     --weight_schedule uniform

#########################################################################


#########################################################################
# Image generation for cifar edmedm model
#########################################################################

OPENAI_LOGDIR=./experiment/image_sampling/CIFAREDM mpirun -np 1 python ./scripts/image_sample.py \
    --training_mode one_shot_pinn_edm_edm_teacher \
    --batch_size 128 \
    --sigma_max 80 \
    --sigma_min 0.002 \
    --s_churn 0 \
    --steps 35 \
    --sampler heun_deter \
    --model_path ./model_zoo/edm-cifar10-32x32-uncond-vp.ckpt \
    --attention_resolutions "2"  \
    --class_cond False \
    --dropout 0.0 \
    --image_size 32 \
    --num_channels 192 \
    --num_channels 128 \
    --num_res_blocks 4 \
    --num_samples 128 \
    --resblock_updown True \
    --use_fp16 False \
    --use_scale_shift_norm True \
    --weight_schedule uniform
#########################################################################


#########################################################################
# Image generation for cifar oneshot model
#########################################################################

# OPENAI_LOGDIR=./experiment/image_sampling/CIFARPID mpirun -np 1 python ./scripts/image_sample.py \
#     --training_mode one_shot_pinn_edm_edm_one_shot \
#     --batch_size 128 \
#     --sigma_max 80 \
#     --sigma_min 0.002 \
#     --s_churn 0 \
#     --steps 35 \
#     --sampler oneshot \
#     --model_path ./model_zoo/edm-cifar10-32x32-uncond-vp.ckpt \
#     --attention_resolutions "2"  \
#     --class_cond False \
#     --dropout 0.0 \
#     --image_size 32 \
#     --num_channels 192 \
#     --num_channels 128 \
#     --num_res_blocks 4 \
#     --num_samples 128 \
#     --resblock_updown True \
#     --use_fp16 False \
#     --use_scale_shift_norm True \
#     --weight_schedule uniform
#########################################################################

