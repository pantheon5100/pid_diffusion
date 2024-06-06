"""
Train a diffusion model on images.
"""

import argparse

from cm import dist_util, logger
from cm.image_datasets import load_data
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_one_shot_edmedm_model_and_diffusion,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util import ODETrainLoop
import torch.distributed as dist
import copy
import torch

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )
    
    model_and_diffusion_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
    if args.training_mode == "progdist":
        distillation = False
        model_and_diffusion_kwargs["distillation"] = distillation
        model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
        
    elif "consistency" in args.training_mode:
        distillation = True
        model_and_diffusion_kwargs["distillation"] = distillation
        model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
        
    elif args.training_mode == "one_shot_pinn_edm_edm":
        student_model_and_diffusion_kwargs = copy.deepcopy(model_and_diffusion_kwargs)


        student_model_and_diffusion_kwargs["random_init"] = False

        model, diffusion = create_one_shot_edmedm_model_and_diffusion(**student_model_and_diffusion_kwargs)

    else:
        raise ValueError(f"unknown training mode {args.training_mode}")


    model.to(dist_util.dev())
    model.train()

    if args.use_fp16:
        model.convert_to_fp16()

    # A distribution over timesteps in the diffusion process, intended to reduce
    # variance of the objective.
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size
    # batch_size = 2048 which is the batch_size for each gpu

    data = None

    if len(args.teacher_model_path) > 0:  # path to the teacher score model.
        logger.log(f"loading the teacher model from {args.teacher_model_path}")
        teacher_model_and_diffusion_kwargs = copy.deepcopy(model_and_diffusion_kwargs)
        teacher_model_and_diffusion_kwargs["dropout"] = args.teacher_dropout # 0.1
        teacher_model_and_diffusion_kwargs["distillation"] = False
        teacher_model_and_diffusion_kwargs["random_init"] = False

        if args.training_mode == "one_shot_pinn_edm_edm":
            teacher_model_and_diffusion_kwargs["teacher_precond"] = True
            teacher_model, teacher_diffusion = create_one_shot_edmedm_model_and_diffusion(**teacher_model_and_diffusion_kwargs)
            load_weights = dist_util.load_state_dict(args.teacher_model_path, map_location="cpu")
            new_state_dict = {}
            for k, v in load_weights.items():
                if "map_augment" in k:
                    continue
                new_key = k.replace("model.", "")
                new_state_dict[new_key] = v
            teacher_model.load_state_dict(new_state_dict)

            model.load_state_dict(new_state_dict)
                  
        else:
            teacher_model, teacher_diffusion = create_model_and_diffusion(
                **teacher_model_and_diffusion_kwargs,
            )
            teacher_model.load_state_dict(
                dist_util.load_state_dict(args.teacher_model_path, map_location="cpu"),
            )

        teacher_model.to(dist_util.dev())
        teacher_model.eval()


        if args.use_fp16:
            teacher_model.convert_to_fp16()

    else:
        teacher_model = None
        teacher_diffusion = None

    logger.log("training...")
    ODETrainLoop(
        model=model,
        teacher_model=teacher_model,
        teacher_diffusion=teacher_diffusion,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps, # 600000
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch, # -1
        lr=args.lr, # 1e-4
        ema_rate=args.ema_rate, # 0.999,0.9999,0.9999432189950708
        log_interval=args.log_interval, # 10
        save_interval=args.save_interval, # 10k
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16, # True
        fp16_scale_growth=args.fp16_scale_growth, # 1e-3
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay, # 0
        lr_anneal_steps=args.lr_anneal_steps, # 0
        methodology=args.methodology,
        optimizer=args.optimizer,
        opt_eps=args.opt_eps,
        eval_interval=args.eval_interval,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="../cifar10/train",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        methodology="Euler",
        use_target_model=False,
        optimizer='radam',
        opt_eps=1e-8,
        eval_interval=10000,
        random_init_stu=False,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
