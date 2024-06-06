"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    create_one_shot_edmedm_model_and_diffusion,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample
import torchvision
import PIL
from cleanfid.features import build_feature_extractor, get_reference_statistics
import scipy.linalg
import glob
import csv
from pathlib import Path
import pandas

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

def main(args, model=None, diffusion=None):

    if model==None or diffusion==None:

        if "one_shot_pinn_edm_edm_teacher" == args.training_mode:
            model, diffusion = create_one_shot_edmedm_model_and_diffusion(teacher_precond=True,
                **args_to_dict(args, model_and_diffusion_defaults().keys()),
            )
            load_weights = dist_util.load_state_dict(args.model_path, map_location="cpu")
            new_state_dict = {}
            for k, v in load_weights.items():
                if "map_augment" in k:
                    continue
                new_key = k.replace("model.", "")
                new_state_dict[new_key] = v
            
            model.load_state_dict(new_state_dict)
        elif "one_shot_pinn_edm_edm_one_shot" == args.training_mode:
            model, diffusion = create_one_shot_edmedm_model_and_diffusion(teacher_precond=False,
                **args_to_dict(args, model_and_diffusion_defaults().keys()),
            )
            load_weights = dist_util.load_state_dict(args.model_path, map_location="cpu")
            new_state_dict = {}
            for k, v in load_weights.items():
                if "map_augment" in k:
                    continue
                new_key = k.replace("model.", "")
                new_state_dict[new_key] = v
            
            model.load_state_dict(new_state_dict)
        else:
            raise ValueError(f"training mode {args.training_mode} not supported")

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None
    

    mode='legacy_tensorflow'
    fid_evaluation_feat_model = build_feature_extractor(mode, dist_util.dev(), use_dataparallel=False)
    
    if dist.get_rank() == 0:
        dataset_name=args.fid_dataset
        if dataset_name == 'cifar10':
            fpath = './model_zoo/stats/cifar10-32x32.npz'
        elif dataset_name == 'imagenet':
            fpath = './model_zoo/stats/imagenet-64x64.npz'
        else:
            raise ValueError(f"fid evaluation error: not support dataset {dataset_name}.")
        stats = np.load(fpath)
        ref_mu, ref_sigma = stats["mu"], stats["sigma"]
        

    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)
    
    feature_dim = 2048
    mu = th.zeros([feature_dim], dtype=th.float64, device=dist_util.dev())
    sigma = th.zeros([feature_dim, feature_dim], dtype=th.float64, device=dist_util.dev())
    l_feats = []
    
    generated_img = False
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        sample_ori = karras_sample(
            diffusion,
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            generator=generator,
            ts=ts,
        )
        sample = ((sample_ori + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

        with th.no_grad():
            feat = fid_evaluation_feat_model(((sample_ori + 1.) * 0.5).mul(255).clip(0, 255).to(dist_util.dev())).to(th.float64)
            # l_feats.append(feat.cpu())
            mu += feat.sum(0)
            sigma += feat.T @ feat
        
        if not generated_img and dist.get_rank() == 0:
            generated_img = True
            torchvision.utils.save_image((sample_ori+1.)/2., os.path.join(logger.get_dir(), f"{args.model_name}_samples.png"), nrow = 10)
            
            
    mu = mu.unsqueeze(0)
    sigma = sigma.unsqueeze(0)

    all_mu = [th.zeros_like(mu) for _ in range(dist.get_world_size())]
    dist.all_gather(all_mu, mu)  # gather not supported with NCCL
    all_mu = th.cat(all_mu, axis=0)

    all_sigma = [th.zeros_like(sigma) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sigma, sigma)  # gather not supported with NCCL
    all_sigma = th.cat(all_sigma, axis=0)

    mu = all_mu.sum(0)
    sigma = all_sigma.sum(0)
    
    num_images = args.num_samples
    mu /= num_images
    sigma -= mu.ger(mu) * num_images
    sigma /= num_images - 1
    
    mu = mu.cpu().numpy()
    sigma = sigma.cpu().numpy()
    

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    fid_score = 0.
    if dist.get_rank() == 0:
        
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"{args.model_name}_samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
            
            
        fid_score = calculate_fid_from_inception_stats(
            mu, sigma, 
            ref_mu, 
            ref_sigma
            )

    dist.barrier()
    logger.log("sampling complete")
    return fid_score


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
        fid_dataset="cifar10",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    exp_dir = Path(args.exp_dir)
    os.environ["OPENAI_LOGDIR"] = str(exp_dir / "FID")
    
    dist_util.setup_dist()
    logger.configure()
    assert args.num_samples % (args.batch_size * dist.get_world_size()) == 0
    
    result_file = exp_dir / f"fid_results_{args.seed}.csv"
    if result_file.exists():
        logger.log("exist results, check result...")
        df = pandas.read_csv(str(result_file))
        tested_model = list(df["model_name"])
        logger.log(f"Tested model: {tested_model}")
    else:
        tested_model = []
        if dist.get_rank() == 0:
            with open(str(result_file), mode='w') as f:
                fid_writer = csv.writer(f, delimiter=',')
                fid_writer.writerow(['model_name', 'FID'])
                
    model_list_ = exp_dir.glob("*.pt")
    model_list = []
    
    for model_dir in model_list_:
        model_name = str(model_dir.stem)
        model_dir = str(model_dir)
        model_list.append(
            {
                "model_name": model_name,
                "path": model_dir,
            }
        )
    
    for model_test in model_list:
        args.model_path = model_test["path"]
        args.model_name = model_test["model_name"]
        logger.log(f"\nmodel: {args.model_name}")
        
        if args.model_name in tested_model:
            continue
        
        fid_score = main(args)
        
        logger.log(f"\nmodel: {args.model_name} \nFID {fid_score:.4f}\n")
        
        if dist.get_rank() == 0:
            with open(str(result_file), mode='a') as f:
                fid_writer = csv.writer(f, delimiter=',')
                fid_writer.writerow([args.model_name, fid_score])

