import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from memo.inference import MemoInferenceModels, inference


logger = logging.getLogger("memo")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for MEMO")

    parser.add_argument("--config", type=Path, default="configs/inference.yaml")
    parser.add_argument("--input_image", type=Path)
    parser.add_argument("--input_audio", type=Path)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Loading config from {args.config}")
    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif config.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        weight_dtype = torch.float32

    inference(
        input_image_path=args.input_image.absolute(),
        input_audio_path=args.input_audio.absolute(),
        output_video_path=args.output_path.absolute(),
        overwrite_output=False,
        seed=args.seed,
        weight_dtype=weight_dtype,
        output_resolution=config.resolution,
        fps=config.fps,
        num_generated_frames_per_clip=config.num_generated_frames_per_clip,
        num_init_past_frames=config.num_init_past_frames,
        num_past_frames=config.num_past_frames,
        inference_steps=config.inference_steps,
        cfg_scale=config.cfg_scale,
        enable_xformers_memory_efficient_attention=config.enable_xformers_memory_efficient_attention,
        models=MemoInferenceModels(
            memo_model=config.model_name_or_path,
            vae_model=config.vae,
            wav2vec_model=config.wav2vec,
            emotion2vec_model=config.emotion2vec,
            models_dir=Path("checkpoints").absolute(),
        ),
    )
