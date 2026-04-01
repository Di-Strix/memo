import argparse
import logging
import os

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from packaging import version
from tqdm import tqdm
from pathlib import Path

from memo.models.audio_proj import AudioProjModel
from memo.models.image_proj import ImageProjModel
from memo.models.unet_2d_condition import UNet2DConditionModel
from memo.models.unet_3d import UNet3DConditionModel
from memo.pipelines.video_pipeline import VideoPipeline
from memo.utils.audio_utils import extract_audio_emotion_labels, preprocess_audio, resample_audio
from memo.utils.vision_utils import preprocess_image, tensor_to_video


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


def inference(
    input_image_path: Path,
    input_audio_path: Path,
    output_video_path: Path,
    models_dir: Path,
    overwrite_output: bool = False,
    seed: int = 42,
    device: torch.device | None = None,
    weight_dtype: torch.dtype = torch.bfloat16,
    output_resolution: int = 512,
    fps: int = 30,
    num_generated_frames_per_clip: int = 16,
    num_init_past_frames: int = 16,
    num_past_frames: int = 16,
    inference_steps: int = 20,
    cfg_scale: float = 3.5,
    enable_xformers_memory_efficient_attention: bool = True,
    model_name_or_path: Path | str = "memoavatar/memo",
    vae_model: str = "stabilityai/sd-vae-ft-mse",
    wav2vec_model: str = "facebook/wav2vec2-base-960h",
    emotion2vec_model: str = "iic/emotion2vec_plus_large",
):
    assert input_image_path.is_file(), "input_image_path must point to a file"
    assert input_audio_path.is_file(), "input_audio_path must point to a file"

    if input_audio_path.suffix != ".wav":
        logger.warning("MEMO might not generate full-length video for non-wav audio file.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = output_video_path.parent
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_video_path) and not overwrite_output:
        logger.info(f"Output file {output_video_path} already exists. Skipping inference.")
        return

    generator = torch.manual_seed(seed)

    # Download face analysis and vocal separator models, if they do not exist
    face_analysis = models_dir / "misc" / "face_analysis"
    os.makedirs(face_analysis, exist_ok=True)
    for model in [
        "1k3d68.onnx",
        "2d106det.onnx",
        "face_landmarker_v2_with_blendshapes.task",
        "genderage.onnx",
        "glintr100.onnx",
        "scrfd_10g_bnkps.onnx",
    ]:
        model_path = face_analysis / "models" / model
        if not os.path.exists(model_path):
            logger.info(f"Downloading {model} to {face_analysis}/models")
            os.system(
                f"wget -P {face_analysis}/models https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/{model}"
            )
            # Check if the download was successful
            if not os.path.exists(model_path):
                raise RuntimeError(f"Failed to download {model} to {model_path}")
            # File size check
            if os.path.getsize(model_path) < 1024 * 1024:
                raise RuntimeError(f"{model_path} file seems incorrect (too small), delete it and retry.")
    logger.info(f"Use face analysis models from {face_analysis}")

    vocal_separator = models_dir / "misc" / "vocal_separator" / "Kim_Vocal_2.onnx"
    if os.path.exists(vocal_separator):
        logger.info(f"Vocal separator {vocal_separator} already exists. Skipping download.")
    else:
        logger.info(f"Downloading vocal separator to {vocal_separator}")
        os.makedirs(os.path.dirname(vocal_separator), exist_ok=True)
        os.system(
            f"wget -P {os.path.dirname(vocal_separator)} https://huggingface.co/memoavatar/memo/resolve/main/misc/vocal_separator/Kim_Vocal_2.onnx"
        )

    logger.info(f"Inference dtype: {weight_dtype}")

    logger.info(f"Processing image {input_image_path}")
    img_size = (output_resolution, output_resolution)
    pixel_values, face_emb = preprocess_image(
        face_analysis_model=str(face_analysis),
        image_path=str(input_image_path),
        image_size=output_resolution,
    )

    logger.info(f"Processing audio {input_audio_path}")
    cache_dir = output_dir / "audio_preprocess"
    os.makedirs(cache_dir, exist_ok=True)
    input_audio_path = Path(
        resample_audio(
            str(input_audio_path),
            str(cache_dir / f"{os.path.basename(input_audio_path).split('.')[0]}-16k.wav"),
        )
    )
    audio_emb, audio_length = preprocess_audio(
        wav_path=str(input_audio_path),
        num_generated_frames_per_clip=num_generated_frames_per_clip,
        fps=fps,
        wav2vec_model=wav2vec_model,
        vocal_separator_model=str(vocal_separator),
        cache_dir=str(cache_dir),
        device=device,
    )

    logger.info("Processing audio emotion")
    audio_emotion, num_emotion_classes = extract_audio_emotion_labels(
        model="memoavatar/memo",
        wav_path=str(input_audio_path),
        emotion2vec_model=emotion2vec_model,
        audio_length=audio_length,
        device=device,
    )

    logger.info("Loading models")
    vae = AutoencoderKL.from_pretrained(vae_model).to(device=device, dtype=weight_dtype)
    reference_net = UNet2DConditionModel.from_pretrained(
        model_name_or_path, subfolder="reference_net", use_safetensors=True
    )
    diffusion_net = UNet3DConditionModel.from_pretrained(
        model_name_or_path, subfolder="diffusion_net", use_safetensors=True
    )
    image_proj = ImageProjModel.from_pretrained(model_name_or_path, subfolder="image_proj", use_safetensors=True)
    audio_proj = AudioProjModel.from_pretrained(model_name_or_path, subfolder="audio_proj", use_safetensors=True)

    vae.requires_grad_(False).eval()
    reference_net.requires_grad_(False).eval()
    diffusion_net.requires_grad_(False).eval()
    image_proj.requires_grad_(False).eval()
    audio_proj.requires_grad_(False).eval()

    # Enable memory-efficient attention for xFormers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.info(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            reference_net.enable_xformers_memory_efficient_attention()
            diffusion_net.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Create inference pipeline
    noise_scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline = VideoPipeline(
        vae=vae,
        reference_net=reference_net,
        diffusion_net=diffusion_net,
        scheduler=noise_scheduler,
        image_proj=image_proj,
    )
    pipeline.to(device=device, dtype=weight_dtype)

    video_frames = []
    num_clips = audio_emb.shape[0] // num_generated_frames_per_clip
    for t in tqdm(range(num_clips), desc="Generating video clips"):
        if len(video_frames) == 0:
            # Initialize the first past frames with reference image
            past_frames = pixel_values.repeat(num_init_past_frames, 1, 1, 1)
            past_frames = past_frames.to(dtype=pixel_values.dtype, device=pixel_values.device)
            pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)
        else:
            past_frames = video_frames[-1][0]
            past_frames = past_frames.permute(1, 0, 2, 3)
            past_frames = past_frames[0 - num_past_frames :]
            past_frames = past_frames * 2.0 - 1.0
            past_frames = past_frames.to(dtype=pixel_values.dtype, device=pixel_values.device)
            pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)

        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

        audio_tensor = (
            audio_emb[
                t * num_generated_frames_per_clip : min((t + 1) * num_generated_frames_per_clip, audio_emb.shape[0])
            ]
            .unsqueeze(0)
            .to(device=audio_proj.device, dtype=audio_proj.dtype)
        )
        audio_tensor = audio_proj(audio_tensor)

        audio_emotion_tensor = audio_emotion[
            t * num_generated_frames_per_clip : min((t + 1) * num_generated_frames_per_clip, audio_emb.shape[0])
        ]

        pipeline_output = pipeline(
            ref_image=pixel_values_ref_img,
            audio_tensor=audio_tensor,
            audio_emotion=audio_emotion_tensor,
            emotion_class_num=num_emotion_classes,
            face_emb=face_emb,
            width=img_size[0],
            height=img_size[1],
            video_length=num_generated_frames_per_clip,
            num_inference_steps=inference_steps,
            guidance_scale=cfg_scale,
            generator=generator,
            is_new_audio=t == 0,
        )

        video_frames.append(pipeline_output.videos)

    video_frames = torch.cat(video_frames, dim=2)
    video_frames = video_frames.squeeze(0)
    video_frames = video_frames[:, :audio_length]

    tensor_to_video(video_frames, output_video_path, input_audio_path, fps=config.fps)


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
        models_dir=Path("checkpoints").absolute(),
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
        model_name_or_path=config.model_name_or_path,
        vae_model=config.vae,
        wav2vec_model=config.wav2vec,
        emotion2vec_model=config.emotion2vec,
    )
