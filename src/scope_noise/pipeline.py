from typing import TYPE_CHECKING
import time

import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .noise import generate_noise_field
from .schema import NoiseConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


class NoisePipeline(Pipeline):
    """Perlin 3D noise generator inspired by TouchDesigner's Noise TOP."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return NoiseConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.z_offset = 0.0
        self.last_time = time.time()

    def prepare(self, **kwargs) -> Requirements:
        """We need exactly one input frame per call."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Blend Simplex 3D noise with input video frames."""
        video = kwargs.get("video")
        if video is None:
            raise ValueError("NoisePipeline requires video input")

        # Stack input frames -> (T, H, W, C) and normalize to [0, 1]
        frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)
        frames = frames.to(device=self.device, dtype=torch.float32) / 255.0

        T, H, W, C = frames.shape

        # Read runtime parameters from kwargs
        monochrome = kwargs.get("monochrome", True)
        period = kwargs.get("period", 1.0)
        amplitude = kwargs.get("amplitude", 1.0)
        offset_x = kwargs.get("offset_x", 0.0)
        offset_y = kwargs.get("offset_y", 0.0)
        blend = kwargs.get("blend", 0.5)
        z_speed = kwargs.get("z_speed", 0.1)

        # Update Z offset based on time delta and z_speed
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time

        # Increment z_offset by z_speed * delta_time (velocity-based)
        self.z_offset += z_speed * delta_time

        offset_z = self.z_offset

        # Generate noise for each frame
        noise_frames = []
        for t in range(T):
            # Generate noise field
            noise = generate_noise_field(
                shape=(H, W),
                period=period,
                amplitude=amplitude,
                offset_x=offset_x,
                offset_y=offset_y,
                offset_z=offset_z + (t * 0.01),  # Small per-frame offset for temporal coherence
                device=self.device,
            )

            if monochrome:
                # Repeat single channel to RGB
                noise_rgb = noise.unsqueeze(-1).repeat(1, 1, 3)
            else:
                # Generate different noise for each RGB channel
                noise_r = noise
                noise_g = generate_noise_field(
                    shape=(H, W),
                    period=period,
                    amplitude=amplitude,
                    offset_x=offset_x + 100,  # Offset each channel
                    offset_y=offset_y,
                    offset_z=offset_z + (t * 0.01),
                    device=self.device,
                )
                noise_b = generate_noise_field(
                    shape=(H, W),
                    period=period,
                    amplitude=amplitude,
                    offset_x=offset_x + 200,
                    offset_y=offset_y,
                    offset_z=offset_z + (t * 0.01),
                    device=self.device,
                )
                noise_rgb = torch.stack([noise_r, noise_g, noise_b], dim=-1)

            noise_frames.append(noise_rgb)

        # Stack all noise frames
        noise_video = torch.stack(noise_frames, dim=0)

        # Blend noise with input video
        result = frames * (1.0 - blend) + noise_video * blend

        return {"video": result.clamp(0, 1)}
