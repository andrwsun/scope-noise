from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, ui_field_config


class NoiseConfig(BasePipelineConfig):
    """Configuration for the Noise pipeline."""

    pipeline_id = "noise"
    pipeline_name = "Noise"
    pipeline_description = (
        "Simplex 3D noise generator inspired by TouchDesigner's Noise TOP. "
        "Blends animated noise with input video."
    )

    supports_prompts = False

    modes = {"video": ModeDefaults(default=True)}

    # --- Noise Parameters ---

    monochrome: bool = Field(
        default=True,
        description="Generate grayscale noise (True) or colored noise per RGB channel (False)",
        json_schema_extra=ui_field_config(order=1, label="Monochrome"),
    )

    period: float = Field(
        default=1.0,
        ge=0.01,
        le=10.0,
        description="Scale of the noise pattern (smaller = more zoomed in)",
        json_schema_extra=ui_field_config(order=2, label="Period"),
    )

    amplitude: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Intensity of the noise (0 = no noise, 2 = maximum)",
        json_schema_extra=ui_field_config(order=3, label="Amplitude"),
    )

    offset_x: float = Field(
        default=0.0,
        ge=-10.0,
        le=10.0,
        description="Horizontal offset in noise space",
        json_schema_extra=ui_field_config(order=4, label="Offset X"),
    )

    offset_y: float = Field(
        default=0.0,
        ge=-10.0,
        le=10.0,
        description="Vertical offset in noise space",
        json_schema_extra=ui_field_config(order=5, label="Offset Y"),
    )

    blend: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Blend amount (0 = original video, 1 = full noise)",
        json_schema_extra=ui_field_config(order=6, label="Blend"),
    )

    # --- Transform Parameters ---

    # --- Harmonics (Fractal Noise) ---

    octaves: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of noise layers (harmonics) - more octaves = more detail",
        json_schema_extra=ui_field_config(order=7, label="Octaves"),
    )

    lacunarity: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="Frequency multiplier between octaves (typically 2.0)",
        json_schema_extra=ui_field_config(order=8, label="Lacunarity"),
    )

    gain: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Amplitude multiplier between octaves (persistence)",
        json_schema_extra=ui_field_config(order=9, label="Gain"),
    )

    # --- Transform Parameters ---

    z_speed: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Speed of Z-axis animation through noise space (z = speed * time)",
        json_schema_extra=ui_field_config(order=10, label="Z Speed"),
    )
