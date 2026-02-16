# scope-noise

Simplex 3D noise generator for Daydream Scope, inspired by TouchDesigner's Noise TOP.

## What it does

Generates animated Simplex 3D noise and blends it with input video. Features time-based Z-axis animation for smooth, evolving noise patterns.

## Installation

### From Git (for sharing)

```bash
# In Scope Settings > Plugins, install from:
git+https://github.com/YOUR_USERNAME/scope-noise.git
```

### Local Development

```bash
# In Scope Settings > Plugins, browse to:
/Users/andrew/Desktop/scope local/scope-noise
```

Click **Install** and Scope will restart with the plugin loaded.

## Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| Monochrome | Bool | - | True | Generate grayscale (True) or colored noise (False) |
| Period | Float | 0.01 - 10.0 | 1.0 | Scale of noise pattern (smaller = more zoomed in) |
| Amplitude | Float | 0.0 - 2.0 | 1.0 | Intensity of the noise |
| Offset X | Float | -10.0 - 10.0 | 0.0 | Horizontal offset in noise space |
| Offset Y | Float | -10.0 - 10.0 | 0.0 | Vertical offset in noise space |
| Blend | Float | 0.0 - 1.0 | 0.5 | Mix amount (0 = original, 1 = full noise) |
| Z Speed | Float | 0.0 - 2.0 | 0.1 | Speed of Z animation (z = speed × time) |

All parameters are **runtime** - adjust them live during streaming!

## Usage

1. Connect a video source (camera or file)
2. Select **Noise** from the pipeline dropdown
3. Adjust **Period** to control noise scale
4. Set **Z Speed** to animate through noise space
5. Use **Blend** to mix noise with original video
6. Toggle **Monochrome** for grayscale or colored noise

## How it works

**Simplex 3D Noise:**
- Generates coherent 3D noise using Kenneth Perlin's Simplex algorithm
- Samples the noise field at (x, y, z) coordinates
- Z coordinate animates based on time: `z = z_speed * time_seconds`

**Blending:**
```python
result = original_video * (1 - blend) + noise * blend
```

**Colored vs Monochrome:**
- Monochrome: Single noise field repeated to RGB
- Colored: Separate noise fields for R, G, B channels (with spatial offsets)

## Development

After editing the code:
1. Go to Settings > Plugins
2. Click **Reload** next to "scope-noise"
3. Changes take effect immediately (no reinstall needed)

## TouchDesigner Comparison

This plugin recreates TouchDesigner's Noise TOP with:
- ✅ Simplex 3D noise mode
- ✅ Noise page parameters (Period, Amplitude, Offset, Monochrome)
- ✅ Transform Z animation
- ✅ Real-time blending with video input

Simplified from TD's full feature set for core functionality.
