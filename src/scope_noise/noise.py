"""Improved Perlin noise implementation for PyTorch."""

import torch


# Classic Perlin noise permutation table
PERMUTATION = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
]

# Double the permutation table and convert to tensor
P = torch.tensor(PERMUTATION + PERMUTATION, dtype=torch.long)


def fade(t: torch.Tensor) -> torch.Tensor:
    """Improved fade function: 6t^5 - 15t^4 + 10t^3"""
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear interpolation."""
    return a + t * (b - a)


def grad(hash_val: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Convert hash value to gradient and compute dot product.
    Uses the 12 edge gradients of a cube.
    """
    h = hash_val & 15

    # Select one of 12 gradient directions
    u = torch.where(h < 8, x, y)
    v = torch.where(h < 4, y, torch.where((h == 12) | (h == 14), x, z))

    # Apply sign based on hash bits
    result = torch.where((h & 1) == 0, u, -u)
    result = result + torch.where((h & 2) == 0, v, -v)

    return result


def perlin_noise_3d(coords: torch.Tensor) -> torch.Tensor:
    """
    Generate 3D Perlin noise using the classic algorithm.

    Args:
        coords: Tensor of shape (..., 3) with XYZ coordinates

    Returns:
        Tensor of shape (...) with noise values roughly in range [-1, 1]
    """
    device = coords.device
    P_device = P.to(device)

    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    # Find unit cube containing point
    X = torch.floor(x).long() & 255
    Y = torch.floor(y).long() & 255
    Z = torch.floor(z).long() & 255

    # Find relative position in cube
    x = x - torch.floor(x)
    y = y - torch.floor(y)
    z = z - torch.floor(z)

    # Compute fade curves
    u = fade(x)
    v = fade(y)
    w = fade(z)

    # Hash coordinates of 8 cube corners using permutation table
    A = P_device[X] + Y
    AA = P_device[A] + Z
    AB = P_device[A + 1] + Z
    B = P_device[X + 1] + Y
    BA = P_device[B] + Z
    BB = P_device[B + 1] + Z

    # Blend results from 8 corners of cube
    result = lerp(
        lerp(
            lerp(grad(P_device[AA], x, y, z), grad(P_device[BA], x - 1, y, z), u),
            lerp(grad(P_device[AB], x, y - 1, z), grad(P_device[BB], x - 1, y - 1, z), u),
            v
        ),
        lerp(
            lerp(grad(P_device[AA + 1], x, y, z - 1), grad(P_device[BA + 1], x - 1, y, z - 1), u),
            lerp(grad(P_device[AB + 1], x, y - 1, z - 1), grad(P_device[BB + 1], x - 1, y - 1, z - 1), u),
            v
        ),
        w
    )

    return result


def generate_noise_field(
    shape: tuple,
    period: float = 1.0,
    amplitude: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    octaves: int = 1,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Generate a 2D field of fractal Perlin 3D noise.

    Args:
        shape: (height, width) of the output field
        period: Scale of the noise pattern (smaller = more detail)
        amplitude: Intensity of the noise
        offset_x, offset_y, offset_z: Position in noise space
        octaves: Number of noise layers (harmonics)
        lacunarity: Frequency multiplier between octaves
        gain: Amplitude multiplier between octaves (persistence)
        device: Torch device

    Returns:
        Tensor of shape (height, width) with noise values in range [0, 1]
    """
    H, W = shape

    # Create coordinate grid
    y_coords = torch.linspace(0, 1, H, device=device)
    x_coords = torch.linspace(0, 1, W, device=device)

    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Initialize result
    result = torch.zeros_like(xx)

    # Fractal noise - layer multiple octaves
    frequency = 1.0
    octave_amplitude = 1.0
    max_value = 0.0  # Track max for normalization

    for octave in range(octaves):
        # Scale coordinates for this octave
        scale = (8.0 / period) * frequency
        xx_octave = (xx * scale) + offset_x
        yy_octave = (yy * scale) + offset_y
        zz_octave = torch.full_like(xx, offset_z * frequency)

        # Stack coordinates
        coords = torch.stack([xx_octave, yy_octave, zz_octave], dim=-1)

        # Generate noise for this octave
        noise = perlin_noise_3d(coords)

        # Add to result with current amplitude
        result += noise * octave_amplitude

        # Track max possible value for normalization
        max_value += octave_amplitude

        # Update frequency and amplitude for next octave
        frequency *= lacunarity
        octave_amplitude *= gain

    # Normalize to [-1, 1] based on theoretical max
    if max_value > 0:
        result = result / max_value

    # Scale by amplitude and normalize to [0, 1]
    result = (result + 1.0) / 2.0  # Normalize to [0, 1]
    result = result * amplitude  # Scale by amplitude

    return result.clamp(0, 1)
