"""Simplex 3D noise implementation for PyTorch."""

import math

import torch


def simplex_noise_3d(coords: torch.Tensor) -> torch.Tensor:
    """
    Generate 3D Simplex noise.

    Args:
        coords: Tensor of shape (..., 3) with XYZ coordinates

    Returns:
        Tensor of shape (...) with noise values in range [-1, 1]
    """
    # Simplex 3D implementation based on Ken Perlin's improved noise
    # This is a simplified version optimized for GPU

    # Skewing and unskewing factors for 3D
    F3 = 1.0 / 3.0
    G3 = 1.0 / 6.0

    # Get input coordinates
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    # Skew the input space to determine which simplex cell we're in
    s = (x + y + z) * F3
    i = torch.floor(x + s)
    j = torch.floor(y + s)
    k = torch.floor(z + s)

    t = (i + j + k) * G3
    X0 = i - t
    Y0 = j - t
    Z0 = k - t

    x0 = x - X0
    y0 = y - Y0
    z0 = z - Z0

    # Determine which simplex we are in
    mask_xy = x0 >= y0
    mask_xz = x0 >= z0
    mask_yz = y0 >= z0

    i1 = torch.zeros_like(x0)
    j1 = torch.zeros_like(y0)
    k1 = torch.zeros_like(z0)

    i2 = torch.ones_like(x0)
    j2 = torch.ones_like(y0)
    k2 = torch.ones_like(z0)

    # X Y Z order
    mask = mask_xy & mask_xz
    i1 = torch.where(mask, torch.ones_like(i1), i1)
    i2 = torch.where(mask, torch.ones_like(i2), i2)
    j2 = torch.where(mask, torch.ones_like(j2), j2)
    k2 = torch.where(mask, torch.zeros_like(k2), k2)

    # X Z Y order
    mask = (~mask_xy) & mask_xz
    i1 = torch.where(mask, torch.ones_like(i1), i1)
    j1 = torch.where(mask, torch.zeros_like(j1), j1)
    k1 = torch.where(mask, torch.ones_like(k1), k1)

    # Z X Y order
    mask = (~mask_xy) & (~mask_xz)
    j1 = torch.where(mask, torch.zeros_like(j1), j1)
    k1 = torch.where(mask, torch.ones_like(k1), k1)
    i2 = torch.where(mask, torch.ones_like(i2), i2)
    j2 = torch.where(mask, torch.zeros_like(j2), j2)

    # Y Z X order
    mask = mask_xy & (~mask_yz)
    i1 = torch.where(mask, torch.zeros_like(i1), i1)
    j1 = torch.where(mask, torch.ones_like(j1), j1)
    k2 = torch.where(mask, torch.ones_like(k2), k2)

    # Z Y X order
    mask = (~mask_xy) & (~mask_yz)
    k1 = torch.where(mask, torch.ones_like(k1), k1)

    # Y X Z order
    mask = mask_xy & mask_yz
    j1 = torch.where(mask, torch.ones_like(j1), j1)

    # Offsets for remaining corners
    x1 = x0 - i1 + G3
    y1 = y0 - j1 + G3
    z1 = z0 - k1 + G3

    x2 = x0 - i2 + 2.0 * G3
    y2 = y0 - j2 + 2.0 * G3
    z2 = z0 - k2 + 2.0 * G3

    x3 = x0 - 1.0 + 3.0 * G3
    y3 = y0 - 1.0 + 3.0 * G3
    z3 = z0 - 1.0 + 3.0 * G3

    # Calculate contribution from each corner
    def surflet(x, y, z, ix, jy, kz):
        t = 0.6 - x * x - y * y - z * z
        t = torch.clamp(t, min=0.0)

        # Pseudo-random gradient based on integer coordinates
        h = ((ix * 1619 + jy * 31337 + kz * 6971) & 0xFF) / 255.0
        grad_x = torch.cos(h * 2 * math.pi)
        grad_y = torch.sin(h * 2 * math.pi)
        grad_z = torch.cos(h * 3 * math.pi)

        return t * t * t * t * (grad_x * x + grad_y * y + grad_z * z)

    n0 = surflet(x0, y0, z0, i, j, k)
    n1 = surflet(x1, y1, z1, i + i1, j + j1, k + k1)
    n2 = surflet(x2, y2, z2, i + i2, j + j2, k + k2)
    n3 = surflet(x3, y3, z3, i + 1, j + 1, k + 1)

    # Sum contributions and scale to [-1, 1]
    return 32.0 * (n0 + n1 + n2 + n3)


def generate_noise_field(
    shape: tuple,
    period: float = 1.0,
    amplitude: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Generate a 2D field of Simplex 3D noise.

    Args:
        shape: (height, width) of the output field
        period: Scale of the noise pattern
        amplitude: Intensity of the noise
        offset_x, offset_y, offset_z: Position in noise space
        device: Torch device

    Returns:
        Tensor of shape (height, width) with noise values in range [-amplitude, amplitude]
    """
    H, W = shape

    # Create coordinate grid
    y_coords = torch.linspace(0, 1, H, device=device)
    x_coords = torch.linspace(0, 1, W, device=device)

    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Apply period and offset
    xx = (xx / period) + offset_x
    yy = (yy / period) + offset_y
    zz = torch.full_like(xx, offset_z)

    # Stack coordinates
    coords = torch.stack([xx, yy, zz], dim=-1)

    # Generate noise
    noise = simplex_noise_3d(coords)

    # Scale by amplitude and normalize to [0, 1]
    noise = (noise * amplitude + 1.0) / 2.0

    return noise.clamp(0, 1)
