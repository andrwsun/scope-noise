"""Perlin-style 3D noise implementation for PyTorch."""

import torch


def fade(t: torch.Tensor) -> torch.Tensor:
    """Fade function for smooth interpolation."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear interpolation."""
    return a + t * (b - a)


def grad(hash: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Convert hash value to gradient direction and compute dot product.
    """
    # Use hash to select gradient direction
    h = hash & 15

    # Gradient directions (12 edges of a cube + 4 diagonals)
    u = torch.where(h < 8, x, y)
    v = torch.where(h < 4, y, torch.where((h == 12) | (h == 14), x, z))

    return torch.where(h & 1 == 0, u, -u) + torch.where(h & 2 == 0, v, -v)


def perlin_noise_3d(coords: torch.Tensor) -> torch.Tensor:
    """
    Generate 3D Perlin noise.

    Args:
        coords: Tensor of shape (..., 3) with XYZ coordinates

    Returns:
        Tensor of shape (...) with noise values in range [-1, 1]
    """
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

    # Create permutation table (simple hash)
    # Using a reproducible but pseudo-random permutation
    def hash_coord(xi, yi, zi):
        return ((xi * 1619 + yi * 31337 + zi * 6971) & 0xFFFFFF) % 256

    # Hash coordinates of 8 cube corners
    aaa = hash_coord(X, Y, Z)
    aba = hash_coord(X, Y + 1, Z)
    aab = hash_coord(X, Y, Z + 1)
    abb = hash_coord(X, Y + 1, Z + 1)
    baa = hash_coord(X + 1, Y, Z)
    bba = hash_coord(X + 1, Y + 1, Z)
    bab = hash_coord(X + 1, Y, Z + 1)
    bbb = hash_coord(X + 1, Y + 1, Z + 1)

    # Compute gradients at cube corners and interpolate
    x1 = lerp(
        grad(aaa, x, y, z),
        grad(baa, x - 1, y, z),
        u
    )
    x2 = lerp(
        grad(aba, x, y - 1, z),
        grad(bba, x - 1, y - 1, z),
        u
    )
    y1 = lerp(x1, x2, v)

    x1 = lerp(
        grad(aab, x, y, z - 1),
        grad(bab, x - 1, y, z - 1),
        u
    )
    x2 = lerp(
        grad(abb, x, y - 1, z - 1),
        grad(bbb, x - 1, y - 1, z - 1),
        u
    )
    y2 = lerp(x1, x2, v)

    return lerp(y1, y2, w)


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
    Generate a 2D field of Perlin 3D noise.

    Args:
        shape: (height, width) of the output field
        period: Scale of the noise pattern
        amplitude: Intensity of the noise
        offset_x, offset_y, offset_z: Position in noise space
        device: Torch device

    Returns:
        Tensor of shape (height, width) with noise values in range [0, 1]
    """
    H, W = shape

    # Create coordinate grid
    y_coords = torch.linspace(0, 1, H, device=device)
    x_coords = torch.linspace(0, 1, W, device=device)

    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Apply period and offset
    # Scale coordinates to create larger patterns
    xx = (xx * 10.0 / period) + offset_x
    yy = (yy * 10.0 / period) + offset_y
    zz = torch.full_like(xx, offset_z)

    # Stack coordinates
    coords = torch.stack([xx, yy, zz], dim=-1)

    # Generate noise
    noise = perlin_noise_3d(coords)

    # Scale by amplitude and normalize to [0, 1]
    noise = (noise * amplitude + 1.0) / 2.0

    return noise.clamp(0, 1)
