import torch



def spherical_add(x, y):
    r1, theta1, phi1 = x
    r2, theta2, phi2 = y

    x = r1 * torch.sin(theta1) * torch.cos(phi1) + r2 * torch.sin(theta2) * torch.cos(phi2)
    y = r1 * torch.sin(theta1) * torch.sin(phi1) + r2 * torch.sin(theta2) * torch.sin(phi2)
    z = r1 * torch.cos(theta1) + r2 * torch.cos(theta2)

    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)
    phi = torch.atan2(y, x)

    return r, theta, phi


def spherical_noise(x, scale=None):
    noise = torch.randn_like(x)
    scales = torch.rand(noise.shape[0]) * scale if scale is not None else torch.rand(noise.shape[0])
    noise = noise - (noise * x).sum(-1, keepdim=True) * x
    # x = spherical_add(x, scale * noise)
    while len(scales.shape) < len(noise.shape):
        scales = scales.unsqueeze(-1)
    x = x + scales * noise
    return x
