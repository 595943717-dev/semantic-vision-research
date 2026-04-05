import torch

def vanishing_point(X, vp):
    lines = X[..., 6:9]
    centroids = X[..., 9:12]

    constrained_lines = torch.cross(centroids, vp.unsqueeze(-2), dim=-1)
    con_line_norms = torch.norm(constrained_lines[..., 0:2], dim=-1, keepdim=True)
    constrained_lines = constrained_lines / (con_line_norms + 1e-8)
    line_norms = torch.norm(lines[..., 0:2], dim=-1, keepdim=True)
    lines = lines / (line_norms + 1e-8)
    distances = 1 - torch.abs((lines[..., 0:2] * constrained_lines[..., 0:2]).sum(dim=-1))

    return distances

def transfer_error(X, h):
    h_dims = h.size()
    batch_dims = list(h_dims)[:-1]
    H_dims = tuple(list(h_dims)[:-1] + [3, 3])

    N = X.size(-2)

    H = h.view(H_dims)
    I = torch.eye(3, device=X.device).view([1]*(len(H_dims)-2) + [3, 3])
    det = torch.det(H).view(batch_dims + [1, 1])
    H_ = torch.where(det < 1e-6, I, H)

    X1 = torch.ones(batch_dims + [N, 3, 1], device=X.device)
    X2 = torch.ones(batch_dims + [N, 3, 1], device=X.device)
    X1[..., 0:2, 0] = X[..., 0:2]
    X2[..., 0:2, 0] = X[..., 2:4]

    HX1 = H.unsqueeze(-3) @ X1
    HX1[..., 0, 0] /= torch.clamp(HX1[..., 2, 0], min=1e-8)
    HX1[..., 1, 0] /= torch.clamp(HX1[..., 2, 0], min=1e-8)
    HX1[..., 2, 0] /= torch.clamp(HX1[..., 2, 0], min=1e-8)

    HX2 = torch.linalg.solve(H_.unsqueeze(-3), X2)
    HX2[..., 0, 0] /= torch.clamp(HX2[..., 2, 0], min=1e-8)
    HX2[..., 1, 0] /= torch.clamp(HX2[..., 2, 0], min=1e-8)
    HX2[..., 2, 0] /= torch.clamp(HX2[..., 2, 0], min=1e-8)

    signed_distances_1 = HX1 - X2
    distances_1 = (signed_distances_1 * signed_distances_1).sum(dim=-2).squeeze(-1)
    signed_distances_2 = HX2 - X1
    distances_2 = (signed_distances_2 * signed_distances_2).sum(dim=-2).squeeze(-1)

    distances = distances_1 + distances_2

    return distances


def sampson_distance(X, f, squared=False):
    f_dims = f.size() 
    batch_dims = list(f_dims)[:-1] 
    F_dims = tuple(list(f_dims)[:-1] + [3, 3]) 

    degenerate = torch.norm(f, dim=-1) < 1e-8 

    N = X.size(-2) 

    F = f.view(F_dims)[..., None, :, :]

    x1 = X[..., 0:2] 
    x2 = X[..., 2:4]
    X1 = torch.ones(batch_dims + [N, 3, 1], device=X.device)
    X2 = torch.ones(batch_dims + [N, 3, 1], device=X.device)
    X1[..., 0:2, 0] = x1
    X2[..., 0:2, 0] = x2

    Fx1 = F @ X1
    Fx2 = F.transpose(-1, -2) @ X2

    xFx = X2 * Fx1
    xFx = torch.sum(xFx[..., 0], dim=-1) ** 2

    denom = Fx1[..., 0, 0] ** 2 + Fx1[..., 1, 0] ** 2 + Fx2[..., 0, 0] ** 2 + Fx2[..., 1, 0] ** 2
    denom = torch.clamp(denom, min=1e-8) 

    sq_distances = xFx / denom

    sq_distances = sq_distances + degenerate[..., None].float() * 1e6

    if squared:
        return sq_distances
    else:
        return torch.sqrt(sq_distances)


mapping = {"vp": vanishing_point, "homography": transfer_error, "fundamental": sampson_distance}


def quantize_tensor(tensor: torch.Tensor, bits: int, scale: float) -> torch.Tensor:
    """Quantizes a tensor using a symmetric quantization scheme.

    Args:
        tensor: The input tensor
        bits: The number of bits for quantization (e.g., 8 for int8)
        scale: The scale parameter for quantization

    Returns:
        A quantized tensor
    """
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    q_tensor = torch.round(tensor / scale).clamp(qmin, qmax) * scale
    return q_tensor

def vanishing_point_Q(X: torch.Tensor, vp: torch.Tensor, quantize: bool=False, bits: int=8, scale: float=0.1) -> torch.Tensor:
    """
    Computes residuals for vanishing point fitting.

    Args:
        X: input tensor [..., num_points, 12]
        vp: vanishing point [..., 3]
        quantize: if to quantize
        bits: number of bits used for quantization
        scale: quantization scale
    """

    lines = X[..., 6:9]
    centroids = X[..., 9:12]

    constrained_lines = torch.cross(centroids, vp.unsqueeze(-2), dim=-1)
    con_line_norms = torch.norm(constrained_lines[..., 0:2], dim=-1, keepdim=True)
    constrained_lines = constrained_lines / (con_line_norms + 1e-8)
    line_norms = torch.norm(lines[..., 0:2], dim=-1, keepdim=True)
    lines = lines / (line_norms + 1e-8)
    distances = 1 - torch.abs((lines[..., 0:2] * constrained_lines[..., 0:2]).sum(dim=-1))

    if quantize:
      distances = quantize_tensor(distances, bits=bits, scale=scale)

    return distances


def transfer_error_Q(X: torch.Tensor, h: torch.Tensor, quantize: bool=False, bits: int=8, scale: float=0.1) -> torch.Tensor:
    """
      Computes transfer error for homography fitting.

      Args:
          X: Input tensor [..., num_points, 5]
          h: Homography parameters
          quantize: if to quantize
          bits: number of bits used for quantization
          scale: quantization scale
      Returns:
        Tensor: transfer error between the transformed points
    """
    h_dims = h.size()
    batch_dims = list(h_dims)[:-1]
    H_dims = tuple(list(h_dims)[:-1] + [3, 3])

    N = X.size(-2)

    H = h.view(H_dims)
    I = torch.eye(3, device=X.device).view([1]*(len(H_dims)-2) + [3, 3])
    det = torch.det(H).view(batch_dims + [1, 1])
    H_ = torch.where(det < 1e-6, I, H)

    X1 = torch.ones(batch_dims + [N, 3, 1], device=X.device)
    X2 = torch.ones(batch_dims + [N, 3, 1], device=X.device)
    X1[..., 0:2, 0] = X[..., 0:2]
    X2[..., 0:2, 0] = X[..., 2:4]

    HX1 = H.unsqueeze(-3) @ X1
    HX1[..., 0, 0] /= torch.clamp(HX1[..., 2, 0], min=1e-8)
    HX1[..., 1, 0] /= torch.clamp(HX1[..., 2, 0], min=1e-8)
    HX1[..., 2, 0] /= torch.clamp(HX1[..., 2, 0], min=1e-8)

    HX2 = torch.linalg.solve(H_.unsqueeze(-3), X2)
    HX2[..., 0, 0] /= torch.clamp(HX2[..., 2, 0], min=1e-8)
    HX2[..., 1, 0] /= torch.clamp(HX2[..., 2, 0], min=1e-8)
    HX2[..., 2, 0] /= torch.clamp(HX2[..., 2, 0], min=1e-8)

    signed_distances_1 = HX1 - X2
    distances_1 = (signed_distances_1 * signed_distances_1).sum(dim=-2).squeeze(-1)
    signed_distances_2 = HX2 - X1
    distances_2 = (signed_distances_2 * signed_distances_2).sum(dim=-2).squeeze(-1)

    distances = distances_1 + distances_2

    if quantize:
      distances = quantize_tensor(distances, bits=bits, scale=scale)
    return distances


def sampson_distance_Q(X: torch.Tensor, f: torch.Tensor, squared=False, quantize: bool=False, bits: int=8, scale: float=0.1) -> torch.Tensor:
    """
      Computes the Sampson distance between a pair of corresponding points using the fundamental matrix.
      Args:
        X: Input tensor [..., num_points, 5]
        f: Fundamental matrix [..., 9]
        squared: If true, return squared distance, otherwise, return square root distance.
        quantize: if to quantize
        bits: number of bits used for quantization
        scale: quantization scale
    """
    f_dims = f.size()
    batch_dims = list(f_dims)[:-1]
    F_dims = tuple(list(f_dims)[:-1] + [3, 3])

    degenerate = torch.norm(f, dim=-1) < 1e-8

    N = X.size(-2)

    F = f.view(F_dims)[..., None, :, :]

    x1 = X[..., 0:2]
    x2 = X[..., 2:4]
    X1 = torch.ones(batch_dims + [N, 3, 1], device=X.device)
    X2 = torch.ones(batch_dims + [N, 3, 1], device=X.device)
    X1[..., 0:2, 0] = x1
    X2[..., 0:2, 0] = x2

    Fx1 = F @ X1
    Fx2 = F.transpose(-1, -2) @ X2

    xFx = X2 * Fx1
    xFx = torch.sum(xFx[..., 0], dim=-1) ** 2

    denom = Fx1[..., 0, 0] ** 2 + Fx1[..., 1, 0] ** 2 + Fx2[..., 0, 0] ** 2 + Fx2[..., 1, 0] ** 2
    denom = torch.clamp(denom, min=1e-8)

    sq_distances = xFx / denom

    sq_distances = sq_distances + degenerate[..., None].float() * 1e6


    if quantize:
        sq_distances = quantize_tensor(sq_distances, bits=bits, scale=scale)

    if squared:
        return sq_distances
    else:
        return torch.sqrt(sq_distances)

mapping = {"vp": vanishing_point_Q, "homography": transfer_error_Q, "fundamental": sampson_distance_Q}

def compute_residuals_Q(opt, X: torch.Tensor, hypotheses: torch.Tensor, quantize: bool=False, bits: int=8, scale: float=0.1) -> torch.Tensor:
    """
        Computes residuals for all the given hypotheses.
    Args:
        opt: Config options
        X: Input data [batch, N, dim]
        hypotheses: Hypotheses [batch, samplecount, hypothesis, instance, model]
        quantize: if to quantize the residual
        bits: Number of bits to quantize
        scale: scale used for quantization
    """

    B, N, C = X.size()
    _, K, S, M, D = hypotheses.size()

    X_e = X.view(B, 1, 1, 1, N, C).expand(B, K, S, M, N, C)

    residuals = mapping[opt.problem](X_e, hypotheses, quantize=quantize, bits=bits, scale=scale)


    return residuals

