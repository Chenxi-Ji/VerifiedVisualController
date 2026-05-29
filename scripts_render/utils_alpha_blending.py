import torch

@torch.no_grad()
def compute_bound_exp(x_lb, x_ub, eps= 1e-12):
    exp_lb = torch.exp(-0.5 * x_lb) # (num_valid, tile_h, tile_w)
    exp_ub = torch.exp(-0.5 * x_ub)

    dx = (x_ub - x_lb).clamp_min(eps)
    k = (exp_ub - exp_lb) / dx

    b_ub = exp_lb - k * x_lb

    # b_lb = 2k(log(-2k) - 1)
    neg_2k = (-2.0 * k).clamp_min(eps)
    b_lb = 2.0 * k * (torch.log(neg_2k) - 1.0)

    k_lb = k
    k_ub = k

    # handle degenerate interval x_lb == x_ub
    mask_equal = (x_ub - x_lb).abs() < eps
    if mask_equal.any():
        x0 = x_lb[mask_equal]
        y0 = exp_lb[mask_equal]

        # tangent at x0
        k0 = -0.5 * y0
        b0 = y0 - k0 * x0

        k_lb = k_lb.clone()
        k_ub = k_ub.clone()
        b_lb = b_lb.clone()
        b_ub = b_ub.clone()

        k_lb[mask_equal] = k0
        k_ub[mask_equal] = k0
        b_lb[mask_equal] = b0
        b_ub[mask_equal] = b0

    return k_lb, k_ub, b_lb, b_ub

# previous alpha blending function (for reference)
# @torch.no_grad()
# def compute_alpha_blending(w, colors, threshold=1e-3):
#     N, H, W = w.shape
#     device, dtype = w.device, w.dtype

#     T = torch.ones((H, W), device=device, dtype=dtype)
#     rgb = torch.zeros((H, W, 3), device=device, dtype=dtype)

#     for i in range(w.shape[0]):
#         a = w[i]  # (tile_h, tile_w)
#         c = colors[i]  # (3,)

#         contrib = (a.unsqueeze(-1) * T.unsqueeze(-1)) * c  # (tile_h, tile_w, 3)
#         rgb = rgb + contrib
#         T = T * (1 - a)

#         if (T < threshold).all():
#             # print(f"early stop")
#             break

#     return rgb

@torch.no_grad()
def compute_alpha_blending(w, colors):
    N, H, W = w.shape
    device, dtype = w.device, w.dtype

    rgb = torch.zeros((H, W, 3), device=device, dtype=dtype)

    for i in range(w.shape[0]-1, -1, -1):
        c = colors[i].view(1, 1, 3) # (1, 1, 3)
        d = c - rgb # (H, W, 3)
        rgb = rgb + d * w[i][..., None]

    # print(f"rgb.shape={rgb.shape}, rgb.min={rgb.min():.6f}, rgb.max={rgb.max():.6f}, rgb.mean={rgb.mean():.6f}")

    return rgb

@torch.no_grad()
def compute_interval_bound_alpha_blending(w_lb, w_ub, colors):
    N, H, W = w_lb.shape
    device, dtype = w_lb.device, w_lb.dtype

    rgb_lb = torch.zeros((H, W, 3), device=device, dtype=dtype)
    rgb_ub = torch.zeros((H, W, 3), device=device, dtype=dtype)

    for i in range(w_lb.shape[0]-1, -1, -1):
        c = colors[i].view(1, 1, 3) # (1, 1, 3)

        d_lb = c - rgb_lb # (H, W, 3)
        d_ub = c - rgb_ub

        w_l = w_lb[i][..., None] # (H, W, 1)
        w_u = w_ub[i][..., None]

        m_lb = (d_lb >= 0) # (H, W, 3) -> (H, W, 1) via broadcasting
        w_sel_lb = torch.where(m_lb, w_l, w_u) # (H, W, 1)
        rgb_lb = rgb_lb + d_lb * w_sel_lb # (H, W, 3)

        m_ub = (d_ub >= 0)
        w_sel_ub = torch.where(m_ub, w_u, w_l)
        rgb_ub = rgb_ub + d_ub * w_sel_ub

    return rgb_lb, rgb_ub


@torch.no_grad()
def compute_linear_bound_alpha_blending(
    w_lb, w_ub,
    w_k_lb, w_k_ub,
    w_b_lb, w_b_ub,
    colors
):
    N, H, W = w_lb.shape
    device, dtype = w_lb.device, w_lb.dtype

    rgb_lb = torch.zeros((H, W, 3), device=device, dtype=dtype)
    rgb_ub = torch.zeros((H, W, 3), device=device, dtype=dtype)

    rgb_k_lb = torch.zeros((N, H, W, 3), device=device, dtype=dtype)
    rgb_k_ub = torch.zeros((N, H, W, 3), device=device, dtype=dtype)

    rgb_b_lb = torch.zeros((H, W, 3), device=device, dtype=dtype)
    rgb_b_ub = torch.zeros((H, W, 3), device=device, dtype=dtype)

    for i in range(N - 1, -1, -1):

        c = colors[i].view(1, 1, 3)

        # -------------------------
        # residuals
        # -------------------------
        d_lb = c - rgb_ub
        d_ub = c - rgb_lb

        w_l = w_lb[i][..., None]
        w_u = w_ub[i][..., None]

        k_l = w_k_lb[i][..., None]
        k_u = w_k_ub[i][..., None]

        b_l = w_b_lb[i][..., None]
        b_u = w_b_ub[i][..., None]

        # =====================================================
        # LOWER (single consistent mask)
        # =====================================================
        m_lb = (d_lb >= 0)

        w_sel_lb = torch.where(m_lb, w_l, w_u)
        rgb_lb = rgb_lb + d_lb * w_sel_lb

        k_sel_lb = torch.where(m_lb, k_l, k_u)
        b_sel_lb = torch.where(m_lb, b_l, b_u)

        rgb_k_lb[i] = d_lb * k_sel_lb
        rgb_b_lb = rgb_b_lb + d_lb * b_sel_lb

        # =====================================================
        # UPPER (single consistent mask)
        # =====================================================
        m_ub = (d_ub >= 0)

        w_sel_ub = torch.where(m_ub, w_u, w_l)
        rgb_ub = rgb_ub + d_ub * w_sel_ub

        k_sel_ub = torch.where(m_ub, k_u, k_l)
        b_sel_ub = torch.where(m_ub, b_u, b_l)

        rgb_k_ub[i] = d_ub * k_sel_ub
        rgb_b_ub = rgb_b_ub + d_ub * b_sel_ub

    return rgb_lb, rgb_ub, rgb_k_lb, rgb_k_ub, rgb_b_lb, rgb_b_ub



# -----------------------------
# MC test
# -----------------------------

@torch.no_grad()
def test_compute_bound_exp(
    N=2000,
    H=32,
    W=32,
    num_samples=100,
    device="cuda",
    seed=0
):
    """
    Monte Carlo soundness validation for compute_bound_exp().
    """

    torch.manual_seed(seed)

    print("=" * 80)
    print("Generating random intervals...")

    # x_lb >= 0
    x_lb = torch.rand(
        N, H, W,
        device=device
    ) * 20.0

    # positive interval width
    width = torch.rand(
        N, H, W,
        device=device
    ) * 5.0

    x_ub = x_lb + width

    assert (x_lb >= 0).all()
    assert (x_lb <= x_ub).all()

    print(f"x_lb range: [{x_lb.min():.4f}, {x_lb.max():.4f}]")
    print(f"x_ub range: [{x_ub.min():.4f}, {x_ub.max():.4f}]")


    print("=" * 80)
    print("Computing certified bounds...")

    k_lb, k_ub, b_lb, b_ub = compute_bound_exp(x_lb, x_ub)

    print("=" * 80)
    print("Running Monte Carlo validation...")

    worst_lb_violation = -float("inf")
    worst_ub_violation = -float("inf")

    max_lb_error = 0.0
    max_ub_error = 0.0

    for i in range(num_samples):

        # sample x uniformly inside interval
        alpha = torch.rand_like(x_lb)

        x = x_lb + alpha * (x_ub - x_lb)

        y_true = torch.exp(-0.5 * x)

        y_lb = k_lb * x + b_lb
        y_ub = k_ub * x + b_ub

        # should be >= 0 if violated
        lb_violation = (y_lb - y_true).max().item()
        ub_violation = (y_true - y_ub).max().item()

        worst_lb_violation = max(worst_lb_violation, lb_violation)
        worst_ub_violation = max(worst_ub_violation, ub_violation)

        max_lb_error = max(
            max_lb_error,
            (y_true - y_lb).max().item()
        )

        max_ub_error = max(
            max_ub_error,
            (y_ub - y_true).max().item()
        )

        if (i + 1) % 10 == 0:
            print(
                f"[{i+1:03d}/{num_samples}] "
                f"worst_lb_violation={worst_lb_violation:.3e}, "
                f"worst_ub_violation={worst_ub_violation:.3e}"
            )

    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print(
        f"Worst lower-bound violation: "
        f"{worst_lb_violation:.6e}"
    )

    print(
        f"Worst upper-bound violation: "
        f"{worst_ub_violation:.6e}"
    )

    print(
        f"Max lower gap (true - lb): "
        f"{max_lb_error:.6e}"
    )

    print(
        f"Max upper gap (ub - true): "
        f"{max_ub_error:.6e}"
    )

    passed = (
        worst_lb_violation <= 1e-6
        and worst_ub_violation <= 1e-6
    )

    print("=" * 80)

    if passed:
        print("✅ SOUNDNESS TEST PASSED")
    else:
        print("❌ SOUNDNESS TEST FAILED")

    print("=" * 80)

    return passed


# def test():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     aMin = torch.tensor([0.7, 0.8, 0.6, 0.3]).to(device)
#     aMax = torch.tensor([0.8, 1.0, 1.0, 0.7]).to(device)

#     aMin = aMin.unsqueeze(-1).unsqueeze(-1)  # (N,) -> (N, 1, 1)
#     aMax = aMax.unsqueeze(-1).unsqueeze(-1)  # (N,) -> (N, 1, 1)

#     # N=4, ch=3
#     c = torch.tensor([
#         [0.6, 0.4, 0.5, 0.3],
#         [0.2, 0.3, 0.7, 0.1],
#         [0.9, 0.1, 0.4, 0.8],
#     ]).to(device)

#     CMax, CMin = compute_bound_alpha_blending(aMin, aMax, c)

#     print("CMax:", CMax)
#     print("CMin:", CMin)



if __name__ == "__main__":
    test_compute_bound_exp()