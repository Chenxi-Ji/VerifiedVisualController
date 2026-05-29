from time import time
import torch
torch.set_printoptions(precision=4, sci_mode=True)

@torch.no_grad()
def quad_kkt_bound(X_lb, X_ub, A, B, lam, find_max=True):
    """
    Compute strict KKT bound of:
        g(x) = x^T (A - lam * B) x

    over box:
        X_lb <= x <= X_ub

    Inputs
    ------
    X_lb : (N,3)
    X_ub : (N,3)
    A    : (N,H,W,3,3)
    B    : (N,3,3)
    lam  : (N,H,W)

    Returns
    -------
    out  : (N,H,W)
    """

    eps = 1e-12
    fill = -torch.inf if find_max else torch.inf
    reduce_fn = torch.maximum if find_max else torch.minimum

    dtype = A.dtype
    device = A.device

    # -------------------------------------------------------
    # unpack symmetric matrix
    # -------------------------------------------------------
    a = A[..., 0, 0] - lam * B[..., 0:1, 0:1] # (N,H,W)
    b = A[..., 0, 1] - lam * B[..., 0:1, 1:2] # (N,H,W)
    c = A[..., 0, 2] - lam * B[..., 0:1, 2:3] # (N,H,W)

    d = A[..., 1, 1] - lam * B[..., 1:2, 1:2] # (N,H,W)
    e = A[..., 1, 2] - lam * B[..., 1:2, 2:3] # (N,H,W)

    f = A[..., 2, 2] - lam * B[..., 2:3, 2:3] # (N,H,W)

    # -------------------------------------------------------
    # box bounds
    # shape: (N,1,1)
    # -------------------------------------------------------
    xl = X_lb[:, 0][:, None, None] # (N,1,1)
    yl = X_lb[:, 1][:, None, None]
    zl = X_lb[:, 2][:, None, None]

    xu = X_ub[:, 0][:, None, None]
    yu = X_ub[:, 1][:, None, None]
    zu = X_ub[:, 2][:, None, None]

    out = torch.full_like(a, fill) # (N,H,W)

    # -------------------------------------------------------
    # quadratic
    # -------------------------------------------------------
    def quad(x, y, z):
        return (
            a * x * x
            + d * y * y
            + f * z * z
            + 2 * b * x * y
            + 2 * c * x * z
            + 2 * e * y * z
        )

    def update(out, g, valid):
        return reduce_fn(
            out,
            torch.where(valid, g, fill)
        )

    valid_all = torch.ones_like(a, dtype=torch.bool)

    # =======================================================
    # 1. CORNERS (8)
    # =======================================================
    for x in (xl, xu):
        for y in (yl, yu):
            for z in (zl, zu):
                g = quad(x, y, z)
                out = update(out, g, valid_all)

    # =======================================================
    # 2. EDGES (12)
    # =======================================================

    # -------------------------------------------------------
    # edge: fix x,y → solve z
    # -------------------------------------------------------
    for x in (xl, xu):
        for y in (yl, yu):

            valid = f.abs() > eps

            z = -(c * x + e * y) / (f + eps)

            inside = (
                (z >= zl)
                & (z <= zu)
                & valid
            )

            g = quad(x, y, z)
            out = update(out, g, inside)

    # -------------------------------------------------------
    # edge: fix x,z → solve y
    # -------------------------------------------------------
    for x in (xl, xu):
        for z in (zl, zu):

            valid = d.abs() > eps

            y = -(b * x + e * z) / (d + eps)

            inside = (
                (y >= yl)
                & (y <= yu)
                & valid
            )

            g = quad(x, y, z)
            out = update(out, g, inside)

    # -------------------------------------------------------
    # edge: fix y,z → solve x
    # -------------------------------------------------------
    for y in (yl, yu):
        for z in (zl, zu):

            valid = a.abs() > eps

            x = -(b * y + c * z) / (a + eps)

            inside = (
                (x >= xl)
                & (x <= xu)
                & valid
            )

            g = quad(x, y, z)
            out = update(out, g, inside)

    # =======================================================
    # 3. FACES (6)
    # =======================================================

    # -------------------------------------------------------
    # face x fixed → optimize (y,z)
    # -------------------------------------------------------
    for x in (xl, xu):

        det = d * f - e * e
        valid = det.abs() > eps

        y = x * (e * c - f * b) / (det + eps)
        z = x * (e * b - d * c) / (det + eps)

        inside = (
            valid
            & (y >= yl)
            & (y <= yu)
            & (z >= zl)
            & (z <= zu)
        )

        g = quad(x, y, z)
        out = update(out, g, inside)

    # -------------------------------------------------------
    # face y fixed → optimize (x,z)
    # -------------------------------------------------------
    for y in (yl, yu):

        det = a * f - c * c
        valid = det.abs() > eps

        x = y * (c * e - f * b) / (det + eps)
        z = y * (c * b - a * e) / (det + eps)

        inside = (
            valid
            & (x >= xl)
            & (x <= xu)
            & (z >= zl)
            & (z <= zu)
        )

        g = quad(x, y, z)
        out = update(out, g, inside)

    # -------------------------------------------------------
    # face z fixed → optimize (x,y)
    # -------------------------------------------------------
    for z in (zl, zu):

        det = a * d - b * b
        valid = det.abs() > eps

        x = z * (b * e - d * c) / (det + eps)
        y = z * (b * c - a * e) / (det + eps)

        inside = (
            valid
            & (x >= xl)
            & (x <= xu)
            & (y >= yl)
            & (y <= yu)
        )

        g = quad(x, y, z)
        out = update(out, g, inside)

    # =======================================================
    # 4. INTERIOR
    # =======================================================
    inside_zero = (
        (xl <= 0)
        & (xu >= 0)
        & (yl <= 0)
        & (yu >= 0)
        & (zl <= 0)
        & (zu >= 0)
    )

    g0 = torch.zeros_like(out)

    out = update(out, g0, inside_zero)

    return out

# ============================================================
# RATIONAL QUAD BOUND (FAST VERSION)
# ============================================================
@torch.no_grad()
def compute_rational_eig_max(semi_A, semi_B):
    A_eig_max = (semi_A**2).sum(dim=(-1,-2)) # (N,H,W)
    B_det = torch.linalg.det(semi_B)**2 # (N,)
    B_fro2 = (semi_B @ semi_B.transpose(-1, -2)).square().sum(dim=(-1, -2))
    B_eig_min = B_det / B_fro2 # (N,)

    rational_eig_max = A_eig_max / B_eig_min[:, None, None] # (N,H,W)
    return rational_eig_max
    

def rational_quad_bound(
    X_lb,
    X_ub,
    semi_A,
    semi_B,
    num_bisect=60,
    max_cap=1e6,
    min_cap=1e-6,
    tol=1e-6,
    debug=False,
):
    A = semi_A @ semi_A.transpose(-1, -2) # (N,H,W,3,3)
    B = semi_B @ semi_B.transpose(-1, -2) # (N,3,3)
    # ---------------------------
    # extract X bounds
    # ---------------------------
    global_lb = torch.full_like(A[..., 0, 0], min_cap)

    global_ub = compute_rational_eig_max(semi_A, semi_B)
    global_ub.clamp_(max=max_cap)

    lo_u = global_lb.clone()
    hi_u = global_ub.clone()

    lo_l = global_lb.clone()
    hi_l = global_ub.clone()

    for _ in range(num_bisect):
        active_u = (hi_u - lo_u) > tol # True if upper bound is not tight
        active_l = (hi_l - lo_l) > tol # True if lower bound is not tight

        lam_u = 0.5 * (lo_u + hi_u)
        lam_l = 0.5 * (lo_l + hi_l)

        active_u_n = active_u.any(dim=(1, 2))
        active_l_n = active_l.any(dim=(1, 2))

        idx_u = active_u_n.nonzero(as_tuple=True)[0]
        idx_l = active_l_n.nonzero(as_tuple=True)[0]

        # ---------------------------
        # upper bound branch
        # ---------------------------
        if idx_u.numel() > 0:
            lam = lam_u[idx_u] # (N_active_u, H, W)

            g_max = quad_kkt_bound(
                X_lb[idx_u],
                X_ub[idx_u],
                A[idx_u],
                B[idx_u],
                lam,
                find_max=True,
            )

            feasible = (g_max >= 0) & active_u[idx_u]

            lo_u[idx_u] = torch.where(feasible, lam_u[idx_u], lo_u[idx_u])
            hi_u[idx_u] = torch.where(feasible, hi_u[idx_u], lam_u[idx_u])

        # ---------------------------
        # lower bound branch
        # ---------------------------
        if idx_l.numel() > 0:
            lam = lam_l[idx_l] # (N_active_l, H, W)
            
            g_min = quad_kkt_bound(
                X_lb[idx_l],
                X_ub[idx_l],
                A[idx_l],
                B[idx_l],
                lam,
                find_max=False,
            )
                

            feasible = (g_min <= 0) & active_l[idx_l]
            hi_l[idx_l] = torch.where(feasible, lam_l[idx_l], hi_l[idx_l])
            lo_l[idx_l] = torch.where(feasible, lo_l[idx_l], lam_l[idx_l])

    assert (lo_l <= hi_l).all()
    assert (lo_u <= hi_u).all()

    gap = max(tol, max_cap / (2 ** num_bisect))
    lower = torch.clamp(lo_l-gap, min=0.0)
    upper = torch.clamp(hi_u+gap, max=max_cap)

    if debug:
        viol = lower > upper
        num = viol.sum().item()
        diff = (lower - upper)
        max_val, flat_idx = diff.flatten().max(dim=0)
        idx = torch.unravel_index(flat_idx, lower.shape)
        print(f"lo_l[idx], hi_l[idx], lo_u[idx], hi_u[idx] = {lo_l[idx].item():.6f}, {hi_l[idx].item():.6f}, {lo_u[idx].item():.6f}, {hi_u[idx].item():.6f}")
        assert not viol.any(), (
            f"violation num = {num}, "
            f"max diff = {max_val.item()}, "
            f"max idx = {idx}, "
            f"lower = {lower[idx].item()}, "
            f"upper = {upper[idx].item()}"
        )

    else: 
        assert (lower <= upper).all(), (
            f"violation num = {(lower > upper).sum().item()}, "
            f"max diff = {(lower - upper).max().item()}"
        )

    return lower, upper
    

@torch.no_grad()
def monte_carlo_test(
    X_lb,
    X_ub,
    semi_A,
    semi_B,
    min_cap=1e-6,
    max_cap=1e+6,
    num_samples=4096,
    chunk_size=256,
):
    N, H, W = semi_A.shape[:3]
    device = semi_A.device

    mc_lb = torch.full(
        (N, H, W),
        float('inf'),
        device=device,
        dtype=semi_A.dtype,
    )  # (N,H,W)

    mc_ub = torch.full(
        (N, H, W),
        -float('inf'),
        device=device,
        dtype=semi_A.dtype,
    )  # (N,H,W)

    for start in range(0, num_samples, chunk_size):

        S = min(chunk_size, num_samples - start)
        r = torch.rand(N, S, 3, device=device)

        X_lb_ = X_lb[:, None, :]
        X_range = (X_ub - X_lb)[:, None, :] # (N,1,3)
        X = X_lb_ + X_range * r # (N,S,3)

        semi_numer = torch.einsum('nsi,nhwij->nhwsj', X, semi_A)  # (N,H,W,S,3)
        numer = (semi_numer * semi_numer).sum(dim=-1)  # (N,H,W,S)
        
        semi_denom = X[:, :, None, :] @ semi_B[:, None, :,: ]  # (N,S,1,3)
        semi_denom = torch.einsum('nsi,nij->nsj', X, semi_B)  # (N,S,3)
        denom = (semi_denom * semi_denom).sum(dim=-1)  # (N,S)

        val = numer / denom[:, None, None, :]  # (N,H,W,S)

        mc_lb = torch.minimum(mc_lb, val.min(dim=-1).values)
        mc_ub = torch.maximum(mc_ub, val.max(dim=-1).values)

    mc_lb = torch.clamp(mc_lb, min=min_cap)
    mc_ub = torch.clamp(mc_ub, max=max_cap) 

    return mc_lb, mc_ub


def make_semi_pd_matrix(*shape, scale=1.0, device='cpu', dtype=torch.float32):
    G = torch.randn(*shape, 3, 3, device=device, dtype=dtype)
    G += scale * torch.eye(3, device=device, dtype=dtype)
    return G
    
if __name__ == '__main__':
    torch.manual_seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64

    N = 2000
    H = 32
    W = 32
    num_bisect=60
    min_cap = 1e-4
    max_cap = 1e+2
    tol = 1e-4
    debug = True

    semi_A = make_semi_pd_matrix(N, H, W, scale=0.5, device=device, dtype=dtype)  # (N,H,W,3,3)
    semi_B = make_semi_pd_matrix(N, scale=0.5, device=device, dtype=dtype)  # (N,3,3)

    center_x = 4 * torch.randn(N, device=device, dtype=dtype)  # (N,)
    center_y = 4 * torch.randn(N, device=device, dtype=dtype)  # (N,)
    center_z = 4 * torch.randn(N, device=device, dtype=dtype)  # (N,)

    radius_x = 0.1 + 2.0 * torch.rand(N, device=device, dtype=dtype)  # (N,)
    radius_y = 0.1 + 2.0 * torch.rand(N, device=device, dtype=dtype)  # (N,)
    radius_z = 0.1 + 2.0 * torch.rand(N, device=device, dtype=dtype)  # (N,)

    X_lb = torch.stack([
        center_x - radius_x,
        center_y - radius_y,
        center_z - radius_z,
    ], dim=-1)  # (N,3)

    X_ub = torch.stack([
        center_x + radius_x,
        center_y + radius_y,
        center_z + radius_z,
    ], dim=-1)  # (N,3)

    start_time = time()
    lower, upper = rational_quad_bound(
        X_lb,
        X_ub,
        semi_A,
        semi_B,
        min_cap=min_cap,
        max_cap=max_cap,
        num_bisect=num_bisect,
        tol=tol,
        debug=debug,
    )  # (N,H,W)
    end_time = time()
    print(f'Time for Bound Computation: {end_time - start_time:.4f} seconds')


    start_time = time()
    mc_lb, mc_ub = monte_carlo_test(
        X_lb,
        X_ub,
        semi_A,
        semi_B,
        num_samples=4096,
        chunk_size=64,
        min_cap=min_cap,
        max_cap=max_cap,
    )  # (N,H,W)
    end_time = time()
    print(f'Time for Monte Carlo Test: {end_time - start_time:.4f} seconds')

    lower_violation = torch.clamp(lower - mc_lb, min=0)  # (N,H,W)
    upper_violation = torch.clamp(mc_ub - upper, min=0)  # (N,H,W)
    violation_mask = ( (lower_violation > 0) | (upper_violation > 0) )  # (N,H,W)
    total_violation = violation_mask.sum().item()

    print('=' * 80)
    print('BOUND SOUNDNESS CHECK')
    print('=' * 80)

    print(f'N={N}, H={H}, W={W}')
    print(f'Violation count: {total_violation}')

    print(
        'Max lower violation:',
        lower_violation.max().item(),
    )

    print(
        'Max upper violation:',
        upper_violation.max().item(),
    )

    print(
        'Worst lower margin:',
        (mc_lb - lower).min().item(),
    )

    print(
        'Worst upper margin:',
        (upper - mc_ub).min().item(),
    )

    
    if debug: #and total_violation > 0:
        for n in range(2):
            for h in range(2):
                for w in range(2):

                    print('-' * 80)
                    print(f'case=({n},{h},{w})')

                    print(
                        f'x=[{X_lb[n, 0].item():.6f}, '
                        f'{X_ub[n, 0].item():.6f}]'
                    )

                    print(
                        f'y=[{X_lb[n, 1].item():.6f}, '
                        f'{X_ub[n, 1].item():.6f}]'
                    )

                    print(
                        f'z=[{X_lb[n, 2].item():.6f}, '
                        f'{X_ub[n, 2].item():.6f}]'
                    )

                    print(
                        'computed:',
                        lower[n, h, w].item(),
                        upper[n, h, w].item(),
                    )

                    print(
                        'mc:',
                        mc_lb[n, h, w].item(),
                        mc_ub[n, h, w].item(),
                    )
