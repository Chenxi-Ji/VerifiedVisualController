import torch

def build_matrices(semi_A, semi_B):
    A = semi_A @ semi_A.transpose(-1, -2)                 # (N,H,W,3,3)
    B = semi_B[:, None, None] @ semi_B[:, None, None].transpose(-1, -2)  # (N,H,W,3,3)
    return A, B


def q_func(x, A, B):
    x = x.view(1,1,1,3,1)                                 # (1,1,1,3,1)

    Ax = A @ x                                            # (N,H,W,3,1)
    Bx = B @ x                                            # (N,H,W,3,1)

    xAx = (x.transpose(-1,-2) @ Ax).squeeze(-1).squeeze(-1)  # (N,H,W)
    xBx = (x.transpose(-1,-2) @ Bx).squeeze(-1).squeeze(-1)  # (N,H,W)

    return -0.5 * xAx / (xBx + 1e-8)                      # (N,H,W)


def render(q, opacities, colors):
    w = opacities[:, None, None] * q                      # (N,H,W)

    T = torch.ones_like(w[0])                             # (H,W)
    out = torch.zeros((H:=w.shape[1], W:=w.shape[2], 3), device=w.device)

    for i in range(w.shape[0]):
        wi = w[i]                                          # (H,W)
        ci = colors[i]                                    # (3,)
        out = out + wi[..., None] * T[..., None] * ci     # (H,W,3)
        T = T * (1 - wi)                                  # (H,W)

    return out                                            # (H,W,3)


def taylor_bound(x0, X_lb, X_ub, f, samples=10):

    # center + anisotropic radius
    r = 0.5 * (X_ub - X_lb)                       # (3,)
    x0 = x0.clone().detach().requires_grad_(True) # (3,)

    f0 = f(x0)                                    # scalar
    g = torch.autograd.grad(f0, x0)[0]           # (3,)

    # ---- Hessian box bound (anisotropic) ----
    M = 0.0

    for _ in range(samples):

        x = X_lb + torch.rand_like(X_lb) * (X_ub - X_lb)   # (3,)
        x.requires_grad_(True)

        y = f(x)
        grad = torch.autograd.grad(y, x, create_graph=True)[0]  # (3,)

        Hv_sum = 0.0

        for i in range(3):

            ei = torch.zeros_like(x)
            ei[i] = 1.0                                     # (3,)

            gv = (grad * ei).sum()
            hv = torch.autograd.grad(gv, x, retain_graph=True)[0]  # (3,)

            Hv_sum = Hv_sum + torch.abs(hv[i]) * r[i] * r[i]

        M = max(M, Hv_sum.item())

    # ---- final affine model ----
    def L(x):
        dx = x - x0                                       # (3,)
        return f0 + (g * dx).sum() - 0.5 * M              # scalar

    def U(x):
        dx = x - x0                                       # (3,)
        return f0 + (g * dx).sum() + 0.5 * M              # scalar

    return L, U, g, f0, M

def test_taylor_tightness():

    torch.manual_seed(0)

    N, H, W = 20, 16, 16

    semi_A = torch.randn(N,H,W,3,3)
    semi_B = torch.randn(N,3,3).abs() + 0.5
    opacities = torch.rand(N)
    colors = torch.rand(N,3)
    cam_const = torch.zeros(N,3)

    X_lb = torch.tensor([-0.5,-0.5,-0.5])
    X_ub = torch.tensor([ 0.5, 0.5, 0.5])

    A, B = build_matrices(semi_A, semi_B)

    def f(x):
        q = q_func(x, A, B)
        return render(q, opacities, colors).sum()

    x0 = 0.5 * (X_lb + X_ub)

    L, U, g, f0, M = taylor_bound(x0, X_lb, X_ub, f, samples=20)

    # Monte Carlo ground truth check
    err = []

    for _ in range(200):
        x = X_lb + torch.rand(3) * (X_ub - X_lb)

        fx = f(x).item()
        lx = L(x).item()
        ux = U(x).item()

        err.append(fx < lx or fx > ux)

    print("violation rate:", sum(err)/len(err))
    print("M:", M)

def main():
    torch.manual_seed(0)

    # -----------------------------
    # toy config
    # -----------------------------
    N, H, W = 10, 8, 8

    semi_A = torch.randn(N, H, W, 3, 3)
    semi_B = torch.randn(N, 3, 3).abs() + 0.5
    opacities = torch.rand(N)
    colors = torch.rand(N, 3)
    cam_const = torch.zeros(N, 3)

    X_lb = torch.tensor([-0.5, -0.3, -0.2])
    X_ub = torch.tensor([ 0.6,  0.4,  0.5])

    # -----------------------------
    # build model
    # -----------------------------
    A, B = build_matrices(semi_A, semi_B)

    def f(x):
        q = q_func(x, A, B)
        return render(q, opacities, colors).sum()

    x0 = 0.5 * (X_lb + X_ub)

    # -----------------------------
    # Taylor bound
    # -----------------------------
    L, U, g, f0, M = taylor_bound(x0, X_lb, X_ub, f, samples=10)

    # -----------------------------
    # Monte Carlo test
    # -----------------------------
    viol = 0
    num_test = 200

    for _ in range(num_test):
        x = X_lb + torch.rand(3) * (X_ub - X_lb)

        fx = f(x).item()
        lx = L(x).item()
        ux = U(x).item()

        if fx < lx - 1e-6 or fx > ux + 1e-6:
            viol += 1

    print("==== Taylor Bound Test ====")
    print("f(x0) =", f0.item())
    print("M =", M)
    print("violation rate =", viol / num_test)

if __name__ == "__main__":
    main()

# @torch.no_grad()
# def extract_coeff(
#     w_lb, w_ub,
#     colors
# ):
#     N, H, W = w_lb.shape
#     device, dtype = w_lb.device, w_lb.dtype

#     rgb_lb = torch.zeros((H, W, 3), device=device, dtype=dtype)
#     rgb_ub = torch.zeros((H, W, 3), device=device, dtype=dtype)

#     d_lb = torch.zeros((N, H, W, 3), device=device, dtype=dtype)
#     d_ub = torch.zeros((N, H, W, 3), device=device, dtype=dtype)

#     for i in range(N - 1, -1, -1):
#         c = colors[i].view(1, 1, 3)

#         d_lb[i] = c - rgb_lb
#         d_ub[i] = c - rgb_ub

#         w_l = w_lb[i][..., None] # (H, W, 1)
#         w_u = w_ub[i][..., None] # (H, W, 1)

#         m_lb = (d_lb[i] >= 0)
#         w_sel_lb = torch.where(m_lb, w_l, w_u) # (H, W, 1)
#         rgb_lb = rgb_lb + d_lb[i] * w_sel_lb 

#         m_ub = (d_ub[i] >= 0)
#         w_sel_ub = torch.where(m_ub, w_u, w_l)
#         rgb_ub = rgb_ub + d_ub[i] * w_sel_ub

#     return d_lb, d_ub


# @torch.no_grad()
# def sum_exp_quad_bound(X_lb, X_ub,
#                         Z_ratio, opacities,
#                         semi_A, semi_B, 
#                         d_lb, d_ub, 
#                         cam_const,
#                         sample=32):
        
#     A = semi_A @ semi_A.transpose(-1, -2) # (N,H,W,3,3)
#     B_ub = semi_B @ semi_B.transpose(-1, -2) # (N,3,3)
#     B_lb = B_ub/(Z_ratio[:, None, None])**2 # (N, 3, 3)

#     rgb_lb, rgb_ub = sample_func(X_lb, X_ub, 
#                             opacities,
#                             A, B_lb, B_ub,
#                             d_lb, d_ub,
#                             cam_const,
#                             sample=sample)

#     return rgb_lb, rgb_ub

# @torch.no_grad()
# def sample_func(
#     X_lb, X_ub,
#     opacities,
#     A, B_lb, B_ub,
#     d_lb, d_ub,
#     cam_const,
#     sample=32
# ):
#     N, H, W = A.shape[0:3]
#     device, dtype = A.device, A.dtype

#     # ✔ robust initialization
#     rgb_lb = torch.full((H, W, 3), float('inf'), device=device, dtype=dtype)
#     rgb_ub = torch.full((H, W, 3), float('-inf'), device=device, dtype=dtype)

#     def sum_exp_quad(X, o, A, B, d):
#         # print(f"x,A,B,shape:",X.shape, A.shape, B.shape)
#         Num = X[:, None, None, None, :] @ A @ X[:, None, None, :, None]
#         Num = Num.squeeze(-1).squeeze(-1)  # (N,H,W)

#         Denom = X[:, None, :] @ B @ X[:, :, None]  # (N,1,1)

#         ratio = -0.5 * (Num / Denom)
#         w = o[:, None, None] * torch.exp(ratio)

#         return (d * w[..., None]).sum(dim=0)  # (H,W,3)

#     # -------------------------
#     # helper: evaluate
#     # -------------------------
#     def eval_point(X):
#         rgb_min = sum_exp_quad(X, opacities, A, B_lb, d_ub) # (H,W,3)
#         rgb_max = sum_exp_quad(X, opacities, A, B_lb, d_lb) #sum_exp_quad(X, opacities, A, B_ub, d_ub)
#         #print(rgb_min[0,0], rgb_max[0,0])

#         return rgb_min, rgb_max

#     # =========================================================
#     # 1. corners (8)
#     # =========================================================
#     xl, yl, zl = X_lb[0:1], X_lb[1:2], X_lb[2:3]
#     xu, yu, zu = X_ub[0:1], X_ub[1:2], X_ub[2:3]

#     for x in (xl, xu):
#         for y in (yl, yu):
#             for z in (zl, zu):
                
#                 X = torch.cat((x, y, z), dim=-1) # (3, )
#                 X = X.unsqueeze(0) + cam_const # (N,3)
#                 rgb_min, rgb_max = eval_point(X)

#                 # print(f"rgb_min:", rgb_min)
#                 # print(f"fgb_max:", rgb_max)

#                 # rgb_tmp_min = torch.minimum(rgb_min, rgb_max)
#                 # rgb_tmp_max = torch.maximum(rgb_min, rgb_max)

#                 # rgb_min = rgb_tmp_min
#                 # rgb_max = rgb_tmp_max

#                 #print(f"rgb_min[0,0], rgb_max[0,0]:",rgb_min[0,0], rgb_max[0,0])

#                 def check_bound(rgb_min, rgb_max):
#                     mask = rgb_min > rgb_max
#                     if mask.any():
#                         idx = mask.nonzero(as_tuple=False)
#                         print("[BOUND VIOLATION]")
#                         print("num:", idx.shape[0])
#                         print("max diff:", (rgb_min - rgb_max).max().item())
#                         print("sample idx:", idx[:5])
#                         raise RuntimeError("invalid bound")
                    
#                 check_bound(rgb_min, rgb_max)
                

#                 rgb_lb = torch.minimum(rgb_lb, rgb_min)
#                 rgb_ub = torch.maximum(rgb_ub, rgb_max)

#     # =========================================================
#     # 2. center point
#     # =========================================================
#     X_mid = 0.5 * (X_lb + X_ub)
#     X_mid = X_mid.unsqueeze(0) + cam_const # (N, 3)
#     rgb_min, rgb_max = eval_point(X_mid)

#     rgb_lb = torch.minimum(rgb_lb, rgb_min)
#     rgb_ub = torch.maximum(rgb_ub, rgb_max)

#     # =========================================================
#     # 3. random samples
#     # =========================================================
#     for _ in range(sample):
#         eps = torch.rand_like(X_lb)
#         X_rand = X_lb + eps * (X_ub - X_lb)
#         X_rand = X_rand.unsqueeze(0) + cam_const

#         rgb_min, rgb_max = eval_point(X_rand)

#         rgb_lb = torch.minimum(rgb_lb, rgb_min)
#         rgb_ub = torch.maximum(rgb_ub, rgb_max)

#     # =========================================================
#     # clamp to valid color range
#     # =========================================================
#     rgb_lb = rgb_lb.clamp(0.0, 1.0)
#     rgb_ub = rgb_ub.clamp(0.0, 1.0)

#     return rgb_lb, rgb_ub
                

