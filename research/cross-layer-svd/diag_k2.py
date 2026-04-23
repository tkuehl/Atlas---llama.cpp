"""Quick diagnostic: K=1 vs K=2 reconstruction scale analysis."""
import pickle
import numpy as np
from littlebit_multirank import _dual_svid_multirank_numpy

with open('qwen05b_gate12_xtx.pkl', 'rb') as f:
    data = pickle.load(f)
W = data['W'].double().numpy()
H = data['XTX'].double().numpy()
r = 512


def reconstruct(W_np, r, K):
    Up, Vp, U_mag, V_mag_u, V_mag_g, V_mag_lv, r_eff, K_eff = \
        _dual_svid_multirank_numpy(W_np, r, K)
    S_u = np.sign(Up)
    S_v = np.sign(Vp)
    Up_abs = U_mag.astype(np.float64) @ V_mag_u.astype(np.float64)
    Vp_abs = V_mag_g.astype(np.float64) @ V_mag_lv.astype(np.float64)
    Up_approx = S_u * Up_abs
    Vp_approx = S_v * Vp_abs
    W_hat = Up_approx @ Vp_approx.T
    return W_hat, Up_abs, Vp_abs


print('Scale analysis of reconstructed W at different K:\n')
W_norm = np.linalg.norm(W)
print(f'Original W: ||W||_F = {W_norm:.4f}')
print(f'Max per-row norm: {np.linalg.norm(W, axis=1).max():.4f}')
print(f'Mean per-row norm: {np.linalg.norm(W, axis=1).mean():.4f}\n')

np.random.seed(42)
x_scale = 1.0 / np.sqrt(W.shape[1])
x = np.random.randn(32, W.shape[1]) * x_scale

for K in [1, 2, 4, 8, 16]:
    W_hat, Up_abs, Vp_abs = reconstruct(W, r, K)
    wh_norm = np.linalg.norm(W_hat)
    err_norm = np.linalg.norm(W - W_hat)
    up_neg_pct = (Up_abs < 0).mean() * 100
    vp_neg_pct = (Vp_abs < 0).mean() * 100

    y = x @ W_hat.T
    y_teach = x @ W.T
    scale_ratio = np.linalg.norm(y) / np.linalg.norm(y_teach)
    out_err = np.linalg.norm(y - y_teach) / np.linalg.norm(y_teach)

    # Max per-row norm of reconstructed W
    row_norms = np.linalg.norm(W_hat, axis=1)

    print(f'K={K:3d}: ||W_hat||={wh_norm:.4f}  ratio_to_W={wh_norm/W_norm:.4f}  '
          f'max_row={row_norms.max():.4f}')
    print(f'       Up_abs neg={up_neg_pct:.3f}%  Vp_abs neg={vp_neg_pct:.3f}%')
    print(f'       random-x forward: ||y||/||y_teach||={scale_ratio:.4f}  '
          f'out_rel_err={out_err:.4f}')
    print()
