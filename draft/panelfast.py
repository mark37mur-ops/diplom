"""Векторизованная версия алгоритма для панели с пропусками (те же шаги, что в panelmiss.py и panelsoft.py).

Y_it = c_i + X_it^T b_i + g_i^T f_t + eps_it, наблюдаются клетки с mask[i, t] = True.
Наклоны: жёсткий малый ранг b_i = P^T l_i, мягкий (G_i + kappa Q)^{-1} Z_i, общие для всех, общие внутри известных групп.
"""
import numpy as np


def _H(F):
    T = F.shape[0]
    return np.column_stack([np.ones(T), F])


def unit_stats(X, Y, mask, F, ridge=1e-8):
    """G_i = X_i^T M_i X_i, Z_i = X_i^T M_i Y_i, yy_i = ||M_i Y_i||^2, M_i — остатки регрессии на (1, F) по наблюдённым t."""
    m = mask.astype(float); H = _H(F if F is not None else np.zeros((X.shape[1], 0))); k = H.shape[1]
    Xm = X * m[..., None]; Ym = Y * m; XmT = Xm.transpose(0, 2, 1)
    XX = XmT @ X; XY = (XmT @ Y[..., None])[..., 0]; YYs = np.sum(Ym * Y, 1)
    XH = XmT @ H; YH = Ym @ H; HH = (m @ (H[:, :, None] * H[:, None, :]).reshape(len(H), -1)).reshape(-1, k, k)
    HHi = np.linalg.inv(HH + 1e-12 * np.eye(k))
    XHHi = XH @ HHi
    G = XX - XHHi @ XH.transpose(0, 2, 1) + ridge * np.eye(X.shape[2])
    Z = XY - (XHHi @ YH[..., None])[..., 0]
    yy = YYs - np.einsum('ik,ik->i', (YH[:, None, :] @ HHi)[:, 0, :], YH)
    return G, Z, yy, mask.sum(1)


def solve(G, Z):
    return np.linalg.solve(G, Z[..., None])[..., 0]


def principal_axes(B, m):
    return np.linalg.svd(B, full_matrices=False)[2][:m]


def complement(P):
    return np.eye(P.shape[1]) - P.T @ P


def residuals(Y, X, B):
    return Y - np.einsum('itd,id->it', X, B)


def factor_step(W, mask, r, F0=None, inner=10):
    """min по (c, Gamma, F) суммы (W_it - c_i - g_i^T f_t)^2 по наблюдённым клеткам: чередование регрессий."""
    n, T = W.shape; m = mask.astype(float); Wm = np.where(mask, W, 0.0)
    if r == 0:
        return np.zeros((T, 0)), Wm.sum(1) / m.sum(1), np.zeros((n, 0))
    if F0 is None:
        c = Wm.sum(1) / m.sum(1)
        F = np.sqrt(T) * np.linalg.svd(np.where(mask, W - c[:, None], 0.0), full_matrices=False)[2][:r].T
    else:
        F = F0.copy()
    outer = lambda A: (A[:, :, None] * A[:, None, :]).reshape(len(A), -1)
    for _ in range(inner):
        H = _H(F)
        A = (m @ outer(H)).reshape(n, r + 1, r + 1) + 1e-10 * np.eye(r + 1); bb = Wm @ H
        C = np.linalg.solve(A, bb[..., None])[..., 0]; c, Gm = C[:, 0], C[:, 1:]
        At = (m.T @ outer(Gm)).reshape(T, r, r) + 1e-10 * np.eye(r); bt = np.where(mask, W - c[:, None], 0.0).T @ Gm
        F = np.linalg.solve(At, bt[..., None])[..., 0]
    F = F - F.mean(0); F = np.sqrt(T) * np.linalg.svd(F, full_matrices=False)[0]
    H = _H(F)
    A = (m @ outer(H)).reshape(n, r + 1, r + 1) + 1e-10 * np.eye(r + 1)
    C = np.linalg.solve(A, (Wm @ H)[..., None])[..., 0]
    return F, C[:, 0], C[:, 1:]


def ssr(Y, X, mask, B, F, c, Gm):
    R = residuals(Y, X, B) - c[:, None] - (Gm @ F.T if F.shape[1] else 0.0)
    return float(np.sum(R[mask] ** 2))


def lowrank_slopes(G, Z, P0, iters=5):
    n, d = Z.shape; m = P0.shape[0]
    if m >= d:
        B = solve(G, Z); return np.eye(d), B
    Bm = P0.copy()
    for _ in range(iters):
        BG = np.einsum('ka,iab->ikb', Bm, G)
        L = np.linalg.solve(np.einsum('ikb,lb->ikl', BG, Bm) + 1e-10 * np.eye(m), np.einsum('ka,ia->ik', Bm, Z)[..., None])[..., 0]
        A = np.einsum('ik,il,iab->kalb', L, L, G).reshape(m * d, m * d)
        Bm = np.linalg.solve(A + 1e-10 * np.eye(m * d), np.einsum('ik,ia->ka', L, Z).reshape(m * d)).reshape(m, d)
    B = L @ Bm
    return principal_axes(B, m), B


def soft_slopes(G, Z, P, kappa):
    if np.isinf(kappa):
        PG = np.einsum('ka,iab->ikb', P, G)
        L = np.linalg.solve(np.einsum('ikb,lb->ikl', PG, P), np.einsum('ka,ia->ik', P, Z)[..., None])[..., 0]
        return L @ P
    return solve(G + kappa * complement(P), Z)


def group_slopes(G, Z, groups):
    B = np.zeros_like(Z)
    for g in np.unique(groups):
        s = groups == g
        B[s] = np.linalg.solve(G[s].sum(0), Z[s].sum(0))
    return B


def _init_F(Y, X, mask, r):
    G, Z, _, _ = unit_stats(X, Y, mask, None)
    return factor_step(residuals(Y, X, solve(G, Z)), mask, r)[0] if r else np.zeros((Y.shape[1], 0))


def fit(Y, X, mask, m, r, kind='lr', groups=None, max_iter=300, tol=1e-7, F0=None):
    """kind: 'lr' — малый ранг ранга m (m >= d — МНК по каждому), 'common' — общие наклоны, 'group' — общие внутри групп."""
    n, T, d = X.shape
    F = _init_F(Y, X, mask, r) if F0 is None else F0
    G, Z, _, _ = unit_stats(X, Y, mask, F)
    P = principal_axes(solve(G, Z), min(m, d)) if kind == 'lr' else None
    hist = []
    for it in range(max_iter):
        G, Z, _, _ = unit_stats(X, Y, mask, F)
        if kind == 'lr':
            P, B = lowrank_slopes(G, Z, P)
        elif kind == 'common':
            B = np.tile(np.linalg.solve(G.sum(0), Z.sum(0)), (n, 1))
        else:
            B = group_slopes(G, Z, groups)
        F, c, Gm = factor_step(residuals(Y, X, B), mask, r, F0=F if r else None)
        hist.append(ssr(Y, X, mask, B, F, c, Gm))
        if r == 0 or (it > 0 and hist[-2] - hist[-1] < tol * hist[-1]):
            break
    return dict(P=P, B=B, F=F, c=c, G=Gm, hist=np.array(hist), iters=len(hist))


def eb_parts(G, Z, yy, nobs, P, r):
    d, m = P.shape[1], P.shape[0]
    Bo = solve(G, Z); Q = complement(P); Gi = np.linalg.inv(G)
    s2 = float((yy - np.einsum('ia,ia->i', Z, Bo)).sum() / np.sum(nobs - d - 1 - r))
    q = float(np.mean(np.sum((Bo @ Q) ** 2, 1))); tr = float(np.mean(np.einsum('ab,iba->i', Q, Gi)))
    tau2 = max(q - s2 * tr, 0.0) / (d - m)
    return s2, tau2, (d - m) * tau2 / q if q > 0 else 0.0


def tau2_split(X, Y, mask, F, P):
    T = Y.shape[1]; h = T // 2
    m1 = mask.copy(); m1[:, h:] = False; m2 = mask.copy(); m2[:, :h] = False
    G1, Z1, _, _ = unit_stats(X, Y, m1, F); G2, Z2, _, _ = unit_stats(X, Y, m2, F)
    Q = complement(P)
    return max(float(np.mean(np.sum((solve(G1, Z1) @ Q) * (solve(G2, Z2) @ Q), 1))), 0.0) / (P.shape[1] - P.shape[0])


def fit_soft(Y, X, mask, m, r, kappa, hard, max_iter=300, tol=1e-7):
    """Блочный спуск по SSR + kappa sum ||Q_P b_i||^2 со старта в жёсткой оценке. kappa — число, 'eb' или 'split'."""
    n, T, d = X.shape
    F, P = hard['F'], hard['P']
    G, Z, yy, no = unit_stats(X, Y, mask, F)
    s2, tau2, share = eb_parts(G, Z, yy, no, P, r)
    t2s = None
    if kappa == 'eb':
        kappa = s2 / tau2 if tau2 > 0 else np.inf
    elif kappa == 'split':
        t2s = tau2_split(X, Y, mask, F, P); kappa = s2 / t2s if t2s > 0 else np.inf
    info = dict(kappa=kappa, s2=s2, tau2=tau2, share=share, tau2_split=t2s)
    if np.isinf(kappa):
        return dict(hard, **info, iters=0)
    hist = []
    for it in range(max_iter):
        G, Z, _, _ = unit_stats(X, Y, mask, F)
        B = soft_slopes(G, Z, P, kappa)
        P = principal_axes(B, m)
        F, c, Gm = factor_step(residuals(Y, X, B), mask, r, F0=F if r else None)
        hist.append(ssr(Y, X, mask, B, F, c, Gm) + kappa * np.sum((B @ complement(P)) ** 2))
        if it > 0 and hist[-2] - hist[-1] < tol * hist[-1]:
            break
    return dict(P=P, B=B, F=F, c=c, G=Gm, hist=np.array(hist), iters=len(hist), **info)


def fit_twoway(Y, X, mask, max_iter=300, tol=1e-9):
    n, T, d = X.shape; delta = np.zeros(T); prev = np.inf; m = mask.astype(float)
    for _ in range(max_iter):
        G, Z, _, _ = unit_stats(X, Y - delta[None, :], mask, None)
        B = solve(G, Z)
        R = residuals(Y, X, B); c = np.where(mask, R - delta[None, :], 0).sum(1) / m.sum(1)
        R = R - c[:, None]
        delta = np.where(mask, R, 0).sum(0) / np.maximum(m.sum(0), 1); delta -= delta.mean()
        s = float(np.sum((R - delta[None, :])[mask] ** 2))
        if prev - s < tol * s: break
        prev = s
    return dict(B=B, c=c, delta=delta, kind='twoway')
