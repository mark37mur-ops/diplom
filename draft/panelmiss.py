"""Многоиндексная модель с латентными факторами для несбалансированной панели (версия, использованная на данных Dominick's).

Y_it = c_i + X_it^T P^T l_i + g_i^T f_t + eps_it,  наблюдаются только клетки (i, t) с mask[i, t] = True.
Старт факторов — по остаткам МНК без факторов.
"""
import numpy as np


def unit_stats(X, Y, mask, F):
    n, T, d = X.shape
    G = np.zeros((n, d, d)); Z = np.zeros((n, d))
    for i in range(n):
        o = mask[i]
        H = np.ones((o.sum(), 1)) if F is None or F.shape[1] == 0 else np.column_stack([np.ones(o.sum()), F[o]])
        Q = np.linalg.qr(H)[0]
        MX = X[i, o] - Q @ (Q.T @ X[i, o])
        G[i] = MX.T @ MX; Z[i] = MX.T @ Y[i, o]
    return G, Z


def ols_slopes(G, Z, ridge=1e-8):
    return np.linalg.solve(G + ridge * np.eye(G.shape[1]), Z[..., None])[..., 0]


def principal_axes(Bmat, m):
    return np.linalg.svd(Bmat, full_matrices=False)[2][:m]


def lowrank_slopes(G, Z, P0, iters=5):
    n, d = Z.shape; m = P0.shape[0]
    if m >= d:                                       # полный ранг = МНК по каждому магазину
        Bmat = ols_slopes(G, Z); return np.eye(d), Bmat, Bmat
    Bm = P0.copy()
    for _ in range(iters):
        BG = np.einsum('ka,iab->ikb', Bm, G)
        Lm = np.linalg.solve(np.einsum('ikb,lb->ikl', BG, Bm) + 1e-10 * np.eye(m),
                             np.einsum('ka,ia->ik', Bm, Z)[..., None])[..., 0]
        A = np.einsum('ik,il,iab->kalb', Lm, Lm, G).reshape(m * d, m * d)
        rhs = np.einsum('ik,ia->ka', Lm, Z).reshape(m * d)
        Bm = np.linalg.solve(A + 1e-10 * np.eye(m * d), rhs).reshape(m, d)
    Bmat = Lm @ Bm; P = principal_axes(Bmat, m)
    return P, Bmat @ P.T, Bmat


def residuals(Y, X, Bmat):
    return Y - np.einsum('itd,id->it', X, Bmat)


def factor_step(W, mask, r, F0=None, inner=5):
    n, T = W.shape
    if r == 0:
        return np.zeros((T, 0)), np.array([W[i, mask[i]].mean() for i in range(n)]), np.zeros((n, 0))
    if F0 is None:
        c = np.array([W[i, mask[i]].mean() for i in range(n)])
        F = np.sqrt(T) * np.linalg.svd(np.where(mask, W - c[:, None], 0.0), full_matrices=False)[2][:r].T
    else:
        F = F0.copy()
    for _ in range(inner):
        C = np.array([np.linalg.lstsq(np.column_stack([np.ones(mask[i].sum()), F[mask[i]]]), W[i, mask[i]], rcond=None)[0] for i in range(n)])
        c, Gm = C[:, 0], C[:, 1:]
        for t in range(T):
            o = mask[:, t]; F[t] = np.linalg.lstsq(Gm[o], W[o, t] - c[o], rcond=None)[0]
    F = F - F.mean(0); F = np.sqrt(T) * np.linalg.svd(F, full_matrices=False)[0]
    C = np.array([np.linalg.lstsq(np.column_stack([np.ones(mask[i].sum()), F[mask[i]]]), W[i, mask[i]], rcond=None)[0] for i in range(n)])
    return F, C[:, 0], C[:, 1:]


def fit(Y, X, mask, m, r, max_iter=200, tol=1e-7):
    n, T, d = X.shape
    if r == 0:
        F = np.zeros((T, 0))
    else:                                            # старт: факторы по остаткам МНК без факторов
        G0, Z0 = unit_stats(X, Y, mask, None)
        F, _, _ = factor_step(residuals(Y, X, ols_slopes(G0, Z0)), mask, r)
    P = principal_axes(ols_slopes(*unit_stats(X, Y, mask, F)), min(m, d))
    hist = []
    for it in range(max_iter):
        G, Z = unit_stats(X, Y, mask, F)
        P, Lm, Bmat = lowrank_slopes(G, Z, P)
        F, c, Gm = factor_step(residuals(Y, X, Bmat), mask, r, F0=F if r > 0 else None)
        R = residuals(Y, X, Bmat) - c[:, None] - (Gm @ F.T if r else 0.0)
        hist.append(float(np.sum(R[mask] ** 2)))
        if r == 0 or (it > 0 and hist[-2] - hist[-1] < tol * hist[-1]):
            break
    return dict(P=P, B=Bmat, F=F, c=c, G=Gm, hist=np.array(hist))


def fit_twoway(Y, X, mask, max_iter=200, tol=1e-9):
    # свои наклоны у каждого магазина + общий эффект недели (фактор с нагрузками 1)
    n, T, d = X.shape; delta = np.zeros(T); prev = np.inf
    for _ in range(max_iter):
        Bmat = ols_slopes(*unit_stats(X, Y - delta[None, :], mask, None))
        W = residuals(Y, X, Bmat) - delta[None, :]
        c = np.array([W[i, mask[i]].mean() for i in range(n)])
        R = residuals(Y, X, Bmat) - c[:, None]
        delta = np.array([R[mask[:, t], t].mean() for t in range(T)]); delta -= delta.mean()
        s = float(np.sum((R - delta[None, :])[mask] ** 2))
        if prev - s < tol * s:
            break
        prev = s
    return dict(B=Bmat, c=c, delta=delta)
