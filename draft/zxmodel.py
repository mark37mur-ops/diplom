"""Модель Zhang, Xia (2026) (без индивидуальных уровней, факторы ортогональны низкоранговому признаку v), решаемая нашим
проекционным алгоритмом: шаг по наклонам через M_F, шаг по факторам — PCA остатков с проекцией M_v."""
import numpy as np
import panelfac as pf


def stats_noc(X, Y, F):
    n, T, d = X.shape
    if F.shape[1] == 0:
        return np.einsum('itd,ite->ide', X, X), np.einsum('itd,it->id', X, Y)
    Q = np.linalg.qr(F)[0]
    MX = X - np.einsum('tk,ikd->itd', Q, np.einsum('tk,itd->ikd', Q, X)); My = Y - (Y @ Q) @ Q.T
    return np.einsum('itd,ite->ide', MX, MX), np.einsum('itd,it->id', MX, Y)


def factors_v(W, r, Mv):
    T = W.shape[1]
    if r == 0:
        return np.zeros((T, 0))
    OP = Mv @ (W.T @ W) @ Mv
    vals, vecs = np.linalg.eigh(OP)
    return np.sqrt(T) * vecs[:, ::-1][:, :r]


def fit_noc(Y, X, m, r, v, max_iter=2000, tol=1e-12, inner=5, F0=None, P0=None):
    n, T, d = X.shape
    Mv = np.eye(T) - v @ np.linalg.solve(v.T @ v, v.T)
    G0, Z0 = stats_noc(X, Y, np.zeros((T, 0)))
    F = factors_v(Y - np.einsum('itd,id->it', X, np.linalg.solve(G0, Z0[..., None])[..., 0]), r, Mv) if F0 is None else F0
    G, Z = stats_noc(X, Y, F)
    P = pf.principal_axes(np.linalg.solve(G, Z[..., None])[..., 0], m) if P0 is None else P0
    hist = []
    for it in range(max_iter):
        G, Z = stats_noc(X, Y, F)
        P, L, Bm = pf.lowrank_slopes(G, Z, P, iters=inner)
        W = Y - np.einsum('itd,id->it', X, Bm)
        F = factors_v(W, r, Mv)
        Gam = W @ F / T if r else np.zeros((n, 0))
        hist.append(float(np.sum((W - Gam @ F.T) ** 2)))
        if it > 0 and abs(hist[-2] - hist[-1]) < tol * hist[-1]:
            break
    return dict(P=P, B=Bm, F=F, G=Gam, hist=np.array(hist), iters=len(hist))


def fit_noc_common(Y, X, r, v, max_iter=3000, tol=1e-12):
    """Общие наклоны, без индивидуальных уровней, факторы ортогональны v."""
    n, T, d = X.shape
    Mv = np.eye(T) - v @ np.linalg.solve(v.T @ v, v.T)
    F = np.zeros((T, 0)); hist = []
    G, Z = stats_noc(X, Y, F); b = np.linalg.solve(G.sum(0), Z.sum(0))
    for it in range(max_iter):
        W = Y - X @ b
        F = factors_v(W, r, Mv)
        G, Z = stats_noc(X, Y, F); b = np.linalg.solve(G.sum(0), Z.sum(0))
        W = Y - X @ b; Gam = W @ F / T if r else np.zeros((n, 0))
        hist.append(float(np.sum((W - Gam @ F.T) ** 2)))
        if it > 0 and abs(hist[-2] - hist[-1]) < tol * hist[-1]: break
    return dict(B=np.tile(b, (n, 1)), F=F, G=Gam, c=np.zeros(n), hist=np.array(hist))
