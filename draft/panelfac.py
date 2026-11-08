"""Многоиндексная модель панельных данных с латентными факторами в ошибке.

Y_it = c_i + X_it^T P^T l_i + g_i^T f_t + eps_it
"""
import numpy as np


# ---------- данные ----------
def make_data(n=100, T=200, d=20, m=2, r=2, conf=0.5, sigma=1.0, rho=0.6, seed=0):
    rng = np.random.default_rng(seed)
    j = np.arange(1, d + 1)
    P = np.vstack([np.sin(np.pi * k * j / (d + 1)) for k in range(1, m + 1)])
    P /= np.linalg.norm(P, axis=1, keepdims=True)           # строки ортонормированы
    L = rng.normal(size=(n, m)) * 0.7 + np.r_[1.5, 0.5, 0.0, 0.0][:m]
    B = L @ P                                               # матрица наклонов n x d, ранг m
    F = np.zeros((T, r))                                    # факторы: AR(1)
    e = rng.normal(size=(T + 50, r))
    f = np.zeros(r)
    for t in range(T + 50):
        f = rho * f + np.sqrt(1 - rho ** 2) * e[t]
        if t >= 50:
            F[t - 50] = f
    G = rng.normal(size=(n, r)) + np.r_[1.0, 0.0, 0.0, 0.0, 0.0, 0.0][:r]   # нагрузки на факторы
    Phi = conf * rng.normal(size=(n, d, r))                  # связь признаков с факторами
    X = np.einsum('idk,tk->itd', Phi, F) + rng.normal(size=(n, T, d))
    c = rng.normal(size=n)
    Y = c[:, None] + np.einsum('itd,id->it', X, B) + G @ F.T + sigma * rng.normal(size=(n, T))
    return dict(Y=Y, X=X, P=P, L=L, B=B, F=F, G=G, c=c)


# ---------- линейная алгебра ----------
def basis(F, T):
    """Ортонормированный базис span(1, F) размера T x (r+1)."""
    H = np.ones((T, 1)) if F is None or F.shape[1] == 0 else np.column_stack([np.ones(T), F])
    Q, _ = np.linalg.qr(H)
    return Q


def proj_stats(X, Y, Q):
    """Сводные статистики после проекции: G_i = X_i^T M X_i, Z_i = X_i^T M Y_i."""
    MX = X - np.einsum('tk,ikd->itd', Q, np.einsum('tk,itd->ikd', Q, X))
    G = np.einsum('itd,ite->ide', MX, MX)
    Z = np.einsum('itd,it->id', MX, Y)
    return G, Z


def ols_slopes(G, Z):
    return np.linalg.solve(G, Z[..., None])[..., 0]


def principal_axes(Bmat, m):
    """Первые m главных осей облака наклонов (PCA без центрирования)."""
    _, _, Vt = np.linalg.svd(Bmat, full_matrices=False)
    return Vt[:m]


def lowrank_slopes(G, Z, P0, iters=20, lam=0.0, tol=1e-10):
    """Многоиндексные наклоны: чередуем l_i и B (релаксация), затем главные оси."""
    n, d = Z.shape
    m = P0.shape[0]
    Bm = P0.copy()
    prev = np.inf
    for _ in range(iters):
        BG = np.einsum('ka,iab->ikb', Bm, G)
        Lm = np.linalg.solve(np.einsum('ikb,lb->ikl', BG, Bm), np.einsum('ka,ia->ik', Bm, Z)[..., None])[..., 0]
        A = np.einsum('ik,il,iab->kalb', Lm, Lm, G).reshape(m * d, m * d)
        rhs = np.einsum('ik,ia->ka', Lm, Z).reshape(m * d)
        Bm = np.linalg.solve(A + lam * np.eye(m * d), rhs + lam * P0.reshape(-1)).reshape(m, d)
        Bmat = Lm @ Bm
        obj = np.einsum('ia,iab,ib->', Bmat, G, Bmat) - 2 * np.sum(Bmat * Z)
        if prev - obj < tol * (abs(obj) + 1):
            break
        prev = obj
    P = principal_axes(Bmat, m)
    return P, Bmat @ P.T, Bmat


def factor_step(Y, X, Bmat, r):
    """Факторы по остаткам: первые r главных компонент центрированной матрицы n x T."""
    W = Y - np.einsum('itd,id->it', X, Bmat)
    W = W - W.mean(axis=1, keepdims=True)
    _, s, Vt = np.linalg.svd(W, full_matrices=False)
    T = Y.shape[1]
    return np.sqrt(T) * Vt[:r].T, s


def ssr(Y, X, Bmat, Q):
    W = Y - np.einsum('itd,id->it', X, Bmat)
    return np.sum(W ** 2) - np.sum((W @ Q) ** 2)


def init_factors(Y, X, r):
    """Начальные факторы: главные компоненты всей панели (ряды Y и все ряды X)."""
    n, T, d = X.shape
    S = np.vstack([Y, X.transpose(0, 2, 1).reshape(n * d, T)])
    S = S - S.mean(axis=1, keepdims=True)
    S = S / (S.std(axis=1, keepdims=True) + 1e-12)
    _, _, Vt = np.linalg.svd(S, full_matrices=False)
    return np.sqrt(T) * Vt[:r].T


def fit(Y, X, m, r, max_iter=500, tol=1e-9, F0=None, inner=5, lam=0.0, verbose=False):
    """Чередующийся МНК: проекция -> многоиндексные наклоны -> факторы по остаткам."""
    n, T, d = X.shape
    F = init_factors(Y, X, r) if F0 is None else F0
    Q = basis(F, T)
    G, Z = proj_stats(X, Y, Q)
    P = principal_axes(ols_slopes(G, Z), m) if T > d + r + 1 else np.linalg.qr(np.random.default_rng(0).normal(size=(d, m)))[0].T
    hist = []
    for it in range(max_iter):
        Q = basis(F, T)
        G, Z = proj_stats(X, Y, Q)
        P, Lm, Bmat = lowrank_slopes(G, Z, P, iters=inner, lam=lam)
        hist.append(ssr(Y, X, Bmat, Q))
        if it > 0 and hist[-2] - hist[-1] < tol * hist[-1]:
            break
        F, _ = factor_step(Y, X, Bmat, r)
    Q = basis(F, T)
    H = Q  # c_i и g_i: регрессия остатков на (1, F)
    Hfull = np.column_stack([np.ones(T), F])
    W = Y - np.einsum('itd,id->it', X, Bmat)
    coef = np.linalg.lstsq(Hfull, W.T, rcond=None)[0].T
    return dict(P=P, L=Lm, B=Bmat, F=F, c=coef[:, 0], G=coef[:, 1:], hist=np.array(hist), iters=len(hist))


# ---------- метрики ----------
def rel_err(Bh, B):
    return np.linalg.norm(Bh - B) / np.linalg.norm(B)


def sin_angle(A, B):
    """Синус наибольшего главного угла между пространствами строк A и B."""
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(B.T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.sqrt(max(0.0, 1 - s.min() ** 2)))


def fit_resid(Y, X, m, r, max_iter=1000, tol=1e-9, inner=5, F0=None):
    """Чередование по остаткам: при фиксированных (F, gamma) вычитаем факторную часть и оцениваем наклоны,
    затем по остаткам заново оцениваем (F, gamma). В отличие от fit, нагрузки gamma не исключаются проекцией."""
    n, T, d = X.shape
    F = init_factors(Y, X, r) if F0 is None else F0
    H1 = basis(None, T)
    W = Y - Y.mean(1, keepdims=True)
    Gm = W @ F / T                                       # начальные нагрузки на факторы
    P = None
    hist = []
    for it in range(max_iter):
        Zy = Y - Gm @ F.T                                # отклик без факторной части
        G, Z = proj_stats(X, Zy, H1)
        if P is None:
            P = principal_axes(ols_slopes(G, Z), m) if T > d + 1 else np.linalg.qr(np.random.default_rng(0).normal(size=(d, m)))[0].T
        P, Lm, Bmat = lowrank_slopes(G, Z, P, iters=inner)
        Wr = Y - np.einsum('itd,id->it', X, Bmat)
        Wr = Wr - Wr.mean(1, keepdims=True)
        _, _, Vt = np.linalg.svd(Wr, full_matrices=False)
        F = np.sqrt(T) * Vt[:r].T
        Gm = Wr @ F / T
        hist.append(np.sum((Wr - Gm @ F.T) ** 2))
        if it > 0 and hist[-2] - hist[-1] < tol * hist[-1]:
            break
    return dict(P=P, B=Bmat, F=F, hist=np.array(hist))
