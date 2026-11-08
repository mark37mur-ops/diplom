"""Мягкий малый ранг: наклоны лежат около подпространства индексов, а не строго в нём.

b_i = P^T l_i + delta_i,  delta_i ортогональны строкам P.
Оценка при известных P и F:  b_i = (G_i + kappa Q)^{-1} Z_i,  Q = I - P^T P.
kappa = 0 — МНК по каждому индивиду, kappa = inf — жёсткий малый ранг.
"""
import numpy as np
import panelfac as pf
import panelmiss as pm


# ---------- общие формулы ----------
def complement(P):
    return np.eye(P.shape[1]) - P.T @ P


def soft_slopes(G, Z, P, kappa):
    """b_i = (G_i + kappa Q)^{-1} Z_i; при kappa = inf — проекция на строки P (жёсткий малый ранг при данном P)."""
    if np.isinf(kappa):
        PG = np.einsum('ka,iab->ikb', P, G)
        L = np.linalg.solve(np.einsum('ikb,lb->ikl', PG, P), np.einsum('ka,ia->ik', P, Z)[..., None])[..., 0]
        return L @ P
    Q = complement(P)
    return np.linalg.solve(G + kappa * Q, Z[..., None])[..., 0]


def eb_stats(G, Z, P):
    """Средние по индивидам ||Q b_ols||^2 и tr(Q G^{-1})."""
    Q = complement(P)
    Gi = np.linalg.inv(G)
    Bo = np.einsum('iab,ib->ia', Gi, Z)
    return float(np.mean(np.sum((Bo @ Q) ** 2, 1))), float(np.mean(np.einsum('ab,iba->i', Q, Gi)))


def eb_kappa(G, Z, P, s2):
    """Эмпирический Байес: E||Q b_ols||^2 = (d-m) tau^2 + s2 * tr(Q G^{-1}),  kappa = s2 / tau^2."""
    d, m = P.shape[1], P.shape[0]
    q, tr = eb_stats(G, Z, P)
    tau2 = max(q - s2 * tr, 0.0) / (d - m)
    return (s2 / tau2 if tau2 > 0 else np.inf), tau2


def shrink_matrix(G, P, kappa):
    """Матрица K в оценке K Z: (G + kappa Q)^{-1}, при kappa = inf — P^T (P G P^T)^{-1} P."""
    if np.isinf(kappa):
        return P.T @ np.linalg.solve(P @ G @ P.T, P)
    return np.linalg.inv(G + kappa * complement(P))


def risk_fixed(G, P, delta, s2, kappa):
    """Риск E||b(kappa) - b||^2 при фиксированном отклонении delta (теорема, часть a)."""
    K = shrink_matrix(G, P, kappa)
    bias = K @ G @ delta - delta
    return float(bias @ bias + s2 * np.trace(K @ G @ K.T))


def risk_bayes(G, P, s2, t2, kappa):
    """Риск при случайных delta ~ (0, t2 Q): t2 tr((KG - I) Q (KG - I)^T) + s2 tr(K G K^T); G может быть стопкой n x d x d."""
    Gs = G[None] if G.ndim == 2 else G
    Q = complement(P); d = P.shape[1]; tot = 0.0
    for Gi in Gs:
        K = shrink_matrix(Gi, P, kappa)
        M = K @ Gi - np.eye(d)
        tot += t2 * np.trace(M @ Q @ M.T) + s2 * np.trace(K @ Gi @ K.T)
    return float(tot)


def risk_iso(kappa, g, s2, d, m, D):
    """Точный риск при G = g I: s2 m/g + (kappa^2 D + s2 g (d-m)) / (g + kappa)^2."""
    if np.isinf(kappa):
        return s2 * m / g + D
    return s2 * m / g + (kappa ** 2 * D + s2 * g * (d - m)) / (g + kappa) ** 2


# ---------- сбалансированная панель ----------
def make_soft_data(tau, n=100, T=100, d=20, m=2, r=2, conf=0.5, sigma=1.0, seed=0):
    """Данные как в pf.make_data, но b_i = P^T l_i + delta_i, delta_i ~ N(0, tau^2 Q)."""
    D = pf.make_data(n=n, T=T, d=d, m=m, r=r, conf=conf, sigma=sigma, seed=seed)
    rng = np.random.default_rng(10_000 + seed)
    Q = complement(D['P'])
    B = D['L'] @ D['P'] + tau * rng.normal(size=(n, d)) @ Q
    Y = D['c'][:, None] + np.einsum('itd,id->it', D['X'], B) + D['G'] @ D['F'].T + sigma * rng.normal(size=(n, T))
    D.update(B=B, Y=Y, tau=tau, sigma=sigma)
    return D


def sigma2_bal(Y, X, F):
    """Оценка дисперсии шума по остаткам МНК по каждому индивиду при данных факторах."""
    n, T, d = X.shape
    Qb = pf.basis(F, T)
    G, Z = pf.proj_stats(X, Y, Qb)
    yy = np.sum(Y ** 2, 1) - np.sum((Y @ Qb) ** 2, 1)
    ssr = yy - np.einsum('ia,ia->i', Z, np.linalg.solve(G, Z[..., None])[..., 0])
    return float(ssr.sum() / (n * (T - d - Qb.shape[1]))), G, Z


def fit_soft(Y, X, m, r, kappa='eb', hard=None, max_iter=300, tol=1e-9):
    """Блочный спуск по J = SSR + kappa * sum ||Q_P b_i||^2:
    наклоны (точно) -> индексы P = главные оси B (точно) -> факторы по остаткам (PCA, точно)."""
    n, T, d = X.shape
    hard = pf.fit(Y, X, m, r) if hard is None else hard
    F, P = hard['F'], hard['P']
    s2, G, Z = sigma2_bal(Y, X, F)
    tau2 = None
    if kappa == 'eb':
        kappa, tau2 = eb_kappa(G, Z, P, s2)
    if np.isinf(kappa):
        return dict(hard, kappa=kappa, tau2=tau2, s2=s2, iters=0)
    hist = []
    for it in range(max_iter):
        Qb = pf.basis(F, T)
        G, Z = pf.proj_stats(X, Y, Qb)
        Bmat = soft_slopes(G, Z, P, kappa)
        P = pf.principal_axes(Bmat, m)
        Qc = complement(P)
        hist.append(pf.ssr(Y, X, Bmat, Qb) + kappa * np.sum((Bmat @ Qc) ** 2))
        if it > 0 and hist[-2] - hist[-1] < tol * hist[-1]:
            break
        F, _ = pf.factor_step(Y, X, Bmat, r)
    Hf = np.column_stack([np.ones(T), F])
    coef = np.linalg.lstsq(Hf, (Y - np.einsum('itd,id->it', X, Bmat)).T, rcond=None)[0].T
    return dict(P=P, B=Bmat, F=F, c=coef[:, 0], G=coef[:, 1:], kappa=kappa, tau2=tau2, s2=s2,
                hist=np.array(hist), iters=len(hist))


def oracle_soft(D, kappa):
    """Мягкая оценка при истинных P и F."""
    n, T, d = D['X'].shape
    G, Z = pf.proj_stats(D['X'], D['Y'], pf.basis(D['F'], T))
    return soft_slopes(G, Z, D['P'], kappa), G


# ---------- панель с пропусками ----------
def unit_stats_y(X, Y, mask, F):
    """G_i, Z_i и ||M_i Y_i||^2, число наблюдений индивида."""
    n, T, d = X.shape
    G = np.zeros((n, d, d)); Z = np.zeros((n, d)); yy = np.zeros(n); no = mask.sum(1)
    for i in range(n):
        o = mask[i]
        H = np.ones((o.sum(), 1)) if F is None or F.shape[1] == 0 else np.column_stack([np.ones(o.sum()), F[o]])
        Qh = np.linalg.qr(H)[0]
        MX = X[i, o] - Qh @ (Qh.T @ X[i, o]); My = Y[i, o] - Qh @ (Qh.T @ Y[i, o])
        G[i] = MX.T @ MX; Z[i] = MX.T @ Y[i, o]; yy[i] = My @ My
    return G, Z, yy, no


def fit_soft_miss(Y, X, mask, m, r, kappa='eb', hard=None, max_iter=200, tol=1e-7, ridge=1e-8):
    n, T, d = X.shape
    hard = pm.fit(Y, X, mask, m=m, r=r) if hard is None else hard
    F, P = hard['F'], hard['P']
    G, Z, yy, no = unit_stats_y(X, Y, mask, F)
    G = G + ridge * np.eye(d)
    ssr = yy - np.einsum('ia,ia->i', Z, np.linalg.solve(G, Z[..., None])[..., 0])
    s2 = float(ssr.sum() / np.sum(no - d - 1 - F.shape[1]))
    kap_eb, tau2 = eb_kappa(G, Z, P, s2)
    q, _ = eb_stats(G, Z, P)
    share = (d - m) * tau2 / q if q > 0 else 0.0          # доля настоящих отклонений в ||Q b_ols||^2
    if kappa == 'eb':
        kappa = kap_eb
    if np.isinf(kappa):
        return dict(hard, kappa=kappa, kappa_eb=kap_eb, tau2=tau2, s2=s2, share=share, iters=0)
    hist = []
    for it in range(max_iter):
        G, Z = pm.unit_stats(X, Y, mask, F)
        Bmat = soft_slopes(G + ridge * np.eye(d), Z, P, kappa)
        P = pm.principal_axes(Bmat, m)
        F, c, Gm = pm.factor_step(pm.residuals(Y, X, Bmat), mask, r, F0=F if r > 0 else None)
        R = pm.residuals(Y, X, Bmat) - c[:, None] - (Gm @ F.T if r else 0.0)
        hist.append(float(np.sum(R[mask] ** 2)) + kappa * np.sum((Bmat @ complement(P)) ** 2))
        if r == 0 and it > 0 and abs(hist[-2] - hist[-1]) < 1e-10 * hist[-1]:
            break
        if it > 0 and hist[-2] - hist[-1] < tol * hist[-1]:
            break
    return dict(P=P, B=Bmat, F=F, c=c, G=Gm, kappa=kappa, kappa_eb=kap_eb, tau2=tau2, s2=s2, share=share,
                hist=np.array(hist), iters=len(hist))


def tau2_split(X, Y, mask, F, P, ridge=1e-8):
    """tau^2 по двум половинам ряда: E<Pi b1, Pi b2> = (d - m) tau^2, если шум в половинах не коррелирован.
    Формула для дисперсии МНК не нужна, поэтому оценка устойчива к гетероскедастичности и автокорреляции внутри половин."""
    n, T, d = X.shape; h = T // 2
    m1 = mask.copy(); m1[:, h:] = False
    m2 = mask.copy(); m2[:, :h] = False
    B1 = pm.ols_slopes(*pm.unit_stats(X, Y, m1, F), ridge=ridge)
    B2 = pm.ols_slopes(*pm.unit_stats(X, Y, m2, F), ridge=ridge)
    Q = complement(P)
    return max(float(np.mean(np.sum((B1 @ Q) * (B2 @ Q), 1))), 0.0) / (d - P.shape[0])


def ar1(rng, shape, rho):
    """Стационарный AR(1) по последней оси времени (ось 1), единичная дисперсия."""
    e = rng.normal(size=shape); x = np.zeros(shape); x[:, 0] = e[:, 0]
    for t in range(1, shape[1]):
        x[:, t] = rho * x[:, t - 1] + np.sqrt(1 - rho ** 2) * e[:, t]
    return x


def tau2_demo(tau, rho_e, rho_x=0.7, n=200, T=100, d=20, m=2, sigma=1.0, seed=0):
    """Сравнение оценок tau^2 при известных P и F (F = 0): обычный ЭБ и по двум половинам."""
    rng = np.random.default_rng(seed)
    P = np.linalg.qr(rng.normal(size=(d, m)))[0].T; Q = complement(P)
    B = rng.normal(size=(n, m)) @ P * 1.5 + tau * rng.normal(size=(n, d)) @ Q
    X = ar1(rng, (n, T, d), rho_x)
    Y = np.einsum('itd,id->it', X, B) + sigma * ar1(rng, (n, T), rho_e)
    mask = np.ones((n, T), bool); F = np.zeros((T, 0))
    G, Z, yy, no = unit_stats_y(X, Y, mask, F)
    s2 = float((yy - np.einsum('ia,ia->i', Z, np.linalg.solve(G, Z[..., None])[..., 0])).sum() / np.sum(no - d - 1))
    return eb_kappa(G, Z, P, s2)[1], tau2_split(X, Y, mask, F, P)
