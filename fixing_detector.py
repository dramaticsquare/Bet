"""
Rilevazione di partite sospette dalle quote (forense, a posteriori).

Uso:
  python fixing_detector.py --synthetic
  python fixing_detector.py --csv https://www.football-data.co.uk/mmz4281/2324/I1.csv

Pipeline:
 1. de-vigging delle quote (metodo power)
 2. modello indipendente: Elo + regressione logistica multinomiale (no look-ahead)
 3. feature di anomalia: movimento quote, deviazione dal modello,
    movimento verso l'esito realizzato, sorpresa del risultato
 4. punteggio: IsolationForest + somma di z-score robusti
 5. conversione in probabilita' a posteriori con Bayes
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import poisson
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

RNG = np.random.default_rng(42)


# ---------- 1. de-vigging ----------
def devig_power(odds):
    """Rimuove il margine: trova k tale che sum((1/o)^k) = 1."""
    q = 1.0 / np.asarray(odds, dtype=float)
    if q.sum() <= 1.0:
        return q / q.sum()
    k = brentq(lambda k: (q ** k).sum() - 1.0, 1.0, 10.0)
    p = q ** k
    return p / p.sum()


def devig_df(df, cols):
    return np.vstack([devig_power(r) for r in df[list(cols)].to_numpy()])


# ---------- 2. dati ----------
def synthetic_league(n_teams=20, seasons=3, fix_rate=0.01):
    strength = RNG.normal(0, 0.35, n_teams)
    rows = []
    for _ in range(seasons):
        for h in range(n_teams):
            for a in range(n_teams):
                if h == a:
                    continue
                lam_h = np.exp(0.25 + strength[h] - strength[a] * 0.5)
                lam_a = np.exp(strength[a] - strength[h] * 0.5)
                g = np.arange(0, 9)
                M = np.outer(poisson.pmf(g, lam_h), poisson.pmf(g, lam_a))
                p_true = np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])
                p_true /= p_true.sum()
                rows.append((h, a, p_true, lam_h, lam_a))
    rows = [rows[i] for i in RNG.permutation(len(rows))]
    out = []
    for h, a, p_true, lh, la in rows:
        fixed = RNG.random() < fix_rate
        # mercato: apertura rumorosa, chiusura piu' informata
        p_open = np.clip(p_true + RNG.normal(0, 0.03, 3), 0.02, None); p_open /= p_open.sum()
        p_close = np.clip(p_true + RNG.normal(0, 0.012, 3), 0.02, None); p_close /= p_close.sum()
        gh, ga = RNG.poisson(lh), RNG.poisson(la)
        if fixed:
            target = RNG.choice(3)  # esito "voluto"
            if RNG.random() < 0.7:  # solo il 70% dei fix lascia traccia nelle quote
                shift = RNG.uniform(0.05, 0.15)
                p_close = p_close * (1 - shift); p_close[target] += shift
            if RNG.random() < 0.9:  # il risultato viene forzato
                if target == 0: gh, ga = max(gh, ga + 1), ga
                elif target == 1: gh = ga
                else: gh, ga = gh, max(ga, gh + 1)
        res = 0 if gh > ga else (1 if gh == ga else 2)
        margin = 1.05
        o_open = 1.0 / (margin * p_open)
        o_close = 1.0 / (margin * p_close)
        out.append(dict(HomeTeam=h, AwayTeam=a, FTR="HDA"[res], fixed=int(fixed),
                        OH=o_open[0], OD=o_open[1], OA=o_open[2],
                        CH=o_close[0], CD=o_close[1], CA=o_close[2], FTHG=gh, FTAG=ga))
    return pd.DataFrame(out)


def load_real(path):
    df = pd.read_csv(path, encoding="latin-1")
    # apertura = B365; chiusura = Pinnacle closing (fallback B365 closing)
    close = ["PSCH", "PSCD", "PSCA"] if "PSCH" in df else ["B365CH", "B365CD", "B365CA"]
    df = df.dropna(subset=["B365H", "B365D", "B365A", *close, "FTR"]).reset_index(drop=True)
    df = df.rename(columns=dict(zip(["B365H", "B365D", "B365A"], ["OH", "OD", "OA"])))
    df = df.rename(columns=dict(zip(close, ["CH", "CD", "CA"])))
    df["fixed"] = np.nan  # nessuna etichetta
    return df


# ---------- 3. modello indipendente (Elo, senza look-ahead) ----------
def elo_features(df, k=20, home_adv=60):
    r, diffs = {}, []
    for h, a, res in zip(df.HomeTeam, df.AwayTeam, df.FTR):
        rh, ra = r.get(h, 1500.0), r.get(a, 1500.0)
        diffs.append(rh + home_adv - ra)
        e = 1 / (1 + 10 ** (-(rh + home_adv - ra) / 400))
        s = {"H": 1.0, "D": 0.5, "A": 0.0}[res]
        r[h], r[a] = rh + k * (s - e), ra - k * (s - e)
    return np.array(diffs)


def model_probs(df):
    X = elo_features(df).reshape(-1, 1)
    y = df.FTR.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    n = len(df)
    P = np.full((n, 3), np.nan)
    warm = n // 5
    # fit su finestre precedenti, previsione sul blocco successivo (no leakage)
    edges = list(range(warm, n, max(n // 10, 1))) + [n]
    for s, e in zip(edges[:-1], edges[1:]):
        clf = LogisticRegression(max_iter=500).fit(X[:s], y[:s])
        P[s:e] = clf.predict_proba(X[s:e])
    return P, y


# ---------- 4. feature e punteggio ----------
def robust_z(x):
    med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826 + 1e-9
    return (x - med) / mad


def build_features(df):
    p_open = devig_df(df, ["OH", "OD", "OA"])
    p_close = devig_df(df, ["CH", "CD", "CA"])
    p_mod, y = model_probs(df)
    ok = ~np.isnan(p_mod[:, 0])
    idx = np.arange(len(df))
    f = pd.DataFrame(index=df.index)
    f["move"] = np.abs(p_close - p_open).max(axis=1)
    f["dev"] = np.abs(p_close - p_mod).max(axis=1)
    f["toward"] = np.clip(p_close[idx, y] - p_open[idx, y], 0, None)
    f["surprise"] = -np.log(p_close[idx, y])
    return f[ok], ok


def score(f):
    Z = f.apply(robust_z)
    composite = Z["move"] + Z["dev"] + Z["toward"] + 0.5 * Z["surprise"]
    iso = -IsolationForest(n_estimators=300, contamination="auto", random_state=0).fit(f).score_samples(f)
    return composite.to_numpy(), iso


# ---------- 5. Bayes ----------
def bayes(prior, tpr, fpr):
    return tpr * prior / (tpr * prior + fpr * (1 - prior))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv")
    a = ap.parse_args()
    df = synthetic_league() if a.synthetic or not a.csv else load_real(a.csv)
    f, ok = build_features(df)
    df = df[ok].reset_index(drop=True); f = f.reset_index(drop=True)
    comp, iso = score(f)
    df["score_composite"], df["score_iso"] = comp, iso
    print(f"Partite analizzate: {len(df)}")

    if df["fixed"].notna().all():
        y = df["fixed"].to_numpy()
        print(f"Partite truccate (sintetiche): {int(y.sum())} ({y.mean():.2%})")
        print(f"Average precision  composito: {average_precision_score(y, comp):.3f} | IsolationForest: {average_precision_score(y, iso):.3f}")
        print(f"Baseline (casuale): {y.mean():.3f}\n")
        print("Soglia (top %)  segnalate  TP  precisione(PPV)  richiamo  FPR")
        for pct in [0.5, 1, 2, 5, 10]:
            thr = np.percentile(comp, 100 - pct)
            flag = comp >= thr
            tp = int((flag & (y == 1)).sum()); n_f = int(flag.sum())
            print(f"{pct:>12}%  {n_f:>9}  {tp:>2}  {tp / n_f:>14.1%}  {tp / y.sum():>8.1%}  {((flag & (y == 0)).sum() / (y == 0).sum()):.2%}")
        thr = np.percentile(comp, 99); flag = comp >= thr
        tpr = (flag & (y == 1)).sum() / y.sum(); fpr = (flag & (y == 0)).sum() / (y == 0).sum()
        print(f"\nPPV teorico (Bayes) alla soglia top 1%: TPR={tpr:.2f}, FPR={fpr:.3f}")
        for prior in [0.001, 0.01, 0.05, 0.20]:
            print(f"  prior {prior:.1%} -> P(truccata | segnalata) = {bayes(prior, tpr, fpr):.1%}")
    else:
        print("\nTop 15 partite piu' anomale (non sono prove, solo candidate da verificare):")
        cols = [c for c in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "OH", "OD", "OA", "CH", "CD", "CA", "score_composite"] if c in df]
        print(df.sort_values("score_composite", ascending=False)[cols].head(15).to_string(index=False))
        print("\nSenza etichette non si puo' stimare la precisione. Con TPR~0.3 e FPR~0.01:")
        for prior in [0.001, 0.01, 0.05]:
            print(f"  prior {prior:.1%} -> P(truccata | segnalata) = {bayes(prior, 0.3, 0.01):.1%}")


if __name__ == "__main__":
    main()
