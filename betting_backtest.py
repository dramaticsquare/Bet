"""
Backtest walk-forward di strategie di value betting con CLV e intervalli di confidenza.

Uso:
  python betting_backtest.py --synthetic
  python betting_backtest.py --csv URL1 URL2 URL3 ...   (stessa lega, piu' stagioni)
  es. https://www.football-data.co.uk/mmz4281/2223/I1.csv

Convenzioni (football-data.co.uk):
  quota "soft" a cui si scommette = B365 (H/D/A)
  riferimento "sharp" al momento della scommessa = Pinnacle PS (H/D/A)
  chiusura per misurare il CLV = Pinnacle closing PSC (H/D/A)
La chiusura NON viene mai usata per decidere (sarebbe look-ahead), solo per valutare.

Strategie:
  model   : Elo + logistica multinomiale (solo passato)
  sharp   : probabilita' de-viggate di Pinnacle come "verita'", si cerca il book soft piu' alto
  blend   : 50% model + 50% sharp
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression

RNG = np.random.default_rng(7)
OUT = {"H": 0, "D": 1, "A": 2}


# ---------- utilita' ----------
def devig_power(odds):
    q = 1.0 / np.asarray(odds, float)
    if q.sum() <= 1.0:
        return q / q.sum()
    k = brentq(lambda k: (q ** k).sum() - 1.0, 1.0, 10.0)
    p = q ** k
    return p / p.sum()


def devig(arr):
    return np.vstack([devig_power(r) for r in arr])


# ---------- dati ----------
def synthetic(seasons=8, n_teams=20, soft_noise=0.03, sharp_noise=0.015, close_noise=0.008):
    strength = RNG.normal(0, 0.35, n_teams)
    rows = []
    for _ in range(seasons):
        pairs = [(h, a) for h in range(n_teams) for a in range(n_teams) if h != a]
        for i in RNG.permutation(len(pairs)):
            h, a = pairs[i]
            lh = np.exp(0.25 + strength[h] - 0.5 * strength[a])
            la = np.exp(strength[a] - 0.5 * strength[h])
            g = np.arange(9)
            M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
            pt = np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()]); pt /= pt.sum()

            def odds(noise, margin):
                p = np.clip(pt + RNG.normal(0, noise, 3), 0.02, None); p /= p.sum()
                return 1.0 / (margin * p)
            gh, ga = RNG.poisson(lh), RNG.poisson(la)
            res = "H" if gh > ga else ("D" if gh == ga else "A")
            s, p_, c = odds(soft_noise, 1.06), odds(sharp_noise, 1.03), odds(close_noise, 1.03)
            rows.append(dict(HomeTeam=h, AwayTeam=a, FTR=res,
                             B365H=s[0], B365D=s[1], B365A=s[2],
                             PSH=p_[0], PSD=p_[1], PSA=p_[2],
                             PSCH=c[0], PSCD=c[1], PSCA=c[2]))
    return pd.DataFrame(rows)


def load_real(paths):
    dfs = [pd.read_csv(p, encoding="latin-1") for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    need = ["B365H", "B365D", "B365A", "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA", "FTR"]
    df = df.dropna(subset=need).reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


# ---------- modello ----------
def elo_diff(df, k=20, home_adv=60):
    r, d = {}, []
    for h, a, res in zip(df.HomeTeam, df.AwayTeam, df.FTR):
        rh, ra = r.get(h, 1500.0), r.get(a, 1500.0)
        d.append(rh + home_adv - ra)
        e = 1 / (1 + 10 ** (-(rh + home_adv - ra) / 400))
        s = {"H": 1.0, "D": 0.5, "A": 0.0}[res]
        r[h], r[a] = rh + k * (s - e), ra - k * (s - e)
    return np.array(d)


def model_probs(df, warm_frac=0.2, block=190):
    X = elo_diff(df).reshape(-1, 1)
    y = df.FTR.map(OUT).to_numpy()
    n = len(df); warm = int(n * warm_frac)
    P = np.full((n, 3), np.nan)
    edges = list(range(warm, n, block)) + [n]
    for s, e in zip(edges[:-1], edges[1:]):
        P[s:e] = LogisticRegression(max_iter=500).fit(X[:s], y[:s]).predict_proba(X[s:e])
    return P, y, warm


# ---------- backtest ----------
def run(p_est, o_soft, p_close, y, thr, max_odds=8.0, kelly_frac=0.25, cap=0.03):
    edge_all = p_est * o_soft - 1
    pick = edge_all.argmax(axis=1)
    idx = np.arange(len(y))
    edge = edge_all[idx, pick]; odds = o_soft[idx, pick]
    mask = (edge > thr) & (odds <= max_odds)
    n = int(mask.sum())
    if n < 30:
        return None
    win = (pick == y)[mask]
    profit = np.where(win, odds[mask] - 1, -1.0)
    clv = (odds * p_close[idx, pick] - 1)[mask]       # EV della quota presa vs chiusura "giusta"
    roi = profit.mean()
    se = profit.std(ddof=1) / np.sqrt(n)
    boot = [RNG.choice(profit, n).mean() for _ in range(2000)]
    # Kelly frazionato
    br, peak, mdd = 1.0, 1.0, 0.0
    for e, o, p in zip(edge[mask], odds[mask], profit):
        f = min(kelly_frac * e / (o - 1), cap)
        br *= 1 + f * p
        peak = max(peak, br); mdd = max(mdd, 1 - br / peak)
    return dict(bets=n, roi=roi, lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
                z=roi / se, clv=clv.mean(), kelly_final=br, max_dd=mdd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", nargs="+")
    ap.add_argument("--soft-noise", type=float, default=0.03, help="solo synthetic: rumore del book soft")
    a = ap.parse_args()
    df = load_real(a.csv) if a.csv else synthetic(soft_noise=a.soft_noise)

    P_mod, y, warm = model_probs(df)
    o_soft = df[["B365H", "B365D", "B365A"]].to_numpy(float)
    p_sharp = devig(df[["PSH", "PSD", "PSA"]].to_numpy(float))
    p_close = devig(df[["PSCH", "PSCD", "PSCA"]].to_numpy(float))
    sl = slice(warm, None)
    o_soft, p_sharp, p_close, P_mod, y = o_soft[sl], p_sharp[sl], p_close[sl], P_mod[sl], y[sl]

    margin = (1 / o_soft).sum(axis=1).mean() - 1
    print(f"Partite fuori campione: {len(y)} | margine medio book soft: {margin:.1%}")
    flat_all = np.mean(np.where(np.arange(3)[None, :] == y[:, None], o_soft - 1, -1.0))
    print(f"ROI scommettendo a caso su tutto (baseline): {flat_all:.1%}\n")

    strategies = {"model": P_mod, "sharp": p_sharp, "blend": 0.5 * P_mod + 0.5 * p_sharp}
    print(f"{'strategia':<8}{'soglia':>7}{'scommesse':>10}{'ROI':>8}{'IC95% ROI':>20}{'z':>6}{'CLV':>8}{'Kelly1/4':>10}{'maxDD':>7}")
    for name, p in strategies.items():
        for thr in [0.02, 0.04, 0.06]:
            r = run(p, o_soft, p_close, y, thr)
            if r is None:
                print(f"{name:<8}{thr:>7.0%}   (meno di 30 scommesse)")
                continue
            print(f"{name:<8}{thr:>7.0%}{r['bets']:>10}{r['roi']:>8.1%}{'[%5.1f%%, %5.1f%%]' % (r['lo']*100, r['hi']*100):>20}"
                  f"{r['z']:>6.1f}{r['clv']:>8.1%}{r['kelly_final']:>10.2f}{r['max_dd']:>7.0%}")
    print("\nLettura: serve CLV > 0 stabile e IC95% del ROI interamente > 0 per parlare di vantaggio.")
    print("Nota: il CLV qui e' contro Pinnacle closing; i book soft limitano chi batte la chiusura.")


if __name__ == "__main__":
    main()
