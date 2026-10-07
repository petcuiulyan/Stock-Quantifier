import base64
import io

import altair as alt
import numpy as np
import requests
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier

st.set_page_config(page_title="Quant Lab", layout="wide")
st.title("Quant Lab - top candidati LONG (NASDAQ + materii prime)")
st.caption("Proiect educational. Nu este sfat financiar. Probabilitatile sunt estimari, nu certitudini.")

def has_secret(key):
    try:
        return key in st.secrets
    except Exception:
        return False


FEATS = ["ret_1", "ret_5", "ret_20", "vol_20", "gap", "range", "vol_ratio", "sma_ratio", "rsi",
         "macd", "bb_pos", "atr", "pos_252", "vwap_dist"]
BINS = [0, .45, .5, .55, .6, 1]
LABELS = ["<45%", "45-50%", "50-55%", "55-60%", ">60%"]

NASDAQ = ("AAPL MSFT NVDA AMZN META GOOGL GOOG AVGO TSLA COST NFLX ASML TMUS CSCO AZN PEP LIN ADBE AMD "
          "TXN QCOM ISRG INTU AMGN BKNG HON AMAT CMCSA PDD VRTX ADP PANW GILD ADI MU SBUX LRCX MELI KLAC "
          "INTC REGN CRWD CDNS SNPS MDLZ CTAS ORLY CEG MAR PYPL ABNB MRVL CSX FTNT ADSK DASH WDAY ROP NXPI "
          "AEP PCAR CHTR MNST FANG CPRT PAYX KDP ROST ODFL FAST AXON VRSK EA DDOG XEL EXC TTWO CTSH LULU "
          "KHC GEHC IDXX CCEP BKR ON TEAM CDW DXCM ZS BIIB MDB TTD PLTR APP ARM SHOP").split()

COMMOD = {"GC=F": "Aur", "SI=F": "Argint", "PL=F": "Platina", "PA=F": "Paladiu", "HG=F": "Cupru",
          "CL=F": "Titei WTI", "BZ=F": "Titei Brent", "NG=F": "Gaz natural", "RB=F": "Benzina",
          "HO=F": "Motorina", "ZC=F": "Porumb", "ZW=F": "Grau", "ZS=F": "Soia", "KC=F": "Cafea",
          "SB=F": "Zahar", "CC=F": "Cacao", "CT=F": "Bumbac", "LE=F": "Vita", "HE=F": "Porc"}

HORIZ = {1: "1 zi", 5: "1 saptamana (5 zile)", 20: "1 luna (20 zile)"}

with st.sidebar:
    st.header("Setari")
    univ = st.multiselect("Univers", ["NASDAQ-100", "Materii prime"], default=["NASDAQ-100", "Materii prime"])
    extra = st.text_input("Companie sau ticker (analiza focalizata; gol = scanez universul)")
    horizon = st.selectbox("Perioada de prognoza", list(HORIZ), index=1, format_func=lambda h: HORIZ[h])
    start = st.date_input("Date de la", pd.Timestamp("2015-01-01"))
    split = st.date_input("Separare train/test", pd.Timestamp("2022-01-01"))
    top_k = st.slider("Cate candidati afisez", 1, 10, 3)
    cost_bps = st.slider("Cost per tranzactie (bps)", 0, 30, 5)
    run = st.button("Analizeaza", type="primary")


@st.cache_data(show_spinner="Descarc datele (poate dura 1-2 minute)...", ttl=3600)
def load(tickers, start):
    frames = []
    for i in range(0, len(tickers), 40):
        part = yf.download(list(tickers[i:i + 40]), start=str(start), auto_adjust=True, progress=False)
        if len(part):
            frames.append(part)
    return pd.concat(frames, axis=1)


def features(raw, t, h):
    d = pd.DataFrame({"open": raw["Open"][t], "high": raw["High"][t], "low": raw["Low"][t],
                      "close": raw["Close"][t], "volume": raw["Volume"][t]}).dropna(subset=["close"])
    for c in ("open", "high", "low"):
        d[c] = d[c].fillna(d["close"])
    d["volume"] = d["volume"].fillna(0)
    r = d["close"].pct_change()
    f = pd.DataFrame(index=d.index)
    f["ret_1"], f["ret_5"], f["ret_20"] = r, d["close"].pct_change(5), d["close"].pct_change(20)
    f["vol_20"] = r.rolling(20).std()
    f["gap"] = d["open"] / d["close"].shift(1) - 1
    f["range"] = (d["high"] - d["low"]) / d["close"]
    f["vol_ratio"] = (d["volume"] / d["volume"].rolling(20).mean()).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    f["sma_ratio"] = d["close"] / d["close"].rolling(50).mean() - 1
    delta = d["close"].diff()
    f["rsi"] = 100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean()
                            / (-delta.clip(upper=0)).rolling(14).mean())
    # chart: MACD (histograma normalizata), pozitie in benzile Bollinger, ATR, pozitie in intervalul 52 sapt.
    macd = d["close"].ewm(span=12).mean() - d["close"].ewm(span=26).mean()
    f["macd"] = (macd - macd.ewm(span=9).mean()) / d["close"]
    sma20, sd20 = d["close"].rolling(20).mean(), d["close"].rolling(20).std()
    f["bb_pos"] = (d["close"] - sma20) / (2 * sd20)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - d["close"].shift(1)).abs(),
                    (d["low"] - d["close"].shift(1)).abs()], axis=1).max(axis=1)
    f["atr"] = tr.rolling(14).mean() / d["close"]
    lo, hi = d["close"].rolling(252, min_periods=60).min(), d["close"].rolling(252, min_periods=60).max()
    f["pos_252"] = (d["close"] - lo) / (hi - lo)
    # volum: distanta fata de VWAP pe 60 zile (aproximare a nivelului de volum maxim)
    vwap = (d["close"] * d["volume"]).rolling(60).sum() / d["volume"].rolling(60).sum()
    f["vwap_dist"] = (d["close"] / vwap - 1).replace([np.inf, -np.inf], np.nan).fillna(0)
    f["fwd"] = d["close"].shift(-h) / d["close"] - 1
    f["ticker"] = t
    return f.dropna(subset=FEATS)


def make_model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=0)


def metrics(s, h):
    ppy = 252 / h
    curve = (1 + s).cumprod()
    return {"Total": f"x{curve.iloc[-1]:.2f}",
            "CAGR": f"{curve.iloc[-1] ** (ppy / len(s)) - 1:.1%}",
            "Sharpe": f"{s.mean() / s.std() * np.sqrt(ppy):.2f}",
            "Max drawdown": f"{(curve / curve.cummax() - 1).min():.1%}"}


@st.cache_data(show_spinner=False, ttl=3600)
def load_ohlc(t):
    d = yf.download(t, period="2y", auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    for c in ("open", "high", "low"):
        d[c] = d[c].fillna(d["close"])
    d["volume"] = d["volume"].fillna(0)
    return d


# --- volume profile ---
def profile(w, edges):
    """Volumul fiecarei zile e impartit egal pe nivelurile de pret dintre low si high."""
    vol = np.zeros(len(edges) - 1)
    for h_, l_, v_ in zip(w["high"], w["low"], w["volume"]):
        m = (edges[1:] > l_) & (edges[:-1] < h_)
        if m.any():
            vol[m] += (v_ if v_ > 0 else 1.0) / m.sum()
    return vol


def value_area(vol, poc, share=0.7):
    lo = hi = poc
    acc, tot = vol[poc], vol.sum()
    while acc < share * tot and (lo > 0 or hi < len(vol) - 1):
        up = vol[hi + 1] if hi < len(vol) - 1 else -1
        dn = vol[lo - 1] if lo > 0 else -1
        if up >= dn:
            hi += 1; acc += vol[hi]
        else:
            lo -= 1; acc += vol[lo]
    return lo, hi


def find_clusters(vol, k=3, sep=3, thr=0.5):
    s = np.convolve(vol, np.ones(3) / 3, mode="same")
    peaks = []
    for i in np.argsort(s)[::-1]:
        if all(abs(i - p) >= sep for p in peaks):
            peaks.append(int(i))
        if len(peaks) == k:
            break
    out = []
    for p in peaks:
        a = b = p
        while a > 0 and s[a - 1] >= thr * s[p]:
            a -= 1
        while b < len(s) - 1 and s[b + 1] >= thr * s[p]:
            b += 1
        out.append((a, b, p))
    return out


def analyze(d, n=60, recent=20, bins=40):
    w = d.tail(n)
    lo, hi = w["low"].min(), w["high"].max()
    edges = np.linspace(lo, hi, bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    vol = profile(w, edges)
    poc = int(vol.argmax())
    v0, v1 = value_area(vol, poc)

    # clusterul de trend = unde se concentreaza volumul in ultimele 'recent' zile
    rvol = profile(w.tail(recent), edges)
    rp = int(rvol.argmax())
    a = b = rp
    while a > 0 and rvol[a - 1] >= 0.5 * rvol[rp]:
        a -= 1
    while b < bins - 1 and rvol[b + 1] >= 0.5 * rvol[rp]:
        b += 1

    c = d["close"]
    px, s20, s50 = c.iloc[-1], c.rolling(20).mean().iloc[-1], c.rolling(50).mean().iloc[-1]
    trend = "Ascendent" if (px > s50 and s20 > s50) else "Descendent" if (px < s50 and s20 < s50) else "Lateral"

    # zona de acumulare: cluster major, in jumatatea de jos a intervalului, cu zile "stranse" si volum peste medie
    acc = None
    avg_rng = ((w["high"] - w["low"]) / w["close"]).mean()
    for ca, cb, _ in find_clusters(vol):
        blo, bhi = edges[ca], edges[cb + 1]
        days = w[(w["close"] >= blo) & (w["close"] <= bhi)]
        if len(days) < 0.25 * len(w):
            continue
        rng_in = ((days["high"] - days["low"]) / days["close"]).mean()
        if (rng_in < 0.9 * avg_rng and days["volume"].mean() > w["volume"].mean()
                and (blo + bhi) / 2 <= lo + 0.5 * (hi - lo)):
            acc = (ca, cb)
            break

    zones = ["Alt volum"] * bins
    for i in range(v0, v1 + 1):
        zones[i] = "In Value Area"
    for i in range(a, b + 1):
        zones[i] = "Cluster trend"
    if acc:
        for i in range(acc[0], acc[1] + 1):
            zones[i] = "Zona de acumulare"
    return dict(edges=edges, vol=vol, zones=zones, px=px, poc_p=mids[poc], trend=trend,
                val=edges[v0], vah=edges[v1 + 1], trend_band=(edges[a], edges[b + 1]),
                acc_band=(edges[acc[0]], edges[acc[1] + 1]) if acc else None,
                shift="in sus" if rp > poc else "in jos" if rp < poc else "stabil")
# --- end volume profile ---


def vp_chart(r):
    df = pd.DataFrame({"lo": r["edges"][:-1], "hi": r["edges"][1:], "vol": r["vol"],
                       "zona": r["zones"], "zero": 0.0})
    y = alt.Y("lo:Q", title="Pret", scale=alt.Scale(zero=False))
    bars = alt.Chart(df).mark_rect().encode(
        y=y, y2="hi:Q", x=alt.X("zero:Q", title="Volum"), x2="vol:Q",
        color=alt.Color("zona:N", legend=alt.Legend(orient="bottom", title=None),
                        scale=alt.Scale(domain=["Zona de acumulare", "Cluster trend", "In Value Area", "Alt volum"],
                                        range=["#f5c542", "#2e9cff", "#7a8499", "#3b4252"])))
    ry = alt.Y("p:Q", scale=alt.Scale(zero=False))
    px = alt.Chart(pd.DataFrame({"p": [r["px"]]})).mark_rule(color="white", strokeDash=[5, 3]).encode(y=ry)
    poc = alt.Chart(pd.DataFrame({"p": [r["poc_p"]]})).mark_rule(color="#ff8c42").encode(y=ry)
    return (bars + px + poc).properties(height=360)


# --- journal ---
JCOLS = ["data_semnal", "ticker", "nume", "tip", "prob", "orizont", "flow_call", "flow_put",
         "gap", "open_high", "open_close", "close_close", "ret_H"]
JNUM = ["prob", "orizont", "flow_call", "flow_put", "gap", "open_high", "open_close", "close_close", "ret_H"]


def get_flow(ticker):
    """LOC REZERVAT pentru fluxul de optiuni. Aici se conecteaza ulterior API-ul Unusual Whales
    (cheia ar sta in st.secrets['UW_API_KEY']) sau datele lipite manual. Intoarce (prima_call, prima_put)."""
    return np.nan, np.nan


def base_rates(raw, ok, years=3):
    parts = []
    for t in ok:
        o, h, c = raw["Open"][t], raw["High"][t], raw["Close"][t]
        df = pd.DataFrame({"cc": c / c.shift(1) - 1, "oh": h / o - 1}).dropna()
        df["tip"] = "Materie prima" if t in COMMOD else "Actiune"
        parts.append(df)
    a = pd.concat(parts)
    a = a[a.index >= a.index.max() - pd.DateOffset(years=years)]
    rows = []
    for tip, g in a.groupby("tip"):
        for name, col in [("Inchidere vs ziua anterioara", "cc"), ("Maxim vs deschidere", "oh")]:
            rows.append({"Grup": tip, "Masura": name,
                         **{f">= {x}%": f"{(g[col] >= x / 100).mean():.2%}" for x in (3, 6, 10)},
                         "Zile-instrument": len(g)})
    return pd.DataFrame(rows)


def _gh():
    try:
        if "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets:
            return ({"Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}",
                     "Accept": "application/vnd.github+json"},
                    f"https://api.github.com/repos/{st.secrets['GITHUB_REPO']}/contents/journal.csv")
    except Exception:
        pass
    return None


def _clean(df):
    df = df.reindex(columns=JCOLS)
    for c in JNUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def journal_remote():
    g = _gh()
    if not g:
        return None, None
    try:
        r = requests.get(g[1], headers=g[0], timeout=20)
        if r.status_code == 200:
            js = r.json()
            return pd.read_csv(io.StringIO(base64.b64decode(js["content"]).decode())), js["sha"]
        if r.status_code == 404:
            return pd.DataFrame(columns=JCOLS), None
        st.warning(f"GitHub a raspuns {r.status_code}; folosesc stocarea din sesiune.")
    except Exception as e:
        st.warning(f"GitHub indisponibil: {e}")
    return None, None


def journal_get():
    if "journal" not in st.session_state:
        df, _ = journal_remote()
        st.session_state["journal"] = _clean(df if df is not None else pd.DataFrame(columns=JCOLS))
    return st.session_state["journal"]


def journal_save(df):
    df = _clean(df)
    st.session_state["journal"] = df
    g = _gh()
    if g:
        _, sha = journal_remote()
        body = {"message": "update journal", "content": base64.b64encode(df.to_csv(index=False).encode()).decode()}
        if sha:
            body["sha"] = sha
        try:
            r = requests.put(g[1], headers=g[0], json=body, timeout=20)
            if r.status_code not in (200, 201):
                st.error(f"Salvarea in GitHub a esuat ({r.status_code}). Jurnalul ramane in sesiune; descarca CSV.")
        except Exception as e:
            st.error(f"Salvarea in GitHub a esuat: {e}")


def fill_outcomes(j):
    """Pentru fiecare semnal (data D, dupa inchidere) completeaza ce s-a intamplat in ziua urmatoare."""
    j = j.copy()
    for i, r in j.iterrows():
        if pd.notna(r["close_close"]) and pd.notna(r["ret_H"]):
            continue
        try:
            d = load_ohlc(r["ticker"])
        except Exception:
            continue
        sd = pd.Timestamp(r["data_semnal"])
        pos = d.index.searchsorted(sd)
        if pos >= len(d) or d.index[pos].normalize() != sd:
            continue
        c0 = d["close"].iloc[pos]
        if pos + 1 < len(d):
            n = d.iloc[pos + 1]
            j.loc[i, ["gap", "open_high", "open_close", "close_close"]] = [
                n["open"] / c0 - 1, n["high"] / n["open"] - 1, n["close"] / n["open"] - 1, n["close"] / c0 - 1]
        h = int(r["orizont"])
        if pos + h < len(d):
            j.loc[i, "ret_H"] = d["close"].iloc[pos + h] / c0 - 1
    return j
# --- end journal ---


# --- focus ---
BULL_PAT = {"Hammer", "Bullish engulfing", "Marubozu bullish"}
BEAR_PAT = {"Shooting star", "Bearish engulfing", "Marubozu bearish"}


def candle_patterns(d):
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    up = (h - np.maximum(o, c)) / rng
    lo = (np.minimum(o, c) - l) / rng
    po, pc = o.shift(1), c.shift(1)
    p = pd.DataFrame(index=d.index)
    p["Doji"] = body / rng < 0.1
    p["Hammer"] = (lo >= 0.6) & (up <= 0.15)
    p["Shooting star"] = (up >= 0.6) & (lo <= 0.15)
    p["Bullish engulfing"] = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    p["Bearish engulfing"] = (pc > po) & (c < o) & (o >= pc) & (c <= po)
    p["Marubozu bullish"] = (c > o) & (body / rng >= 0.9)
    p["Marubozu bearish"] = (c < o) & (body / rng >= 0.9)
    return p.fillna(False).astype(bool)


def pattern_stats(d, p):
    nxt = d["close"].shift(-1) / d["close"] - 1
    vr = d["volume"] / d["volume"].rolling(20).mean()
    rows = []
    for name in p.columns:
        m = p[name] & nxt.notna()
        mv = m & (vr >= 1.5)
        rows.append({
            "Pattern": name, "Aparitii": int(m.sum()),
            "% urmatoarea zi verde": f"{(nxt[m] > 0).mean():.0%}" if m.sum() else "-",
            "Medie urmatoarea zi": f"{nxt[m].mean():+.2%}" if m.sum() else "-",
            "Aparitii cu volum >=1.5x": int(mv.sum()),
            "% verde cu volum": f"{(nxt[mv] > 0).mean():.0%}" if mv.sum() else "-"})
    return pd.DataFrame(rows)


def last_session_profile(d5, min_bars=70, bins=40):
    """Profil de volum al ultimei sesiuni complete, din bare de 5 minute."""
    if d5 is None or len(d5) == 0:
        return None
    counts = d5.groupby(d5.index.date).size()
    full = counts[counts >= min_bars]
    if full.empty:
        return None
    day = full.index[-1]
    s = d5[d5.index.date == day]
    lo, hi = s["low"].min(), s["high"].max()
    if not hi > lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    vol = profile(s, edges)
    poc = int(vol.argmax())
    v0, v1 = value_area(vol, poc)
    zones = ["In Value Area" if v0 <= i <= v1 else "Alt volum" for i in range(bins)]
    return dict(day=day, edges=edges, vol=vol, zones=zones, poc_p=mids[poc], val=edges[v0], vah=edges[v1 + 1],
                high=hi, low=lo, open=s["open"].iloc[0], px=s["close"].iloc[-1])


def parse_money(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("$", "").replace(",", "").upper()
    mult = 1.0
    if s.endswith("K"):
        mult, s = 1e3, s[:-1]
    elif s.endswith("M"):
        mult, s = 1e6, s[:-1]
    elif s.endswith("B"):
        mult, s = 1e9, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def guess_col(cols, keys):
    low = {c: str(c).lower().strip() for c in cols}
    for k in keys:
        for c, l in low.items():
            if l == k:
                return c
    for k in keys:
        for c, l in low.items():
            if k in l:
                return c
    return None


def flow_summary(df, tcol, ycol, pcol, ticker, scol=None, ecol=None):
    g = df[df[tcol].astype(str).str.upper().str.strip() == ticker.upper()].copy()
    if g.empty:
        return None
    g["prem"] = g[pcol].map(parse_money)
    typ = g[ycol].astype(str).str.lower().str.strip()
    is_call = typ.str.contains("call") | typ.isin(["c"])
    is_put = typ.str.contains("put") | typ.isin(["p"])
    if scol:
        side = g[scol].astype(str).str.lower()
        ask, bid = side.str.contains("ask"), side.str.contains("bid")
    else:
        ask = bid = pd.Series(False, index=g.index)
    bull = g.loc[(is_call & ask) | (is_put & bid), "prem"].sum()
    bear = g.loc[(is_put & ask) | (is_call & bid), "prem"].sum()
    out = dict(n=len(g), call=g.loc[is_call, "prem"].sum(), put=g.loc[is_put, "prem"].sum(),
               bull=bull, bear=bear, net=(bull - bear) / (bull + bear) if bull + bear > 0 else np.nan,
               short_share=np.nan)
    if ecol:
        dte = (pd.to_datetime(g[ecol], errors="coerce") - pd.Timestamp.today().normalize()).dt.days
        tot = g["prem"].sum()
        if tot > 0:
            out["short_share"] = g.loc[dte <= 7, "prem"].sum() / tot
    return out
# --- end focus ---


def resolve_symbol(q):
    q = q.strip()
    if q.isupper() or any(ch in q for ch in "=^.-"):
        return q
    try:
        for hit in yf.Search(q, max_results=6).quotes:
            if hit.get("quoteType") in ("EQUITY", "ETF", "FUTURE", "INDEX"):
                return hit["symbol"]
    except Exception:
        pass
    return q.upper()


@st.cache_data(show_spinner=False, ttl=900)
def load_intraday(t):
    d = yf.download(t, period="5d", interval="5m", auto_adjust=True, progress=False)
    if d is None or d.empty:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])


def make_focus_model():
    return HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=120,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def compute_focus(t, start):
    raw = load((t,), start)
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([raw.columns, [t]])
    if t not in raw["Close"].columns or raw["Close"][t].notna().sum() < 700:
        return None
    f = features(raw, t, 1)
    d = pd.DataFrame({"open": raw["Open"][t], "high": raw["High"][t], "low": raw["Low"][t],
                      "close": raw["Close"][t], "volume": raw["Volume"][t]}).dropna(subset=["close"])
    for c in ("open", "high", "low"):
        d[c] = d[c].fillna(d["close"])
    d["volume"] = d["volume"].fillna(0)
    lab = f.dropna(subset=["fwd"]).copy()
    lab["target"] = (lab["fwd"] > 0).astype(int)
    if len(lab) < 600:
        return None
    cut = int(len(lab) * 0.75)
    train, test = lab.iloc[:cut - 1], lab.iloc[cut:].copy()
    test["prob"] = make_focus_model().fit(train[FEATS], train["target"]).predict_proba(test[FEATS])[:, 1]
    n = len(test)
    acc = float(((test["prob"] > 0.5).astype(int) == test["target"]).mean())
    lg, sh = test[test["prob"] >= 0.55], test[test["prob"] <= 0.45]
    final = make_focus_model().fit(lab[FEATS], lab["target"])
    last = f.iloc[[-1]]
    try:
        intraday = last_session_profile(load_intraday(t))
    except Exception:
        intraday = None
    return dict(ticker=t, asof=f.index[-1].date(), p_up=float(final.predict_proba(last[FEATS])[0, 1]),
                n_test=n, acc=acc, ci=float(1.96 * np.sqrt(acc * (1 - acc) / n)), base_up=float(test["target"].mean()),
                long_n=len(lg), long_hit=float((lg["target"] == 1).mean()) if len(lg) else np.nan,
                short_n=len(sh), short_hit=float((sh["target"] == 0).mean()) if len(sh) else np.nan,
                ind=last[FEATS].iloc[0].to_dict(), d=d, intraday=intraday)


def render_focus(F):
    t, p = F["ticker"], F["p_up"]
    st.header(f"{COMMOD.get(t, t)} ({t}) - date la {F['asof']}")
    sig = "LONG" if p >= 0.55 else "SHORT" if p <= 0.45 else "FARA AVANTAJ CLAR"
    dirn = sig if sig in ("LONG", "SHORT") else None
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilitate LONG (ziua urmatoare)", f"{p:.0%}")
    c2.metric("Probabilitate SHORT", f"{1 - p:.0%}")
    c3.metric("Semnal model", sig)
    st.caption("LONG = inchiderea de maine peste cea de azi; SHORT = sub. Prag: >=55% long, <=45% short. "
               "Rulat dupa inchiderea pietei US (~23:00 ora Romaniei), altfel ultima zi e incompleta. "
               "Sinteza cu confirmarile e la finalul paginii.")

    lo_, hi_ = F["acc"] - F["ci"], F["acc"] + F["ci"]
    verdict = ("peste hazard" if lo_ > 0.5 else "nu se distinge statistic de aruncarea monedei")
    st.markdown(
        f"**Cat de mult poti avea incredere?** Testat pe ultimele {F['n_test']} zile, necunoscute modelului: "
        f"acuratete **{F['acc']:.1%}** (±{F['ci']:.1%}), {verdict}. Rata de baza (zile verzi): {F['base_up']:.1%}.  \n"
        f"Cand modelul a dat LONG (>=55%): a crescut in **{F['long_hit']:.0%}** din {F['long_n']} cazuri. "
        f"Cand a dat SHORT (<=45%): a scazut in **{F['short_hit']:.0%}** din {F['short_n']} cazuri."
        if F["long_n"] and F["short_n"] else
        f"**Cat de mult poti avea incredere?** Testat pe ultimele {F['n_test']} zile: acuratete "
        f"**{F['acc']:.1%}** (±{F['ci']:.1%}), {verdict}. Rata de baza (zile verzi): {F['base_up']:.1%}. "
        f"Prea putine semnale clare LONG/SHORT in test ca sa le evaluez.")
    if min(F["long_n"], F["short_n"]) < 30:
        st.warning("Sub 30 de semnale de un fel in test: procentele de mai sus sunt nesigure.")

    # ---- indicatori ----
    st.subheader("Indicatori")
    i = F["ind"]
    rows = [
        ("RSI(14)", f"{i['rsi']:.0f}", "supracumparat" if i["rsi"] > 70 else "supravandut" if i["rsi"] < 30 else "neutru"),
        ("MACD (histograma)", f"{i['macd']:+.4f}", "momentum pozitiv" if i["macd"] > 0 else "momentum negativ"),
        ("Pozitie Bollinger", f"{i['bb_pos']:+.2f}",
         "peste banda de sus" if i["bb_pos"] > 1 else "sub banda de jos" if i["bb_pos"] < -1 else "in banda"),
        ("ATR (volatilitate zilnica)", f"{i['atr']:.1%}", ""),
        ("Pozitie in intervalul 52 sapt.", f"{i['pos_252']:.0%}",
         "aproape de maxim" if i["pos_252"] > 0.9 else "aproape de minim" if i["pos_252"] < 0.1 else ""),
        ("Distanta fata de SMA50", f"{i['sma_ratio']:+.1%}", ""),
        ("Distanta fata de VWAP 60z", f"{i['vwap_dist']:+.1%}", ""),
        ("Volum relativ (vs media 20z)", f"{i['vol_ratio']:.1f}x", "ridicat" if i["vol_ratio"] > 1.5 else ""),
        ("Gap la deschidere (ultima zi)", f"{i['gap']:+.1%}", ""),
        ("Randament 5z / 20z", f"{i['ret_5']:+.1%} / {i['ret_20']:+.1%}", "")]
    st.table(pd.DataFrame(rows, columns=["Indicator", "Valoare", "Citire"]).set_index("Indicator"))

    # ---- profil de volum fix ----
    st.subheader("Profil de volum fix (zilnic)")
    k1, k2 = st.columns(2)
    n_days = k1.slider("Fereastra profil (zile)", 30, 120, 60, key="fn")
    n_rec = k2.slider("Fereastra cluster trend (zile)", 10, 30, 20, key="fr")
    r = analyze(F["d"], n_days, n_rec)
    hl = (f"Zona de acumulare {r['acc_band'][0]:.2f} - {r['acc_band'][1]:.2f}" if r["acc_band"]
          else f"Cluster trend {r['trend_band'][0]:.2f} - {r['trend_band'][1]:.2f}")
    st.markdown(f"**Zona evidentiata:** {hl}  \n**Trend:** {r['trend']} | **Pret:** {r['px']:.2f} | "
                f"**POC:** {r['poc_p']:.2f} | **Value Area:** {r['val']:.2f} - {r['vah']:.2f}  \n"
                f"**Volumul recent se muta:** {r['shift']} fata de POC")
    st.altair_chart(vp_chart(r), use_container_width=True)

    st.subheader("Ziua anterioara (profil din bare de 5 minute)")
    sp = F["intraday"]
    prior = None
    if sp:
        pos = ("deasupra Value Area (presiune cumparatoare)" if sp["px"] > sp["vah"] else
               "sub Value Area (presiune vanzatoare)" if sp["px"] < sp["val"] else "in interiorul Value Area")
        prior = True if sp["px"] > sp["vah"] else False if sp["px"] < sp["val"] else None
        st.markdown(f"**Sesiunea {sp['day']}:** deschidere {sp['open']:.2f}, maxim {sp['high']:.2f}, minim {sp['low']:.2f}, "
                    f"inchidere {sp['px']:.2f}  \n**POC:** {sp['poc_p']:.2f} | **Value Area:** {sp['val']:.2f} - "
                    f"{sp['vah']:.2f}  \nInchiderea e {pos}.")
        st.altair_chart(vp_chart(sp), use_container_width=True)
    else:
        st.caption("Date intraday indisponibile pentru acest simbol (Yahoo ofera 5 minute doar pentru ultimele zile).")

    # ---- candlesticks + volum ----
    st.subheader("Candlesticks si volum")
    pc = candle_patterns(F["d"])
    names = [c for c in pc.columns if pc[c].iloc[-1]]
    vol_ = F["d"]["volume"]
    vr = vol_.iloc[-1] / vol_.iloc[-21:-1].mean() if vol_.iloc[-21:-1].mean() > 0 else np.nan
    st.write(f"Ultima zi: **{', '.join(names) if names else 'niciun pattern'}** | volum relativ: "
             f"{vr:.1f}x" if pd.notna(vr) else f"Ultima zi: {', '.join(names) if names else 'niciun pattern'}")
    candle = (True if set(names) & BULL_PAT and not set(names) & BEAR_PAT else
              False if set(names) & BEAR_PAT and not set(names) & BULL_PAT else None)
    st.caption("Istoricul acestui pattern pe aceasta actiune (ca sa vezi daca are vreo valoare predictiva):")
    st.dataframe(pattern_stats(F["d"], pc), use_container_width=True, hide_index=True)

    # ---- flux optiuni ----
    st.subheader("Flux de optiuni (CSV sau tabel lipit)")
    up = st.file_uploader("Incarca CSV", type=["csv", "txt"], key="flowfile")
    txt = st.text_area("... sau lipeste tabelul copiat de pe site", key="flowtxt", height=100)
    fdf, flow = None, None
    try:
        if up is not None:
            fdf = pd.read_csv(up, sep=None, engine="python")
        elif txt.strip():
            fdf = pd.read_csv(io.StringIO(txt), sep=None, engine="python")
    except Exception as e:
        st.error(f"Nu am putut citi datele: {e}")
    if fdf is not None and len(fdf.columns) > 1:
        cols, none = list(fdf.columns), "(niciuna)"
        guess = {"t": guess_col(cols, ["ticker", "symbol", "underlying", "sym"]),
                 "y": guess_col(cols, ["put/call", "put_call", "option_type", "type", "c/p", "call/put", "cp"]),
                 "p": guess_col(cols, ["total_premium", "premium", "prem", "value"]),
                 "s": guess_col(cols, ["side", "bid/ask", "bid_ask", "execution"]),
                 "e": guess_col(cols, ["expiry", "expiration", "exp"])}
        opts = [none] + cols
        m1, m2, m3, m4, m5 = st.columns(5)
        sel = {k: lab.selectbox(txt_, opts, index=opts.index(guess[k]) if guess[k] in opts else 0, key=f"map{k}")
               for k, lab, txt_ in [("t", m1, "Ticker"), ("y", m2, "Call/Put"), ("p", m3, "Premium"),
                                    ("s", m4, "Side (ask/bid)"), ("e", m5, "Expirare")]}
        if all(sel[k] != none for k in "typ"):
            flow = flow_summary(fdf, sel["t"], sel["y"], sel["p"], t,
                                scol=None if sel["s"] == none else sel["s"],
                                ecol=None if sel["e"] == none else sel["e"])
            if flow is None:
                st.info(f"Nu am gasit tranzactii pentru {t} in fisier.")
            else:
                f1, f2, f3 = st.columns(3)
                f1.metric("Prima call", f"${flow['call'] / 1e6:.2f}M")
                f2.metric("Prima put", f"${flow['put'] / 1e6:.2f}M")
                f3.metric("Net bullish", "n/a" if pd.isna(flow["net"]) else f"{flow['net']:+.0%}")
                st.caption(f"{flow['n']} tranzactii. Net bullish = (call la ask + put la bid - put la ask - call la bid) "
                           f"/ total. Datele de flux NU intra in model (nu avem istoric), doar in sinteza de mai jos.")
    flow_v = None if flow is None or pd.isna(flow["net"]) else (True if flow["net"] > 0.15 else False if flow["net"] < -0.15 else None)

    # ---- sinteza ----
    st.subheader("Sinteza")
    items = [("Trend SMA20/50", {"Ascendent": True, "Descendent": False}.get(r["trend"])),
             ("MACD", bool(i["macd"] > 0)),
             ("Pret fata de POC (60z)", bool(r["px"] > r["poc_p"])),
             ("Inchidere vs Value Area ziua anterioara", prior),
             ("Pattern candlestick ultima zi", candle),
             ("Flux de optiuni", flow_v)]
    if dirn:
        want = dirn == "LONG"
        mark = lambda v: "➖" if v is None else ("✅" if v == want else "❌")
        st.table(pd.DataFrame([(a, mark(v)) for a, v in items], columns=["Semnal", f"Aliniat cu {dirn}"]).set_index("Semnal"))
        ok_n = sum(1 for _, v in items if v is not None and v == want)
        used = sum(1 for _, v in items if v is not None)
        st.write(f"**{ok_n} din {used}** semnale disponibile sunt aliniate cu {dirn}.")
    else:
        st.write("Modelul nu vede un avantaj clar in nicio directie; confirmarile nu schimba asta.")
    st.caption("Proiect educational, nu sfat financiar. Probabilitatile sunt estimari; pe o singura actiune, "
               "modelele zilnice au de obicei avantaj mic sau inexistent.")


fq = extra.strip()
if fq:
    if run:
        sym = resolve_symbol(fq)
        with st.spinner(f"Analizez {sym}..."):
            res = compute_focus(sym, start)
        st.session_state["focus"] = res or {"error": f"Date insuficiente sau simbol necunoscut: {sym}"}
    F = st.session_state.get("focus")
    if not F:
        st.info("Apasa 'Analizeaza' in bara laterala pentru a analiza compania introdusa.")
    elif "error" in F:
        st.error(F["error"])
    else:
        render_focus(F)
    st.stop()


if run:
    tick = []
    if "NASDAQ-100" in univ:
        tick += NASDAQ
    if "Materii prime" in univ:
        tick += list(COMMOD)
    tick += extra.upper().split()
    tick = list(dict.fromkeys(tick))
    if not tick:
        st.error("Alege cel putin un univers.")
        st.stop()

    raw = load(tuple(tick), start)
    avail = raw["Close"].columns
    ok = [t for t in tick if t in avail and raw["Close"][t].notna().sum() > 300]
    skipped = [t for t in tick if t not in ok]
    st.session_state["base_stats"] = base_rates(raw, ok)

    data = pd.concat([features(raw, t, horizon) for t in ok]).reset_index()
    data = data.rename(columns={data.columns[0]: "date"})
    labeled = data.dropna(subset=["fwd"]).copy()
    labeled["target"] = (labeled["fwd"] > 0).astype(int)

    # purge: elimina din train zilele ale caror tinte se suprapun cu perioada de test
    cut = pd.Timestamp(split) - pd.tseries.offsets.BDay(horizon)
    train = labeled[labeled["date"] < cut]
    test = labeled[labeled["date"] >= pd.Timestamp(split)].copy()
    model = make_model().fit(train[FEATS], train["target"])
    test["prob"] = model.predict_proba(test[FEATS])[:, 1]
    test["bucket"] = pd.cut(test["prob"], BINS, labels=LABELS)
    calib = test.groupby("bucket", observed=True).agg(
        n=("fwd", "size"), hit=("fwd", lambda x: (x > 0).mean()),
        media=("fwd", "mean"), mediana=("fwd", "median"))

    # backtest cu rebalansare la fiecare 'horizon' zile (fara suprapuneri)
    dates = sorted(test["date"].unique())[::horizon]
    rows, hits = [], []
    for dt in dates:
        g = test[test["date"] == dt]
        picks = g.nlargest(top_k, "prob")
        rows.append((dt, picks["fwd"].mean() - 2 * cost_bps / 10000, g["fwd"].mean()))
        hits += list(picks["fwd"] > 0)
    res = pd.DataFrame(rows, columns=["date", "Strategie", "Benchmark"]).set_index("date")
    base_rate = (test["fwd"] > 0).mean()

    # semnale curente: model antrenat pe tot istoricul disponibil
    final = make_model().fit(labeled[FEATS], labeled["target"])
    last = data.sort_values("date").groupby("ticker").tail(1)
    last = last[last["date"] >= data["date"].max() - pd.Timedelta(days=4)].copy()
    last["prob"] = final.predict_proba(last[FEATS])[:, 1]
    last["tip"] = last["ticker"].map(lambda t: "Materie prima" if t in COMMOD else "Actiune")
    last["nume"] = last["ticker"].map(lambda t: COMMOD.get(t, t))
    signals = last.sort_values("prob", ascending=False)

    ms, mb = metrics(res["Strategie"], horizon), metrics(res["Benchmark"], horizon)
    st.session_state.update(
        res=res, signals=signals, calib=calib, m=(ms, mb), hit=np.mean(hits), base=base_rate, skipped=skipped,
        n=len(ok), asof=data["date"].max().date(), horizon=horizon,
        summary=(f"Univers: {len(ok)} instrumente, orizont {HORIZ[horizon]}, top {top_k}, cost {cost_bps} bps. "
                 f"Rata de succes picks: {np.mean(hits):.1%} vs baza {base_rate:.1%}. "
                 f"Strategie: {ms}. Benchmark: {mb}. Top semnale la {data['date'].max().date()}: "
                 f"{signals.head(top_k)[['nume', 'prob']].round(3).to_dict('records')}"))

tab1, tabv, tabj, tab2, tab3 = st.tabs(
    ["Top candidati LONG", "Profil volum", "Masurare si jurnal", "Backtest", "Asistent AI"])

with tab1:
    if "signals" not in st.session_state:
        st.info("Alege setarile si apasa 'Analizeaza' in bara laterala.")
    else:
        s = st.session_state["signals"]
        st.subheader(f"Top {top_k} - {HORIZ[st.session_state['horizon']]} (date la {st.session_state['asof']})")
        cols = st.columns(min(top_k, 5))
        for i, (_, r) in enumerate(s.head(top_k).iterrows()):
            c = cols[i % len(cols)]
            c.metric(f"{i + 1}. {r['nume']} ({r['tip']})", f"{r['prob']:.0%}",
                     f"20z: {r['ret_20']:+.1%}", delta_color="off")
            cal = st.session_state["calib"]
            lab = pd.cut([r["prob"]], BINS, labels=LABELS)[0]
            if lab in cal.index:
                c.caption(f"In test, la scor {lab}: a crescut in {cal.loc[lab, 'hit']:.0%} din cazuri, "
                          f"mediana {cal.loc[lab, 'mediana']:+.1%} ({int(cal.loc[lab, 'n'])} cazuri)")
            if r["ret_20"] < -0.15 or r["rsi"] < 20:
                c.caption("⚠️ Scadere puternica recenta: posibil eveniment (rezultate, stiri) pe care modelul nu il vede.")
        st.caption("Procentul = probabilitatea estimata de model ca pretul sa fie mai mare la finalul perioadei. "
                   "Valorile peste 60% sunt rare; 50% = nicio informatie.")
        with st.expander("Clasament complet"):
            st.dataframe(s[["nume", "tip", "prob", "ret_20", "rsi", "sma_ratio"]].reset_index(drop=True),
                         use_container_width=True)
        if st.session_state["skipped"]:
            st.caption(f"Ignorate (fara date suficiente): {', '.join(st.session_state['skipped'])}")

with tabv:
    if "signals" not in st.session_state:
        st.info("Ruleaza intai analiza.")
    else:
        s = st.session_state["signals"]
        stocks = s[s["tip"] == "Actiune"]["ticker"].head(2).tolist()
        com = st.selectbox("Materie prima", list(COMMOD), index=0, format_func=lambda t: COMMOD[t])
        k1, k2 = st.columns(2)
        n_days = k1.slider("Fereastra profil (zile)", 30, 120, 60)
        n_rec = k2.slider("Fereastra cluster trend (zile)", 10, 30, 20)
        vp_notes = []
        for t in stocks + [com]:
            st.subheader(f"{COMMOD.get(t, t)} ({t})")
            d = load_ohlc(t)
            if len(d) < n_days + 60:
                st.warning("Prea putine date pentru profil.")
                continue
            r = analyze(d, n_days, n_rec)
            hl = (f"Zona de acumulare {r['acc_band'][0]:.2f} - {r['acc_band'][1]:.2f}" if r["acc_band"]
                  else f"Cluster trend {r['trend_band'][0]:.2f} - {r['trend_band'][1]:.2f}")
            st.markdown(f"**Zona evidentiata:** {hl}  \n"
                        f"**Trend:** {r['trend']} | **Pret:** {r['px']:.2f} | **POC:** {r['poc_p']:.2f} | "
                        f"**Value Area:** {r['val']:.2f} - {r['vah']:.2f}  \n"
                        f"**Volumul recent se muta:** {r['shift']} fata de POC-ul ferestrei")
            if r["acc_band"]:
                st.caption(f"Cluster trend (ultimele {n_rec} zile): {r['trend_band'][0]:.2f} - {r['trend_band'][1]:.2f}")
            st.altair_chart(vp_chart(r), use_container_width=True)
            vp_notes.append(f"{COMMOD.get(t, t)}: trend {r['trend']}, pret {r['px']:.2f}, POC {r['poc_p']:.2f}, {hl}")
        st.caption("Alb = pret curent, portocaliu = POC. Profil construit din date zilnice (volumul zilei e impartit "
                   "pe intervalul low-high), deci e o aproximare. Zona de acumulare = cluster major in jumatatea "
                   "de jos a intervalului, cu zile cu variatie mica si volum peste medie; daca exista, e evidentiata "
                   "in locul clusterului de trend.")
        st.session_state["vp"] = " | ".join(vp_notes)

with tabj:
    st.subheader("1. Cat de rare sunt miscarile mari")
    if "base_stats" in st.session_state:
        st.dataframe(st.session_state["base_stats"], use_container_width=True, hide_index=True)
        st.caption("Procentul de zile-instrument (ultimii 3 ani) cu miscarea respectiva. Asta e rata de baza: "
                   "orice semnal trebuie sa o bata clar ca sa merite.")
    else:
        st.info("Ruleaza intai analiza (butonul din bara laterala).")

    st.subheader("2. Jurnal de semnale")
    journal = journal_get()
    st.caption(("Stocare: GitHub (persistent)." if _gh() else
                "Stocare: doar sesiunea curenta. Descarca CSV-ul des sau configureaza GitHub (vezi instructiunile).")
               + " Salveaza semnalele dupa inchiderea pietei US (aprox. 23:00 ora Romaniei), nu in timpul sesiunii.")
    b1, b2 = st.columns(2)
    if b1.button("Salveaza top de azi in jurnal", disabled="signals" not in st.session_state):
        s, asof, H = st.session_state["signals"], str(st.session_state["asof"]), st.session_state["horizon"]
        new = []
        for _, r in s.head(top_k).iterrows():
            fc, fp = get_flow(r["ticker"])
            new.append({"data_semnal": asof, "ticker": r["ticker"], "nume": r["nume"], "tip": r["tip"],
                        "prob": round(float(r["prob"]), 4), "orizont": H, "flow_call": fc, "flow_put": fp})
        j = pd.concat([journal, pd.DataFrame(new)]).drop_duplicates(["data_semnal", "ticker", "orizont"])
        journal_save(j)
        st.success(f"Salvat: {len(new)} semnale pentru {asof}.")
        journal = journal_get()
    if b2.button("Actualizeaza rezultatele"):
        with st.spinner("Calculez ce s-a intamplat dupa fiecare semnal..."):
            journal_save(fill_outcomes(journal))
        journal = journal_get()

    done = journal.dropna(subset=["close_close"])
    if done.empty:
        st.info("Niciun semnal rezolvat inca. Salveaza semnale azi, apoi apasa 'Actualizeaza' dupa urmatoarea zi de tranzactionare.")
    else:
        st.write(f"Semnale rezolvate: **{len(done)}** din {len(journal)}")
        summ = {"Maxim >= 3% peste deschidere": (done["open_high"] >= 0.03).mean(),
                "Maxim >= 6% peste deschidere": (done["open_high"] >= 0.06).mean(),
                "Maxim >= 10% peste deschidere": (done["open_high"] >= 0.10).mean(),
                "Inchidere > ziua anterioara": (done["close_close"] > 0).mean()}
        tbl = pd.DataFrame({"Rata": {k: f"{v:.1%}" for k, v in summ.items()}})
        tbl.loc["Medie gap la deschidere"] = f"{done['gap'].mean():+.2%}"
        tbl.loc["Medie deschidere -> inchidere"] = f"{done['open_close'].mean():+.2%}"
        tbl.loc["Medie inchidere vs ziua anterioara"] = f"{done['close_close'].mean():+.2%}"
        st.table(tbl)
        hdone = journal.dropna(subset=["ret_H"])
        if len(hdone):
            st.write(f"Dupa perioada de prognoza: pozitiv in **{(hdone['ret_H'] > 0).mean():.0%}** din "
                     f"{len(hdone)} cazuri, medie {hdone['ret_H'].mean():+.2%}.")
        st.caption("Compara cu rata de baza de mai sus. Sub ~30-50 de semnale rezolvate, rezultatele sunt zgomot.")

    st.dataframe(journal.sort_values("data_semnal", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("Descarca jurnal (CSV)", journal.to_csv(index=False), "journal.csv", "text/csv")
    up = st.file_uploader("Importa jurnal (CSV) - se combina cu cel curent", type="csv")
    if up is not None and st.button("Importa"):
        j = pd.concat([journal, _clean(pd.read_csv(up))]).drop_duplicates(["data_semnal", "ticker", "orizont"])
        journal_save(j)
        st.success("Importat.")

with tab2:
    if "res" not in st.session_state:
        st.info("Ruleaza intai analiza.")
    else:
        ms, mb = st.session_state["m"]
        a, b = st.columns(2)
        a.metric("Rata de succes a picks-urilor", f"{st.session_state['hit']:.1%}")
        b.metric("Rata de baza (toate instrumentele)", f"{st.session_state['base']:.1%}")
        c1, c2 = st.columns(2)
        c1.subheader("Strategie (net de costuri)"); c1.table(pd.Series(ms, name="").to_frame())
        c2.subheader("Benchmark echipondere"); c2.table(pd.Series(mb, name="").to_frame())
        st.subheader("Calibrare: cat de adevarat e scorul")
        st.caption("Pe perioada de test: la fiecare nivel de scor, in cate cazuri a crescut efectiv pretul "
                   "(hit) si cu cat (media / mediana). Daca hit nu creste odata cu scorul, scorul nu valoreaza nimic.")
        st.dataframe(st.session_state["calib"].round(3), use_container_width=True)
        st.line_chart((1 + st.session_state["res"]).cumprod())
        st.warning("Lista curenta de actiuni favorizeaza castigatorii recenti (survivorship bias), iar futures "
                   "Yahoo sunt contracte continue cu salturi la rulare. Rezultatele sunt optimiste.")

with tab3:
    if not has_secret("ANTHROPIC_API_KEY"):
        st.info("Adauga ANTHROPIC_API_KEY in Secrets ca sa activezi asistentul.")
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        chat = st.session_state.setdefault("chat", [])
        for m in chat:
            st.chat_message(m["role"]).write(m["content"])
        if q := st.chat_input("Intrebarea ta..."):
            chat.append({"role": "user", "content": q})
            st.chat_message("user").write(q)
            system = ("Esti asistentul unei aplicatii educationale de trading cantitativ. Raspunde in romana, "
                      "onest, fara promisiuni de profit; mentioneaza riscurile si limitele backtestului. "
                      "Rezultate curente: " + st.session_state.get("summary", "inca nerulat")
                      + " Profil volum: " + st.session_state.get("vp", "indisponibil"))
            r = client.messages.create(model="claude-sonnet-5-5", max_tokens=800, system=system, messages=chat)
            ans = r.content[0].text
            chat.append({"role": "assistant", "content": ans})
            st.chat_message("assistant").write(ans)
