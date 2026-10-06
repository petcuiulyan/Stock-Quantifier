import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier

st.set_page_config(page_title="Quant Lab", layout="wide")
st.title("Quant Lab - top candidati LONG (NASDAQ + materii prime)")
st.caption("Proiect educational. Nu este sfat financiar. Probabilitatile sunt estimari, nu certitudini.")

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
    extra = st.text_input("Tickere extra (separate prin spatiu)")
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

tab1, tab2, tab3 = st.tabs(["Top candidati LONG", "Backtest", "Asistent AI"])

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
    if "ANTHROPIC_API_KEY" not in st.secrets:
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
                      "Rezultate curente: " + st.session_state.get("summary", "inca nerulat"))
            r = client.messages.create(model="claude-sonnet-5-5", max_tokens=800, system=system, messages=chat)
            ans = r.content[0].text
            chat.append({"role": "assistant", "content": ans})
            st.chat_message("assistant").write(ans)
