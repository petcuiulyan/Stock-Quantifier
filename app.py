import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier

st.set_page_config(page_title="Quant Lab", layout="wide")
st.title("Quant Lab - model de tranzactionare US (educational)")
st.caption("Proiect de invatare. Nu este sfat financiar. Rezultatele din trecut nu garanteaza nimic.")

FEATS = ["ret_1", "ret_5", "ret_20", "vol_20", "gap", "range", "vol_ratio", "sma_ratio", "rsi"]
DEFAULT = "AAPL MSFT NVDA AMZN GOOGL META TSLA AMD NFLX JPM V UNH XOM HD COST AVGO ORCL CRM ADBE INTC"

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Setari")
    tickers = st.text_area("Tickere (separate prin spatiu)", DEFAULT).upper().split()
    start = st.date_input("Date de la", pd.Timestamp("2012-01-01"))
    split = st.date_input("Data de separare train/test", pd.Timestamp("2021-01-01"))
    top_k = st.slider("Cate actiuni pe zi (top K)", 1, 10, 3)
    cost_bps = st.slider("Cost per tranzactie (bps)", 0, 30, 5)
    run = st.button("Ruleaza", type="primary")


@st.cache_data(show_spinner="Descarc datele...", ttl=3600)
def load(tickers, start):
    return yf.download(list(tickers), start=str(start), auto_adjust=True, progress=False)


def features(raw, t):
    d = pd.DataFrame({"open": raw["Open"][t], "high": raw["High"][t], "low": raw["Low"][t],
                      "close": raw["Close"][t], "volume": raw["Volume"][t]}).dropna()
    r = d["close"].pct_change()
    f = pd.DataFrame(index=d.index)
    f["ret_1"], f["ret_5"], f["ret_20"] = r, d["close"].pct_change(5), d["close"].pct_change(20)
    f["vol_20"] = r.rolling(20).std()
    f["gap"] = d["open"] / d["close"].shift(1) - 1
    f["range"] = (d["high"] - d["low"]) / d["close"]
    f["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()
    f["sma_ratio"] = d["close"] / d["close"].rolling(50).mean() - 1
    delta = d["close"].diff()
    f["rsi"] = 100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean()
                            / (-delta.clip(upper=0)).rolling(14).mean())
    f["next_ret"] = r.shift(-1)
    f["ticker"] = t
    return f.dropna(subset=FEATS)


def make_model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=0)


def metrics(s):
    curve = (1 + s).cumprod()
    yrs = len(s) / 252
    return {"Total": f"x{curve.iloc[-1]:.2f}", "CAGR": f"{curve.iloc[-1] ** (1 / yrs) - 1:.1%}",
            "Sharpe": f"{s.mean() / s.std() * np.sqrt(252):.2f}",
            "Max drawdown": f"{(curve / curve.cummax() - 1).min():.1%}"}


if run:
    raw = load(tuple(tickers), start)
    ok = [t for t in tickers if t in raw["Close"].columns and raw["Close"][t].notna().sum() > 300]
    data = pd.concat([features(raw, t) for t in ok]).reset_index().rename(columns={"index": "date", "Date": "date"})
    labeled = data.dropna(subset=["next_ret"]).copy()
    labeled["target"] = (labeled["next_ret"] > 0).astype(int)

    train = labeled[labeled["date"] < pd.Timestamp(split)]
    test = labeled[labeled["date"] >= pd.Timestamp(split)].copy()
    model = make_model().fit(train[FEATS], train["target"])
    test["prob"] = model.predict_proba(test[FEATS])[:, 1]

    rows = []
    for date, g in test.groupby("date"):
        net = g.nlargest(top_k, "prob")["next_ret"].mean() - 2 * cost_bps / 10000
        rows.append((date, net, g["next_ret"].mean()))
    res = pd.DataFrame(rows, columns=["date", "Strategie", "Benchmark"]).set_index("date")

    # semnale pentru ultima zi, model antrenat pe tot istoricul
    final = make_model().fit(labeled[FEATS], labeled["target"])
    last = data[data["date"] == data["date"].max()].copy()
    last["prob_crestere"] = final.predict_proba(last[FEATS])[:, 1]
    signals = last.sort_values("prob_crestere", ascending=False)[["ticker", "prob_crestere", "rsi", "gap"]]

    acc = ((test["prob"] > 0.5).astype(int) == (test["next_ret"] > 0).astype(int)).mean()
    ms, mb = metrics(res["Strategie"]), metrics(res["Benchmark"])
    st.session_state["summary"] = (
        f"Tickere: {len(ok)}, test de la {split}, top K={top_k}, cost={cost_bps} bps. "
        f"Acuratete directie: {acc:.3f}. Strategie: {ms}. Benchmark: {mb}. "
        f"Top semnale ultima zi ({data['date'].max().date()}): "
        f"{signals.head(5)[['ticker', 'prob_crestere']].round(3).to_dict('records')}")
    st.session_state["res"], st.session_state["signals"] = res, signals
    st.session_state["m"], st.session_state["acc"] = (ms, mb), acc

tab1, tab2, tab3 = st.tabs(["Backtest", "Semnale ultima zi", "Asistent AI"])

with tab1:
    if "res" not in st.session_state:
        st.info("Apasa 'Ruleaza' in bara laterala.")
    else:
        ms, mb = st.session_state["m"]
        st.metric("Acuratete directie (50% = moneda)", f"{st.session_state['acc']:.1%}")
        c1, c2 = st.columns(2)
        c1.subheader("Strategie (net de costuri)"); c1.table(pd.Series(ms, name="").to_frame())
        c2.subheader("Benchmark echipondere"); c2.table(pd.Series(mb, name="").to_frame())
        st.line_chart((1 + st.session_state["res"]).cumprod())
        st.warning("Lista de actiuni contine castigatori cunoscuti (survivorship bias): rezultatul e optimist.")

with tab2:
    if "signals" in st.session_state:
        st.dataframe(st.session_state["signals"], use_container_width=True)
        st.caption("Probabilitatea estimata de model ca actiunea sa creasca in ziua urmatoare. Nu e certitudine.")
    else:
        st.info("Ruleaza intai modelul.")

with tab3:
    st.write("Intreaba despre rezultate, riscuri sau cum sa imbunatatesti modelul.")
    if "ANTHROPIC_API_KEY" not in st.secrets:
        st.info("Adauga ANTHROPIC_API_KEY in Secrets (Streamlit Cloud) ca sa activezi asistentul.")
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
                      "onest, fara promisiuni de profit, mentioneaza riscurile si limitele backtestului. "
                      "Rezultatele curente: " + st.session_state.get("summary", "inca nerulat"))
            r = client.messages.create(model="claude-sonnet-5-5", max_tokens=800, system=system, messages=chat)
            ans = r.content[0].text
            chat.append({"role": "assistant", "content": ans})
            st.chat_message("assistant").write(ans)
