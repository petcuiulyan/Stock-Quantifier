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