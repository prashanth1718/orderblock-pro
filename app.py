import streamlit as st
import pandas as pd
import numpy as np
import requests
from urllib.parse import quote

st.set_page_config(page_title="OrderBlock Pro", page_icon="📊", layout="wide")
st.title("📊 OrderBlock Pro — Upstox Live Connection Test")
st.caption("NSE F&O research dashboard | 5-minute and 15-minute candle analysis")

def get_token():
    try:
        return st.secrets["upstox"]["access_token"]
    except Exception:
        return None

def fetch_intraday(instrument_key: str, interval: int, token: str) -> pd.DataFrame:
    encoded_key = quote(instrument_key, safe="")
    url = f"https://api.upstox.com/v3/historical-candle/intraday/{encoded_key}/minutes/{interval}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    response = requests.get(url, headers=headers, timeout=20)
    if response.status_code == 401:
        raise RuntimeError("Upstox rejected the token (401). Generate a fresh access token and update Streamlit Secrets.")
    if response.status_code == 403:
        raise RuntimeError("Upstox denied access (403). Check API app permissions and market-data access.")
    response.raise_for_status()
    payload = response.json()
    candles = payload.get("data", {}).get("candles", [])
    if not candles:
        raise RuntimeError("No intraday candles returned. Check the instrument key, market session, and API access.")
    df = pd.DataFrame(candles, columns=["datetime", "open", "high", "low", "close", "volume", "oi"][:len(candles[0])])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df.sort_values("datetime").reset_index(drop=True)

def analyze(df: pd.DataFrame, swing: int, volume_multiple: float) -> pd.DataFrame:
    df = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
    if len(df) < max(25, swing + 3):
        raise ValueError("Not enough candles yet for the indicator set.")
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["vol_avg20"] = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] >= df["vol_avg20"] * volume_multiple
    df["prior_high"] = df["high"].shift(1).rolling(swing).max()
    df["prior_low"] = df["low"].shift(1).rolling(swing).min()
    df["bull_sweep"] = (df["low"] < df["prior_low"]) & (df["close"] > df["prior_low"])
    df["bear_sweep"] = (df["high"] > df["prior_high"]) & (df["close"] < df["prior_high"])
    df["bull_fvg"] = df["low"] > df["high"].shift(2)
    df["bear_fvg"] = df["high"] < df["low"].shift(2)
    df["bull_score"] = (
        df["bull_sweep"].fillna(False).astype(int)
        + df["bull_fvg"].fillna(False).astype(int)
        + df["vol_spike"].fillna(False).astype(int)
        + (df["close"] > df["ema20"]).fillna(False).astype(int)
        + (df["ema20"] > df["ema50"]).fillna(False).astype(int)
    )
    df["bear_score"] = (
        df["bear_sweep"].fillna(False).astype(int)
        + df["bear_fvg"].fillna(False).astype(int)
        + df["vol_spike"].fillna(False).astype(int)
        + (df["close"] < df["ema20"]).fillna(False).astype(int)
        + (df["ema20"] < df["ema50"]).fillna(False).astype(int)
    )
    return df

token = get_token()
if token:
    st.success("Upstox access token found in Streamlit Secrets.")
else:
    st.error("Upstox token not found. In Streamlit App settings → Secrets, add [upstox] and access_token.")

with st.sidebar:
    st.header("Scanner settings")
    timeframe = st.selectbox("Analysis timeframe", ["5-minute", "15-minute"])
    interval = 5 if timeframe == "5-minute" else 15
    swing = st.slider("Swing lookback", 3, 20, 5)
    vol_mult = st.slider("Volume spike threshold", 1.0, 3.0, 1.5, 0.1)

st.subheader("Live candle test")
st.write("Enter a valid Upstox instrument key. Example: `NSE_EQ|INE848E01016` (NHPC).")
instrument_key = st.text_input("Upstox instrument key", value="NSE_EQ|INE848E01016")
if st.button("Fetch live intraday candles", type="primary", disabled=not bool(token)):
    try:
        with st.spinner("Requesting candles from Upstox..."):
            candles = fetch_intraday(instrument_key.strip(), interval, token)
        st.success(f"Received {len(candles)} candles from Upstox.")
        analyzed = analyze(candles, swing, vol_mult)
        last = analyzed.iloc[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest close", f"{last.close:,.2f}")
        m2.metric("Bullish score", f"{int(last.bull_score)}/5")
        m3.metric("Bearish score", f"{int(last.bear_score)}/5")
        m4.metric("Volume spike", "Yes" if bool(last.vol_spike) else "No")
        if last.bull_score >= 4 and last.bull_score > last.bear_score:
            st.success("Bullish setup watch — confirmation required.")
        elif last.bear_score >= 4 and last.bear_score > last.bull_score:
            st.error("Bearish setup watch — confirmation required.")
        else:
            st.warning("No high-confluence setup on the latest candle.")
        display_cols = ["datetime", "open", "high", "low", "close", "volume",
                        "ema20", "ema50", "bull_sweep", "bear_sweep",
                        "bull_fvg", "bear_fvg", "bull_score", "bear_score"]
        st.dataframe(analyzed[display_cols].tail(30), use_container_width=True)
    except requests.HTTPError as exc:
        st.error(f"Upstox HTTP error: {exc}. Check token, instrument key, and API access.")
        if exc.response is not None:
            st.code(exc.response.text[:1000])
    except Exception as exc:
        st.error(str(exc))

st.info("This is a single-instrument live-data connection test, not yet the full NSE F&O scanner. No orders are placed. FVG and sweep labels are simplified research heuristics, not proof of institutional activity.")
st.caption("Options trading involves substantial risk. Verify all data and signals independently.")
