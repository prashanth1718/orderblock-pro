import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="OrderBlock Pro", page_icon="📊", layout="wide")

st.title("📊 OrderBlock Pro — Free Mobile Edition")
st.caption("NSE F&O research dashboard | 5-minute and 15-minute analysis")

st.info(
    "Starter build: upload OHLCV candle data as CSV to scan it. "
    "Live Upstox OAuth/data connection will be configured after this app is deployed "
    "and its public HTTPS address is known. No trades are placed."
)

with st.sidebar:
    st.header("Scanner settings")
    timeframe = st.selectbox("Analysis timeframe", ["5-minute", "15-minute"])
    lookback = st.slider("Swing lookback", 3, 20, 5)
    volume_multiple = st.slider("Volume spike threshold", 1.0, 3.0, 1.5, 0.1)
    st.markdown("---")
    st.write("Expected CSV columns:")
    st.code("datetime, open, high, low, close, volume")

uploaded = st.file_uploader("Upload candle CSV", type=["csv"])

def analyze(df, swing):
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    df.columns = [c.lower().strip() for c in df.columns]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
    if len(df) < max(25, swing + 3):
        raise ValueError("Please upload at least 25 valid candles.")
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["vol_avg20"] = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] >= df["vol_avg20"] * volume_multiple
    df["prior_high"] = df["high"].shift(1).rolling(swing).max()
    df["prior_low"] = df["low"].shift(1).rolling(swing).min()
    df["bull_sweep"] = (df["low"] < df["prior_low"]) & (df["close"] > df["prior_low"])
    df["bear_sweep"] = (df["high"] > df["prior_high"]) & (df["close"] < df["prior_high"])
    # Simple three-candle FVG definitions; these are heuristic zones, not institutional order-flow proof.
    df["bull_fvg"] = df["low"] > df["high"].shift(2)
    df["bear_fvg"] = df["high"] < df["low"].shift(2)
    df["bull_bias"] = df["close"] > df["ema20"]
    df["bear_bias"] = df["close"] < df["ema20"]
    df["bull_score"] = (
        df["bull_sweep"].fillna(False).astype(int)
        + df["bull_fvg"].fillna(False).astype(int)
        + df["vol_spike"].fillna(False).astype(int)
        + df["bull_bias"].fillna(False).astype(int)
        + (df["ema20"] > df["ema50"]).fillna(False).astype(int)
    )
    df["bear_score"] = (
        df["bear_sweep"].fillna(False).astype(int)
        + df["bear_fvg"].fillna(False).astype(int)
        + df["vol_spike"].fillna(False).astype(int)
        + df["bear_bias"].fillna(False).astype(int)
        + (df["ema20"] < df["ema50"]).fillna(False).astype(int)
    )
    return df

if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        result = analyze(raw, lookback)
        last = result.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest close", f"{last.close:,.2f}")
        c2.metric("Bullish score", f"{int(last.bull_score)}/5")
        c3.metric("Bearish score", f"{int(last.bear_score)}/5")
        c4.metric("Volume spike", "Yes" if bool(last.vol_spike) else "No")

        if last.bull_score >= 4 and last.bull_score > last.bear_score:
            st.success("Bullish setup watch — confirm price action before considering a CE.")
        elif last.bear_score >= 4 and last.bear_score > last.bull_score:
            st.error("Bearish setup watch — confirm price action before considering a PE.")
        else:
            st.warning("No high-confluence setup on the latest candle. Wait for confirmation.")

        st.subheader("Recent candle analysis")
        cols = ["open", "high", "low", "close", "volume", "ema20", "ema50",
                "bull_sweep", "bear_sweep", "bull_fvg", "bear_fvg",
                "bull_score", "bear_score"]
        st.dataframe(result[cols].tail(30), use_container_width=True)

        st.download_button(
            "Download analyzed candles",
            result.to_csv(index=False).encode("utf-8"),
            file_name="orderblock_analysis.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(str(e))
else:
    st.subheader("What this starter can scan")
    a, b = st.columns(2)
    with a:
        st.markdown("**Bullish watch conditions**")
        st.write("- Sweep below a recent swing low and close back above")
        st.write("- Bullish three-candle fair-value-gap heuristic")
        st.write("- Price above EMA20 and EMA20 above EMA50")
        st.write("- Relative volume expansion")
    with b:
        st.markdown("**Bearish watch conditions**")
        st.write("- Sweep above a recent swing high and close back below")
        st.write("- Bearish three-candle fair-value-gap heuristic")
        st.write("- Price below EMA20 and EMA20 below EMA50")
        st.write("- Relative volume expansion")
    st.warning(
        "These are rule-based research signals, not verified institutional order blocks "
        "or proven profitable trades. Options strike, Greeks, OI and live quotes are not "
        "connected in this starter."
    )

st.markdown("---")
st.caption("Educational/research tool only. Verify signals independently. Options involve substantial risk.")
