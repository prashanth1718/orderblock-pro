import gzip
import io
import time
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="OrderBlock Pro V2", page_icon="📊", layout="wide")
st.title("📊 OrderBlock Pro V2 — NSE F&O Scanner")
st.caption("Live underlying-stock research scanner | 5-minute & 15-minute | No order placement")

MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
BASE = "https://api.upstox.com/v3"

def get_token():
    try:
        return st.secrets["upstox"]["access_token"]
    except Exception:
        return None

def headers(token):
    return {"Accept": "application/json", "Authorization": f"Bearer {token}"}

@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_fno_universe():
    """Load current Upstox instrument master; use equity underlyings with NSE stock futures."""
    r = requests.get(MASTER_URL, timeout=45)
    r.raise_for_status()
    raw = r.content
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    data = pd.read_json(io.BytesIO(raw))
    required = {"segment", "instrument_type", "underlying_type", "underlying_key", "underlying_symbol"}
    if not required.issubset(data.columns):
        raise RuntimeError("Upstox instrument master format changed; required fields were not found.")
    fut = data[
        (data["segment"] == "NSE_FO")
        & (data["instrument_type"] == "FUT")
        & (data["underlying_type"] == "EQUITY")
    ].copy()
    fut = fut.dropna(subset=["underlying_key", "underlying_symbol"])
    universe = (
        fut.sort_values("expiry")
        .drop_duplicates("underlying_key")
        [["underlying_symbol", "underlying_key"]]
        .sort_values("underlying_symbol")
        .reset_index(drop=True)
    )
    if universe.empty:
        raise RuntimeError("No NSE equity stock-futures underlyings found in the current instrument master.")
    return universe

def fetch_candles(key, minutes, token):
    url = f"{BASE}/historical-candle/intraday/{quote(key, safe='')}/minutes/{minutes}"
    r = requests.get(url, headers=headers(token), timeout=20)
    if r.status_code == 401:
        raise RuntimeError("Upstox token expired or rejected (401). Generate a fresh token and update Streamlit Secrets.")
    r.raise_for_status()
    candles = r.json().get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    cols = ["datetime", "open", "high", "low", "close", "volume", "oi"][:len(candles[0])]
    df = pd.DataFrame(candles, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df.dropna(subset=["datetime", "close"]).sort_values("datetime").reset_index(drop=True)

def score(df, swing, vol_mult):
    if len(df) < 25:
        return None
    x = df.copy()
    x["ema20"] = x.close.ewm(span=20, adjust=False).mean()
    x["ema50"] = x.close.ewm(span=50, adjust=False).mean()
    x["vavg"] = x.volume.rolling(20).mean()
    x["vol_spike"] = x.volume >= x.vavg * vol_mult
    prior_hi = x.high.shift(1).rolling(swing).max()
    prior_lo = x.low.shift(1).rolling(swing).min()
    bull_sweep = (x.low < prior_lo) & (x.close > prior_lo)
    bear_sweep = (x.high > prior_hi) & (x.close < prior_hi)
    bull_fvg = x.low > x.high.shift(2)
    bear_fvg = x.high < x.low.shift(2)
    last = x.iloc[-1]
    bull = int(bool(bull_sweep.iloc[-1])) + int(bool(bull_fvg.iloc[-1])) + int(bool(last.vol_spike)) + int(last.close > last.ema20) + int(last.ema20 > last.ema50)
    bear = int(bool(bear_sweep.iloc[-1])) + int(bool(bear_fvg.iloc[-1])) + int(bool(last.vol_spike)) + int(last.close < last.ema20) + int(last.ema20 < last.ema50)
    return {
        "LTP": float(last.close), "Bull score": bull, "Bear score": bear,
        "Volume spike": "Yes" if bool(last.vol_spike) else "No",
        "Bull sweep": "Yes" if bool(bull_sweep.iloc[-1]) else "No",
        "Bear sweep": "Yes" if bool(bear_sweep.iloc[-1]) else "No",
        "Bull FVG": "Yes" if bool(bull_fvg.iloc[-1]) else "No",
        "Bear FVG": "Yes" if bool(bear_fvg.iloc[-1]) else "No",
        "Last candle": str(last.datetime)
    }

token = get_token()
if token:
    st.success("Upstox access token found.")
else:
    st.error("Token missing. Add [upstox] and access_token in Streamlit → App settings → Secrets.")

with st.sidebar:
    st.header("Scanner controls")
    tf = st.selectbox("Timeframe", ["5-minute", "15-minute"])
    minutes = 5 if tf == "5-minute" else 15
    limit = st.select_slider("Stocks to scan per run", options=[10, 20, 30, 50], value=20)
    threshold = st.slider("Minimum score", 2, 5, 4)
    swing = st.slider("Liquidity-sweep lookback", 3, 20, 5)
    vol_mult = st.slider("Volume spike multiple", 1.0, 3.0, 1.5, 0.1)

st.warning("V2 scans a selectable sample of NSE stock-F&O underlyings. Results are watchlist candidates only—not option-buy signals. Options chain, Greeks, futures OI confirmation and strike/premium selection are not included yet.")

if st.button("Load / refresh NSE F&O stock list"):
    try:
        u = load_stock_fno_universe()
        st.session_state["universe"] = u
        st.success(f"Loaded {len(u)} unique stock-F&O underlyings from Upstox instrument master.")
    except Exception as e:
        st.error(f"Could not load instrument list: {e}")

if "universe" not in st.session_state:
    st.info("Tap **Load / refresh NSE F&O stock list** first.")
else:
    universe = st.session_state["universe"]
    st.caption(f"Available stock-F&O underlyings: {len(universe)}. This run scans the first {min(limit, len(universe))} alphabetically; later versions will add rotating batches and prioritisation.")
    if st.button("🚀 Start live scan", type="primary", disabled=not bool(token)):
        rows, errors = [], []
        selected = universe.head(limit)
        progress = st.progress(0)
        status = st.empty()
        for i, row in selected.iterrows():
            sym, key = row["underlying_symbol"], row["underlying_key"]
            status.write(f"Scanning {sym} ({len(rows)+len(errors)+1}/{len(selected)})…")
            try:
                candles = fetch_candles(key, minutes, token)
                result = score(candles, swing, vol_mult)
                if result:
                    result["Symbol"] = sym
                    rows.append(result)
                else:
                    errors.append(f"{sym}: insufficient candles")
            except Exception as e:
                errors.append(f"{sym}: {str(e)[:120]}")
            progress.progress((len(rows) + len(errors)) / len(selected))
            time.sleep(0.12)
        status.write("Scan complete.")
        st.session_state["scan_rows"] = rows
        st.session_state["scan_errors"] = errors
        st.session_state["scan_tf"] = tf

    if "scan_rows" in st.session_state:
        results = pd.DataFrame(st.session_state["scan_rows"])
        if not results.empty:
            st.subheader(f"Results — {st.session_state.get('scan_tf', tf)}")
            bull = results[(results["Bull score"] >= threshold) & (results["Bull score"] > results["Bear score"])].sort_values(["Bull score", "LTP"], ascending=[False, True])
            bear = results[(results["Bear score"] >= threshold) & (results["Bear score"] > results["Bull score"])].sort_values(["Bear score", "LTP"], ascending=[False, True])
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 🟢 Bullish watchlist")
                st.dataframe(bull.head(5), use_container_width=True, hide_index=True)
            with c2:
                st.markdown("### 🔴 Bearish watchlist")
                st.dataframe(bear.head(5), use_container_width=True, hide_index=True)
            st.markdown("### All scanned stocks")
            st.dataframe(results.sort_values(["Bull score", "Bear score"], ascending=False), use_container_width=True, hide_index=True)
            st.download_button("Download scan CSV", results.to_csv(index=False), "orderblock_pro_scan.csv", "text/csv")
        else:
            st.warning("No usable candle data returned for this batch.")
        if st.session_state.get("scan_errors"):
            with st.expander(f"Skipped / errors ({len(st.session_state['scan_errors'])})"):
                st.write(st.session_state["scan_errors"])

st.info("Signal logic is a simplified technical heuristic: EMA20/EMA50 trend, relative volume, recent swing sweep and three-candle FVG. It does not establish institutional activity or a verified win rate. No trades are placed.")
st.caption("Options trading involves substantial risk. Confirm price, liquidity, spread, expiry and risk independently before any trade.")
