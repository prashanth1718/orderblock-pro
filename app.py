import gzip
import io
import time
from datetime import datetime, time as dtime
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="OrderBlock Pro V3.1", page_icon="📊", layout="wide")
st.title("📊 OrderBlock Pro V3.1 — Full NSE F&O Scanner")
st.caption("5-minute + 15-minute confirmation | Expandable scan up to all loaded stock-F&O underlyings | No order placement")

IST = ZoneInfo("Asia/Kolkata")
MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
BASE = "https://api.upstox.com/v3"

def get_token():
    try:
        return st.secrets["upstox"]["access_token"]
    except Exception:
        return None

def auth_headers(token):
    return {"Accept": "application/json", "Authorization": f"Bearer {token}"}

@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_fno_universe():
    response = requests.get(MASTER_URL, timeout=45)
    response.raise_for_status()
    raw = response.content
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    master = pd.read_json(io.BytesIO(raw))
    needed = {"segment", "instrument_type", "underlying_type", "underlying_key", "underlying_symbol"}
    if not needed.issubset(master.columns):
        raise RuntimeError("Instrument master columns changed; cannot safely build the F&O universe.")
    futures = master[
        (master["segment"] == "NSE_FO")
        & (master["instrument_type"] == "FUT")
        & (master["underlying_type"] == "EQUITY")
    ].dropna(subset=["underlying_key", "underlying_symbol"])
    universe = futures[["underlying_symbol", "underlying_key"]].drop_duplicates()
    return universe.sort_values("underlying_symbol").reset_index(drop=True)

def fetch_candles(key, minutes, token):
    url = f"{BASE}/historical-candle/intraday/{quote(key, safe='')}/minutes/{minutes}"
    response = requests.get(url, headers=auth_headers(token), timeout=25)
    if response.status_code == 401:
        raise RuntimeError("Upstox token rejected/expired (401). Update Streamlit Secrets with a fresh token.")
    if response.status_code == 429:
        raise RuntimeError("Upstox rate limit reached (429). Stop this scan and retry later with a smaller batch.")
    response.raise_for_status()
    candles = response.json().get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    columns = ["datetime", "open", "high", "low", "close", "volume", "oi"][:len(candles[0])]
    frame = pd.DataFrame(candles, columns=columns)
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    return frame.dropna(subset=["datetime", "close"]).sort_values("datetime").reset_index(drop=True)

def analyse(frame, swing_lookback, volume_multiple):
    if len(frame) < 25:
        return None
    x = frame.copy()
    x["ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    x["vol_avg"] = x["volume"].rolling(20).mean()
    x["vol_spike"] = x["volume"] >= x["vol_avg"] * volume_multiple
    prior_high = x["high"].shift(1).rolling(swing_lookback).max()
    prior_low = x["low"].shift(1).rolling(swing_lookback).min()
    bull_sweep = (x["low"] < prior_low) & (x["close"] > prior_low)
    bear_sweep = (x["high"] > prior_high) & (x["close"] < prior_high)
    bull_fvg = x["low"] > x["high"].shift(2)
    bear_fvg = x["high"] < x["low"].shift(2)
    last = x.iloc[-1]
    bull = int(bool(bull_sweep.iloc[-1])) + int(bool(bull_fvg.iloc[-1])) + int(bool(last["vol_spike"])) + int(last["close"] > last["ema20"]) + int(last["ema20"] > last["ema50"])
    bear = int(bool(bear_sweep.iloc[-1])) + int(bool(bear_fvg.iloc[-1])) + int(bool(last["vol_spike"])) + int(last["close"] < last["ema20"]) + int(last["ema20"] < last["ema50"])
    return {
        "close": float(last["close"]), "bull": bull, "bear": bear,
        "volume_spike": bool(last["vol_spike"]),
        "bull_sweep": bool(bull_sweep.iloc[-1]), "bear_sweep": bool(bear_sweep.iloc[-1]),
        "candle_time": last["datetime"],
    }

def market_status(now):
    if now.weekday() >= 5:
        return "Market closed (weekend)"
    if dtime(9, 15) <= now.time() <= dtime(15, 30):
        return "Regular NSE session"
    return "Market closed / outside regular session"

token = get_token()
if token:
    st.success("Upstox access token found in Streamlit Secrets.")
else:
    st.error("Token missing. Add [upstox] and access_token in Streamlit → App settings → Secrets.")

now = datetime.now(IST)
st.info(f"India time: {now:%d %b %Y, %I:%M:%S %p} | {market_status(now)}")

with st.sidebar:
    st.header("Scanner controls")
    scan_size = st.select_slider(
        "Stocks per scan",
        options=[20, 50, 100, 150, 218],
        value=20,
        format_func=lambda n: "All loaded stocks" if n == 218 else str(n)
    )
    min_score = st.slider("Minimum score on each timeframe", 2, 5, 3)
    swing = st.slider("Liquidity-sweep lookback", 3, 20, 5)
    vol_mult = st.slider("Volume spike multiple", 1.0, 3.0, 1.5, 0.1)

st.warning("A full 218-stock scan makes up to 436 candle requests (one 5m and one 15m request per stock). It can take several minutes and may be affected by API limits, network interruptions or token expiry. Start with 20 or 50 first. Results are underlying-stock watchlist candidates only; options chain, Greeks, futures OI confirmation and strike/premium selection are not included.")

if st.button("Load / refresh NSE F&O stock list"):
    try:
        universe = load_stock_fno_universe()
        st.session_state["universe"] = universe
        st.success(f"Loaded {len(universe)} unique NSE stock-F&O underlyings.")
    except Exception as exc:
        st.error(f"Could not load instrument list: {exc}")

if "universe" not in st.session_state:
    st.info("Load the NSE F&O stock list first.")
else:
    universe = st.session_state["universe"]
    available = len(universe)
    count_to_scan = min(scan_size, available)
    st.caption(f"Loaded universe: {available} stocks. This run will scan {count_to_scan}.")
    if scan_size == 218 and available < 218:
        st.warning(f"The current instrument master contains {available} stocks, so the scan will use all {available} available stocks.")
    if st.button("🚀 Run 5m + 15m scan", type="primary", disabled=not bool(token)):
        records, errors = [], []
        chosen = universe.head(count_to_scan)
        progress = st.progress(0)
        status = st.empty()
        for count, (_, item) in enumerate(chosen.iterrows(), start=1):
            symbol, key = item["underlying_symbol"], item["underlying_key"]
            status.write(f"Scanning {symbol}: 5m + 15m ({count}/{count_to_scan})")
            try:
                five = analyse(fetch_candles(key, 5, token), swing, vol_mult)
                fifteen = analyse(fetch_candles(key, 15, token), swing, vol_mult)
                if five is None or fifteen is None:
                    errors.append(f"{symbol}: insufficient 5m or 15m candles")
                else:
                    records.append({
                        "Symbol": symbol, "LTP": round(five["close"], 2),
                        "5m Bull": five["bull"], "15m Bull": fifteen["bull"],
                        "5m Bear": five["bear"], "15m Bear": fifteen["bear"],
                        "Bull total": five["bull"] + fifteen["bull"],
                        "Bear total": five["bear"] + fifteen["bear"],
                        "5m volume spike": "Yes" if five["volume_spike"] else "No",
                        "15m volume spike": "Yes" if fifteen["volume_spike"] else "No",
                        "5m bull sweep": "Yes" if five["bull_sweep"] else "No",
                        "15m bull sweep": "Yes" if fifteen["bull_sweep"] else "No",
                        "5m bear sweep": "Yes" if five["bear_sweep"] else "No",
                        "15m bear sweep": "Yes" if fifteen["bear_sweep"] else "No",
                        "5m candle time": str(five["candle_time"]),
                        "15m candle time": str(fifteen["candle_time"]),
                    })
            except Exception as exc:
                errors.append(f"{symbol}: {str(exc)[:140]}")
                if "429" in str(exc):
                    errors.append("Scan stopped early due to Upstox rate limiting.")
                    break
            progress.progress(count / count_to_scan)
            time.sleep(0.20)
        status.write("Scan finished. Review skipped stocks/errors below.")
        st.session_state["v31_records"] = records
        st.session_state["v31_errors"] = errors
        st.session_state["v31_scan_time"] = datetime.now(IST).strftime("%d %b %Y %I:%M:%S %p")
        st.session_state["v31_requested"] = count_to_scan

    if "v31_records" in st.session_state:
        data = pd.DataFrame(st.session_state["v31_records"])
        st.subheader(f"Scan results | {st.session_state['v31_scan_time']}")
        st.caption(f"Requested: {st.session_state.get('v31_requested', 0)} | Usable results: {len(data)}")
        if not data.empty:
            bullish = data[
                (data["5m Bull"] >= min_score) & (data["15m Bull"] >= min_score)
                & (data["Bull total"] > data["Bear total"])
            ].sort_values("Bull total", ascending=False).head(5)
            bearish = data[
                (data["5m Bear"] >= min_score) & (data["15m Bear"] >= min_score)
                & (data["Bear total"] > data["Bull total"])
            ].sort_values("Bear total", ascending=False).head(5)
            left, right = st.columns(2)
            with left:
                st.markdown("### 🟢 Bullish 5m + 15m")
                st.dataframe(bullish, use_container_width=True, hide_index=True)
            with right:
                st.markdown("### 🔴 Bearish 5m + 15m")
                st.dataframe(bearish, use_container_width=True, hide_index=True)
            st.markdown("### All usable scanned stocks")
            st.dataframe(data.sort_values(["Bull total", "Bear total"], ascending=False), use_container_width=True, hide_index=True)
            st.download_button("Download scan CSV", data.to_csv(index=False), "orderblock_pro_full_scan.csv", "text/csv")
            if bullish.empty and bearish.empty:
                st.info("No stocks met the selected threshold on both timeframes. Do not force signals by lowering filters.")
        else:
            st.warning("No usable stock results were returned.")
        if st.session_state.get("v31_errors"):
            with st.expander(f"Skipped stocks / errors ({len(st.session_state['v31_errors'])})"):
                st.write(st.session_state["v31_errors"])

st.caption("Technical heuristic only: EMA20/EMA50, relative volume, swing-sweep and three-candle FVG checks. It does not prove institutional activity or a profitable edge. No trades are placed.")
