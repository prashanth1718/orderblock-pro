import gzip
import io
import time
from datetime import datetime, time as dtime
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="OrderBlock Pro V4.3", page_icon="📊", layout="wide")
st.title("📊 OrderBlock Pro V4.3 — NSE F&O Trade Selection")
st.caption("5m + 15m stock scan | ATM / ITM / OTM option-chain view | Greeks, OI and liquidity checks | No order placement")

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
    # Use multi-day historical candles so the 15-minute timeframe has enough
    # bars early in the trading session (intraday endpoint returns today only).
    to_date = datetime.now(IST).date()
    from_date = to_date - pd.Timedelta(days=10)
    encoded_key = quote(key, safe="")
    url = f"{BASE}/historical-candle/{encoded_key}/minutes/{minutes}/{to_date}/{from_date}"
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
    return (frame.dropna(subset=["datetime", "close"])
            .drop_duplicates(subset=["datetime"])
            .sort_values("datetime").reset_index(drop=True))


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

def raise_upstox_api_error(response, endpoint_name):
    """Show the API's response detail without ever exposing request credentials."""
    try:
        payload = response.json()
        detail = str(payload)[:700]
    except Exception:
        detail = (response.text or "No response body")[:700]
    if response.status_code == 429:
        raise RuntimeError(f"Upstox {endpoint_name} returned HTTP 429 (rate limit): {detail}")
    if response.status_code == 401:
        raise RuntimeError(f"Upstox {endpoint_name} returned HTTP 401 (authentication/permission response): {detail}")
    raise RuntimeError(f"Upstox {endpoint_name} returned HTTP {response.status_code}: {detail}")

def get_option_contracts(underlying_key, token):
    url = "https://api.upstox.com/v2/option/contract"
    response = requests.get(url, params={"instrument_key": underlying_key}, headers=auth_headers(token), timeout=25)
    if not response.ok:
        raise_upstox_api_error(response, "Option Contracts API")
    return response.json().get("data", [])

def get_option_chain(underlying_key, expiry, token):
    url = "https://api.upstox.com/v2/option/chain"
    response = requests.get(url, params={"instrument_key": underlying_key, "expiry_date": expiry}, headers=auth_headers(token), timeout=25)
    if not response.ok:
        raise_upstox_api_error(response, "Option Chain API")
    return response.json().get("data", [])

def build_option_rows(symbol, underlying_key, direction, token, max_spread_pct):
    contracts = get_option_contracts(underlying_key, token)
    today = datetime.now(IST).date()
    expiries = sorted({str(c.get("expiry")) for c in contracts if c.get("expiry") and str(c.get("expiry")) >= today.isoformat()})
    if not expiries:
        return [], f"{symbol}: no future option expiry returned"
    expiry = expiries[0]
    chain = get_option_chain(underlying_key, expiry, token)
    if not chain:
        return [], f"{symbol}: empty option chain for {expiry}"
    spot = float(chain[0].get("underlying_spot_price") or 0)
    strikes = sorted({float(x.get("strike_price")) for x in chain if x.get("strike_price") is not None})
    if not spot or not strikes:
        return [], f"{symbol}: missing spot/strike data"
    atm = min(strikes, key=lambda x: abs(x-spot))
    atm_idx = strikes.index(atm)
    # Directional contract candidates: bullish uses CE; bearish uses PE.
    opt_type = "CE" if direction == "Bullish" else "PE"
    wanted = []
    for offset, label in [(0, "ATM"), (1, "1-step OTM"), (2, "2-step OTM"), (-1, "1-step ITM"), (-2, "2-step ITM")]:
        idx = atm_idx + offset if opt_type == "CE" else atm_idx - offset
        if 0 <= idx < len(strikes):
            wanted.append((strikes[idx], label))
    by_strike = {float(x["strike_price"]): x for x in chain if x.get("strike_price") is not None}
    rows = []
    for strike, moneyness in wanted:
        item = by_strike.get(strike, {})
        opt = item.get("call_options" if opt_type == "CE" else "put_options") or {}
        md = opt.get("market_data") or {}
        gr = opt.get("option_greeks") or {}
        ltp = float(md.get("ltp") or 0)
        bid = float(md.get("bid_price") or 0)
        ask = float(md.get("ask_price") or 0)
        spread = ((ask-bid)/((ask+bid)/2)*100) if bid > 0 and ask > 0 and ask >= bid else None
        liquid = ltp > 0 and bid > 0 and ask > 0 and spread is not None and spread <= max_spread_pct and float(md.get("volume") or 0) > 0
        # Reference risk levels only; not an execution instruction or tested edge.
        rows.append({
            "Symbol": symbol, "Bias": direction, "Expiry": expiry, "Option": opt_type,
            "Strike": strike, "Moneyness": moneyness, "Spot": round(spot,2),
            "LTP": round(ltp,2), "Bid": round(bid,2), "Ask": round(ask,2),
            "Spread %": round(spread,2) if spread is not None else None,
            "Volume": int(float(md.get("volume") or 0)), "OI": int(float(md.get("oi") or 0)),
            "Delta": gr.get("delta"), "Theta": gr.get("theta"), "Gamma": gr.get("gamma"),
            "Vega": gr.get("vega"), "IV": gr.get("iv"), "Liquidity": "PASS" if liquid else "REJECT",
            "Reference SL (-20%)": round(ltp*0.8,2) if ltp else None,
            "Reference T1 (+40%)": round(ltp*1.4,2) if ltp else None,
            "Reference T2 (+60%)": round(ltp*1.6,2) if ltp else None,
            "Instrument Key": opt.get("instrument_key", "")
        })
    return rows, None

def rank_option_candidates(option_rows, scan_records):
    """Transparent heuristic ranking; not a backtested or predictive model."""
    if not option_rows:
        return pd.DataFrame()
    options = pd.DataFrame(option_rows).copy()
    scan = pd.DataFrame(scan_records)
    if options.empty or scan.empty:
        return pd.DataFrame()
    scan_cols = ["Symbol", "Bull total", "Bear total"]
    options = options.merge(scan[scan_cols], on="Symbol", how="left")
    options["Underlying strength"] = options.apply(
        lambda r: float(r.get("Bull total", 0) if r.get("Bias") == "Bullish" else r.get("Bear total", 0)), axis=1
    )
    for col in ["Volume", "OI", "Spread %", "Delta"]:
        options[col] = pd.to_numeric(options[col], errors="coerce")
    options["Delta suitability"] = options.apply(
        lambda r: 1.0 if pd.notna(r["Delta"]) and ((0.30 <= r["Delta"] <= 0.75) if r["Option"] == "CE" else (-0.75 <= r["Delta"] <= -0.30)) else 0.0,
        axis=1
    )
    options["Moneyness score"] = options["Moneyness"].map({"ATM": 3, "1-step OTM": 2, "2-step OTM": 1, "1-step ITM": 1, "2-step ITM": 0}).fillna(0)
    # Percentile ranks avoid arbitrary absolute volume/OI cutoffs across different stocks.
    options["Volume rank"] = options["Volume"].rank(pct=True).fillna(0)
    options["OI rank"] = options["OI"].rank(pct=True).fillna(0)
    spread_quality = (1 - options["Spread %"].clip(lower=0, upper=10) / 10).fillna(0)
    options["Selection score"] = (
        options["Underlying strength"] * 10
        + options["Volume rank"] * 20
        + options["OI rank"] * 10
        + spread_quality * 20
        + options["Delta suitability"] * 15
        + options["Moneyness score"] * 5
    ).round(1)
    # Ask is the conservative displayed reference for a potential buy; quotes can change instantly.
    options["Indicative entry (Ask)"] = options["Ask"]
    options["Illustrative SL (-20%)"] = (options["Ask"] * 0.80).round(2)
    options["Illustrative T1 (+40%)"] = (options["Ask"] * 1.40).round(2)
    options["Illustrative T2 (+60%)"] = (options["Ask"] * 1.60).round(2)
    options["Illustrative R:R to T1"] = "1:2"
    eligible = options[options["Liquidity"] == "PASS"].copy()
    # One best contract per underlying; rank across all symbols in each direction.
    eligible = eligible.sort_values(["Selection score", "Volume"], ascending=False)
    eligible = eligible.drop_duplicates(subset=["Symbol", "Bias"], keep="first")
    return eligible

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

st.warning("A full 218-stock scan makes up to 436 candle requests. Start with 20 or 50. Options Intelligence is a separate on-demand lookup for the top 5 bullish and bearish scan candidates; it uses the nearest listed expiry and shows live chain fields returned by Upstox. No order placement.")

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


# V4.3 Trade Selection Dashboard: separate, user-triggered option-chain lookup.
st.divider()
st.header("🧠 V4.3 Trade Selection Dashboard")
st.caption("Uses Upstox option contracts + option-chain endpoints. Displays CE for bullish candidates and PE for bearish candidates, with ATM, 1/2-step OTM and 1/2-step ITM. Greeks, OI, volume and bid/ask are shown only when supplied by the API.")
max_spread = st.slider("Maximum bid–ask spread for liquidity PASS (%)", 1, 30, 10)
if st.button("🔎 Load Options Intelligence", disabled=(not bool(token) or "v31_records" not in st.session_state)):
    scan_df = pd.DataFrame(st.session_state.get("v31_records", []))
    if scan_df.empty:
        st.warning("Run the stock scan first.")
    else:
        bull_top = scan_df[(scan_df["5m Bull"] >= min_score) & (scan_df["15m Bull"] >= min_score) & (scan_df["Bull total"] > scan_df["Bear total"])].sort_values("Bull total", ascending=False).head(5)
        bear_top = scan_df[(scan_df["5m Bear"] >= min_score) & (scan_df["15m Bear"] >= min_score) & (scan_df["Bear total"] > scan_df["Bull total"])].sort_values("Bear total", ascending=False).head(5)
        universe_map = st.session_state["universe"].set_index("underlying_symbol")["underlying_key"].to_dict()
        opt_rows, opt_errors = [], []
        picks = [(r["Symbol"], "Bullish") for _, r in bull_top.iterrows()] + [(r["Symbol"], "Bearish") for _, r in bear_top.iterrows()]
        if not picks:
            st.info("No bullish/bearish candidates passed the current scan filters.")
        progress = st.progress(0)
        for i, (sym, bias) in enumerate(picks, start=1):
            try:
                rows, err = build_option_rows(sym, universe_map[sym], bias, token, max_spread)
                opt_rows.extend(rows)
                if err: opt_errors.append(err)
            except Exception as exc:
                opt_errors.append(f"{sym}: {str(exc)[:160]}")
            progress.progress(i/max(len(picks),1))
            time.sleep(0.15)
        st.session_state["v4_options"] = opt_rows
        st.session_state["v4_option_errors"] = opt_errors
        st.session_state["v4_option_time"] = datetime.now(IST).strftime("%d %b %Y %I:%M:%S %p")
if "v4_options" in st.session_state:
    opt_df = pd.DataFrame(st.session_state["v4_options"])
    st.caption(f"Option-chain lookup: {st.session_state.get('v4_option_time','')} | Contracts shown: {len(opt_df)}")
    if not opt_df.empty:
        st.markdown("**Liquidity PASS** means live LTP, bid, ask and volume are present and spread is within your selected limit. REJECT rows are displayed for comparison, not trade selection.**")
        st.dataframe(opt_df.drop(columns=["Instrument Key"]), use_container_width=True, hide_index=True)
        st.download_button("Download Options Intelligence CSV", opt_df.to_csv(index=False), "orderblock_pro_v4_3_options_all.csv", "text/csv")
        ranked = rank_option_candidates(st.session_state["v4_options"], st.session_state.get("v31_records", []))
        st.divider()
        st.subheader("🏆 Ranked trade candidates — research shortlist")
        if ranked.empty:
            st.warning("No contracts passed the current liquidity filter. Do not force a trade; widen the scan or wait for better liquidity.")
        else:
            ce = ranked[ranked["Bias"] == "Bullish"].sort_values("Selection score", ascending=False).head(5)
            pe = ranked[ranked["Bias"] == "Bearish"].sort_values("Selection score", ascending=False).head(5)
            display_cols = ["Symbol", "Bias", "Expiry", "Option", "Strike", "Moneyness", "Spot", "LTP", "Bid", "Ask", "Spread %", "Volume", "OI", "Delta", "Theta", "Selection score", "Indicative entry (Ask)", "Illustrative SL (-20%)", "Illustrative T1 (+40%)", "Illustrative T2 (+60%)", "Illustrative R:R to T1"]
            a, b = st.columns(2)
            with a:
                st.markdown("#### 🟢 Top bullish CE")
                st.dataframe(ce[[c for c in display_cols if c in ce.columns]], use_container_width=True, hide_index=True)
            with b:
                st.markdown("#### 🔴 Top bearish PE")
                st.dataframe(pe[[c for c in display_cols if c in pe.columns]], use_container_width=True, hide_index=True)
            st.download_button("Download ranked shortlist CSV", ranked.to_csv(index=False), "orderblock_pro_v4_3_ranked_shortlist.csv", "text/csv")
        st.info("Ranking is a transparent heuristic using underlying 5m/15m score, option volume/OI percentiles, spread, delta suitability and strike distance. Entry uses displayed Ask. SL/T1/T2 are illustrative premium percentages only—not chart-derived, backtested, or validated trade levels. Quotes change rapidly; confirm the live chain before making any decision.")
    if st.session_state.get("v4_option_errors"):
        with st.expander(f"Option lookup warnings ({len(st.session_state['v4_option_errors'])})"):
            st.write(st.session_state["v4_option_errors"])

st.caption("V4.3 adds a heuristic ranked shortlist on top of the V4.2 options integration, not a proven trading system. Futures OI confirmation and actual Order Block detection are not yet implemented. No automated order placement.")
