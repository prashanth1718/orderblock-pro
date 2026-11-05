import gzip
import io
import time
from datetime import datetime, time as dtime
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="OrderBlock Pro V4.5", page_icon="📊", layout="wide")
st.title("📊 OrderBlock Pro V4.5 — NSE F&O Trade Selection")
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
    # Multi-day historical candles provide enough bars for 15m analysis.
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
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    return (
        frame.dropna(subset=["datetime", "close"])
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def analyse(frame, swing_lookback, volume_multiple):
    if len(frame) < 25:
        return None
    x = frame.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in x.columns:
            x[col] = float("nan")
        x[col] = pd.to_numeric(x[col], errors="coerce").astype("float64")
    x = x.dropna(subset=["high", "low", "close", "volume"]).reset_index(drop=True)
    if len(x) < 25:
        return None

    x["ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    x["vol_avg"] = x["volume"].rolling(20, min_periods=1).mean()
    x["vol_spike"] = x["volume"] >= x["vol_avg"] * volume_multiple
    prior_high = x["high"].shift(1).rolling(swing_lookback).max()
    prior_low = x["low"].shift(1).rolling(swing_lookback).min()
    bull_sweep = (x["low"] < prior_low) & (x["close"] > prior_low)
    bear_sweep = (x["high"] > prior_high) & (x["close"] < prior_high)
    bull_fvg = x["low"] > x["high"].shift(2)
    bear_fvg = x["high"] < x["low"].shift(2)
    last = x.iloc[-1]
    bull = (
        int(bool(bull_sweep.iloc[-1]))
        + int(bool(bull_fvg.iloc[-1]))
        + int(bool(last["vol_spike"]))
        + int(last["close"] > last["ema20"])
        + int(last["ema20"] > last["ema50"])
    )
    bear = (
        int(bool(bear_sweep.iloc[-1]))
        + int(bool(bear_fvg.iloc[-1]))
        + int(bool(last["vol_spike"]))
        + int(last["close"] < last["ema20"])
        + int(last["ema20"] < last["ema50"])
    )
    return {
        "close": float(last["close"]),
        "bull": bull,
        "bear": bear,
        "volume_spike": bool(last["vol_spike"]),
        "bull_sweep": bool(bull_sweep.iloc[-1]),
        "bear_sweep": bool(bear_sweep.iloc[-1]),
        "candle_time": last["datetime"],
    }


def detect_price_action_levels(frame, direction, min_ob_score=3):
    """Experimental OB quality filter; not a validated institutional order-flow model."""
    empty = {
        "OB zone": None, "FVG zone": None, "Structure SL": None,
        "Target 1 (1R)": None, "Target 2 (2R)": None,
        "OB strength": 0, "OB state": "Not found", "Liquidity sweep": "No",
        "Level status": "No qualifying OB found",
    }
    if frame is None or len(frame) < 30:
        empty["Level status"] = "Insufficient candles"
        return empty

    x = frame.tail(100).copy().reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in x.columns:
            x[col] = 0.0
        # Force true numeric dtype before any rolling aggregation.
        x[col] = pd.to_numeric(x[col], errors="coerce").astype("float64")
    x = x.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if len(x) < 30:
        empty["Level status"] = "Insufficient numeric candles"
        return empty

    # FIX: avoid replace(0, pd.NA), which can coerce the series to object dtype.
    # Float NaN keeps the series numeric so rolling().mean() works reliably.
    rng = (x["high"] - x["low"]).astype("float64")
    rng = rng.mask(rng.eq(0), float("nan"))
    body = (x["close"] - x["open"]).abs().astype("float64")
    atr = rng.rolling(14, min_periods=5).mean()
    volavg = x["volume"].rolling(20, min_periods=5).mean()
    last_price = float(x.iloc[-1]["close"])

    fvg = None
    for i in range(len(x) - 1, 1, -1):
        a, c = x.iloc[i - 2], x.iloc[i]
        if direction == "Bullish" and float(c.low) > float(a.high):
            fvg = (float(a.high), float(c.low))
            break
        if direction == "Bearish" and float(c.high) < float(a.low):
            fvg = (float(c.high), float(a.low))
            break

    sweep = False
    for i in range(max(10, len(x) - 8), len(x)):
        prior = x.iloc[max(0, i - 10):i]
        if len(prior) < 5:
            continue
        row = x.iloc[i]
        if direction == "Bullish" and row.low < prior.low.min() and row.close > prior.low.min():
            sweep = True
        if direction == "Bearish" and row.high > prior.high.max() and row.close < prior.high.max():
            sweep = True

    candidates = []
    for i in range(max(2, len(x) - 35), len(x) - 1):
        ob = x.iloc[i]
        nxt = x.iloc[i + 1]
        local = x.iloc[max(0, i - 8):i]
        if len(local) < 5:
            continue
        is_opposite = (ob.close < ob.open) if direction == "Bullish" else (ob.close > ob.open)
        if not is_opposite:
            continue

        if direction == "Bullish":
            displacement = nxt.close > nxt.open and nxt.close > local.high.max()
        else:
            displacement = nxt.close < nxt.open and nxt.close < local.low.min()

        atr_i = float(atr.iloc[i + 1]) if pd.notna(atr.iloc[i + 1]) else 0.0
        vol_i = float(volavg.iloc[i + 1]) if pd.notna(volavg.iloc[i + 1]) else 0.0
        impulse = float(body.iloc[i + 1]) >= 0.8 * atr_i if atr_i > 0 else False
        volume_ok = float(nxt.volume) >= 1.2 * vol_i if vol_i > 0 else False
        if not (displacement and impulse):
            continue

        lo, hi = float(ob.low), float(ob.high)
        later = x.iloc[i + 2:]
        if direction == "Bullish":
            invalid = bool((later.close < lo).any())
            touched = bool(((later.low <= hi) & (later.high >= lo)).any())
            still_valid = last_price > lo
            stop = lo * 0.999
        else:
            invalid = bool((later.close > hi).any())
            touched = bool(((later.high >= lo) & (later.low <= hi)).any())
            still_valid = last_price < hi
            stop = hi * 1.001

        state = "Invalidated" if invalid or not still_valid else ("Mitigated" if touched else "Fresh")
        score = 2 + int(volume_ok) + int(sweep) + int(state == "Fresh")
        candidates.append((score, i, lo, hi, stop, state, volume_ok))

    valid = [c for c in candidates if c[0] >= min_ob_score and c[5] != "Invalidated"]
    if not valid:
        out = empty.copy()
        out["FVG zone"] = f"{fvg[0]:.2f}–{fvg[1]:.2f}" if fvg else None
        out["Liquidity sweep"] = "Yes" if sweep else "No"
        out["OB strength"] = max([c[0] for c in candidates], default=0)
        out["OB state"] = "No qualifying zone"
        out["Level status"] = "No strong, non-invalidated OB"
        return out

    score, i, lo, hi, stop, state, volume_ok = max(valid, key=lambda z: (z[0], z[1]))
    risk = last_price - stop if direction == "Bullish" else stop - last_price
    if risk <= 0:
        empty.update({
            "OB zone": f"{lo:.2f}–{hi:.2f}", "OB strength": score,
            "OB state": state, "Liquidity sweep": "Yes" if sweep else "No",
            "Level status": "Price beyond invalidation level",
        })
        return empty

    t1 = last_price + risk if direction == "Bullish" else last_price - risk
    t2 = last_price + 2 * risk if direction == "Bullish" else last_price - 2 * risk
    return {
        "OB zone": f"{lo:.2f}–{hi:.2f}",
        "FVG zone": f"{fvg[0]:.2f}–{fvg[1]:.2f}" if fvg else None,
        "Structure SL": round(stop, 2),
        "Target 1 (1R)": round(t1, 2),
        "Target 2 (2R)": round(t2, 2),
        "OB strength": score,
        "OB state": state,
        "Liquidity sweep": "Yes" if sweep else "No",
        "Level status": "Experimental quality-filtered structure levels",
    }


def raise_upstox_api_error(response, endpoint_name):
    """Show API response detail without exposing request credentials."""
    try:
        detail = str(response.json())[:700]
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
    expiries = sorted({
        str(c.get("expiry")) for c in contracts
        if c.get("expiry") and str(c.get("expiry")) >= today.isoformat()
    })
    if not expiries:
        return [], f"{symbol}: no future option expiry returned"

    expiry = expiries[0]
    chain = get_option_chain(underlying_key, expiry, token)
    if not chain:
        return [], f"{symbol}: empty option chain for {expiry}"

    spot = float(chain[0].get("underlying_spot_price") or 0)
    strikes = sorted({
        float(item.get("strike_price")) for item in chain
        if item.get("strike_price") is not None
    })
    if not spot or not strikes:
        return [], f"{symbol}: missing spot/strike data"

    atm = min(strikes, key=lambda value: abs(value - spot))
    atm_idx = strikes.index(atm)
    opt_type = "CE" if direction == "Bullish" else "PE"
    wanted = []
    for offset, label in [(0, "ATM"), (1, "1-step OTM"), (2, "2-step OTM"), (-1, "1-step ITM"), (-2, "2-step ITM")]:
        idx = atm_idx + offset if opt_type == "CE" else atm_idx - offset
        if 0 <= idx < len(strikes):
            wanted.append((strikes[idx], label))

    by_strike = {float(item["strike_price"]): item for item in chain if item.get("strike_price") is not None}
    rows = []
    for strike, moneyness in wanted:
        item = by_strike.get(strike, {})
        opt = item.get("call_options" if opt_type == "CE" else "put_options") or {}
        md = opt.get("market_data") or {}
        gr = opt.get("option_greeks") or {}
        ltp = float(md.get("ltp") or 0)
        bid = float(md.get("bid_price") or 0)
        ask = float(md.get("ask_price") or 0)
        spread = ((ask - bid) / ((ask + bid) / 2) * 100) if bid > 0 and ask > 0 and ask >= bid else None
        liquid = (
            ltp > 0 and bid > 0 and ask > 0 and spread is not None
            and spread <= max_spread_pct and float(md.get("volume") or 0) > 0
        )
        rows.append({
            "Symbol": symbol, "Bias": direction, "Expiry": expiry, "Option": opt_type,
            "Strike": strike, "Moneyness": moneyness, "Spot": round(spot, 2),
            "LTP": round(ltp, 2), "Bid": round(bid, 2), "Ask": round(ask, 2),
            "Spread %": round(spread, 2) if spread is not None else None,
            "Volume": int(float(md.get("volume") or 0)), "OI": int(float(md.get("oi") or 0)),
            "Delta": gr.get("delta"), "Theta": gr.get("theta"), "Gamma": gr.get("gamma"),
            "Vega": gr.get("vega"), "IV": gr.get("iv"),
            "Liquidity": "PASS" if liquid else "REJECT",
            "Reference SL (-20%)": round(ltp * 0.8, 2) if ltp else None,
            "Reference T1 (+40%)": round(ltp * 1.4, 2) if ltp else None,
            "Reference T2 (+60%)": round(ltp * 1.6, 2) if ltp else None,
            "Instrument Key": opt.get("instrument_key", ""),
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
        lambda row: float(row.get("Bull total", 0) if row.get("Bias") == "Bullish" else row.get("Bear total", 0)),
        axis=1,
    )
    for col in ["Volume", "OI", "Spread %", "Delta"]:
        options[col] = pd.to_numeric(options[col], errors="coerce")

    options["Delta suitability"] = options.apply(
        lambda row: 1.0 if pd.notna(row["Delta"]) and (
            (0.30 <= row["Delta"] <= 0.75) if row["Option"] == "CE"
            else (-0.75 <= row["Delta"] <= -0.30)
        ) else 0.0,
        axis=1,
    )
    options["Moneyness score"] = options["Moneyness"].map({
        "ATM": 3, "1-step OTM": 2, "2-step OTM": 1, "1-step ITM": 1, "2-step ITM": 0,
    }).fillna(0)
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

    options["Indicative entry (Ask)"] = options["Ask"]
    options["Illustrative SL (-20%)"] = (options["Ask"] * 0.80).round(2)
    options["Illustrative T1 (+40%)"] = (options["Ask"] * 1.40).round(2)
    options["Illustrative T2 (+60%)"] = (options["Ask"] * 1.60).round(2)
    options["Illustrative R:R to T1"] = "1:2"
    eligible = options[options["Liquidity"] == "PASS"].copy()
    eligible = eligible.sort_values(["Selection score", "Volume"], ascending=False)
    return eligible.drop_duplicates(subset=["Symbol", "Bias"], keep="first")


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
        format_func=lambda n: "All loaded stocks" if n == 218 else str(n),
    )
    min_score = st.slider("Minimum score on each timeframe", 2, 5, 3)
    swing = st.slider("Liquidity-sweep lookback", 3, 20, 5)
    vol_mult = st.slider("Volume spike multiple", 1.0, 3.0, 1.5, 0.1)

st.warning(
    "A full 218-stock scan makes up to 436 candle requests. Start with 20 or 50. "
    "Options Intelligence is a separate on-demand lookup for the top 5 bullish and bearish scan candidates; "
    "it uses the nearest listed expiry and shows live chain fields returned by Upstox. No order placement."
)

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
        st.warning(
            f"The current instrument master contains {available} stocks, "
            f"so the scan will use all {available} available stocks."
        )

    if st.button("🚀 Run 5m + 15m scan", type="primary", disabled=not bool(token)):
        records, errors = [], []
        chosen = universe.head(count_to_scan)
        progress = st.progress(0)
        status = st.empty()

        for count, (_, item) in enumerate(chosen.iterrows(), start=1):
            symbol, key = item["underlying_symbol"], item["underlying_key"]
            status.write(f"Scanning {symbol}: 5m + 15m ({count}/{count_to_scan})")
            try:
                five_frame = fetch_candles(key, 5, token)
                fifteen_frame = fetch_candles(key, 15, token)
                five = analyse(five_frame, swing, vol_mult)
                fifteen = analyse(fifteen_frame, swing, vol_mult)
                bull_levels = detect_price_action_levels(five_frame, "Bullish", min_ob_score=3)
                bear_levels = detect_price_action_levels(five_frame, "Bearish", min_ob_score=3)

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
                        **{f"Bull {key}": value for key, value in bull_levels.items()},
                        **{f"Bear {key}": value for key, value in bear_levels.items()},
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
            st.dataframe(
                data.sort_values(["Bull total", "Bear total"], ascending=False),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download scan CSV", data.to_csv(index=False),
                "orderblock_pro_full_scan.csv", "text/csv",
            )

            st.markdown("### 🎯 V4.5 Quality-filtered 5m Order Block / FVG levels")
            st.caption(
                "Underlying-price levels only. OB/FVG detection is an experimental candle heuristic, "
                "not a validated ICT model. Candidate OBs require displacement and a local structure break; "
                "volume, sweep, freshness and invalidation affect filtering/scoring. If no qualifying OB exists, levels are withheld."
            )
            level_cols = [
                "Symbol", "LTP", "Bull OB zone", "Bull FVG zone", "Bull Structure SL",
                "Bull Target 1 (1R)", "Bull Target 2 (2R)", "Bull OB strength", "Bull OB state",
                "Bull Liquidity sweep", "Bull Level status", "Bear OB zone", "Bear FVG zone",
                "Bear Structure SL", "Bear Target 1 (1R)", "Bear Target 2 (2R)",
                "Bear OB strength", "Bear OB state", "Bear Liquidity sweep", "Bear Level status",
            ]
            selected_level_cols = [col for col in level_cols if col in data.columns]
            st.dataframe(data[selected_level_cols], use_container_width=True, hide_index=True)
            st.download_button(
                "Download price-action levels CSV",
                data[selected_level_cols].to_csv(index=False),
                "orderblock_pro_v4_5_price_action_levels.csv",
                "text/csv",
            )

            if bullish.empty and bearish.empty:
                st.info("No stocks met the selected threshold on both timeframes. Do not force signals by lowering filters.")
        else:
            st.warning("No usable stock results were returned.")

        if st.session_state.get("v31_errors"):
            with st.expander(f"Skipped stocks / errors ({len(st.session_state['v31_errors'])})"):
                st.write(st.session_state["v31_errors"])

st.caption(
    "Technical heuristic only: EMA20/EMA50, relative volume, swing-sweep and three-candle FVG checks. "
    "It does not prove institutional activity or a profitable edge. No trades are placed."
)

st.divider()
st.header("🧠 V4.5 Trade Selection Dashboard")
st.caption(
    "Uses Upstox option contracts + option-chain endpoints. Displays CE for bullish candidates and PE for bearish candidates, "
    "with ATM, 1/2-step OTM and 1/2-step ITM. Greeks, OI, volume and bid/ask are shown only when supplied by the API."
)
max_spread = st.slider("Maximum bid–ask spread for liquidity PASS (%)", 1, 30, 10)

if st.button(
    "🔎 Load Options Intelligence",
    disabled=(not bool(token) or "v31_records" not in st.session_state),
):
    scan_df = pd.DataFrame(st.session_state.get("v31_records", []))
    if scan_df.empty:
        st.warning("Run the stock scan first.")
    else:
        bull_top = scan_df[
            (scan_df["5m Bull"] >= min_score) & (scan_df["15m Bull"] >= min_score)
            & (scan_df["Bull total"] > scan_df["Bear total"])
        ].sort_values("Bull total", ascending=False).head(5)
        bear_top = scan_df[
            (scan_df["5m Bear"] >= min_score) & (scan_df["15m Bear"] >= min_score)
            & (scan_df["Bear total"] > scan_df["Bull total"])
        ].sort_values("Bear total", ascending=False).head(5)

        universe_map = st.session_state["universe"].set_index("underlying_symbol")["underlying_key"].to_dict()
        opt_rows, opt_errors = [], []
        picks = (
            [(row["Symbol"], "Bullish") for _, row in bull_top.iterrows()]
            + [(row["Symbol"], "Bearish") for _, row in bear_top.iterrows()]
        )
        if not picks:
            st.info("No bullish/bearish candidates passed the current scan filters.")

        progress = st.progress(0)
        for i, (sym, bias) in enumerate(picks, start=1):
            try:
                rows, err = build_option_rows(sym, universe_map[sym], bias, token, max_spread)
                opt_rows.extend(rows)
                if err:
                    opt_errors.append(err)
            except Exception as exc:
                opt_errors.append(f"{sym}: {str(exc)[:160]}")
            progress.progress(i / max(len(picks), 1))
            time.sleep(0.15)

        st.session_state["v4_options"] = opt_rows
        st.session_state["v4_option_errors"] = opt_errors
        st.session_state["v4_option_time"] = datetime.now(IST).strftime("%d %b %Y %I:%M:%S %p")

if "v4_options" in st.session_state:
    opt_df = pd.DataFrame(st.session_state["v4_options"])
    st.caption(
        f"Option-chain lookup: {st.session_state.get('v4_option_time', '')} | Contracts shown: {len(opt_df)}"
    )
    if not opt_df.empty:
        st.markdown(
            "**Liquidity PASS** means live LTP, bid, ask and volume are present and spread is within your selected limit. "
            "REJECT rows are displayed for comparison, not trade selection."
        )
        st.dataframe(
            opt_df.drop(columns=["Instrument Key"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download Options Intelligence CSV", opt_df.to_csv(index=False),
            "orderblock_pro_v4_3_options_all.csv", "text/csv",
        )

        ranked = rank_option_candidates(st.session_state["v4_options"], st.session_state.get("v31_records", []))
        st.divider()
        st.subheader("🏆 Ranked trade candidates — research shortlist")
        if ranked.empty:
            st.warning(
                "No contracts passed the current liquidity filter. Do not force a trade; "
                "widen the scan or wait for better liquidity."
            )
        else:
            ce = ranked[ranked["Bias"] == "Bullish"].sort_values("Selection score", ascending=False).head(5)
            pe = ranked[ranked["Bias"] == "Bearish"].sort_values("Selection score", ascending=False).head(5)
            display_cols = [
                "Symbol", "Bias", "Expiry", "Option", "Strike", "Moneyness", "Spot", "LTP",
                "Bid", "Ask", "Spread %", "Volume", "OI", "Delta", "Theta", "Selection score",
                "Indicative entry (Ask)", "Illustrative SL (-20%)", "Illustrative T1 (+40%)",
                "Illustrative T2 (+60%)", "Illustrative R:R to T1",
            ]
            a, b = st.columns(2)
            with a:
                st.markdown("#### 🟢 Top bullish CE")
                st.dataframe(
                    ce[[col for col in display_cols if col in ce.columns]],
                    use_container_width=True, hide_index=True,
                )
            with b:
                st.markdown("#### 🔴 Top bearish PE")
                st.dataframe(
                    pe[[col for col in display_cols if col in pe.columns]],
                    use_container_width=True, hide_index=True,
                )
            st.download_button(
                "Download ranked shortlist CSV", ranked.to_csv(index=False),
                "orderblock_pro_v4_3_ranked_shortlist.csv", "text/csv",
            )

        st.info(
            "Ranking is a transparent heuristic using underlying 5m/15m score, option volume/OI percentiles, "
            "spread, delta suitability and strike distance. Entry uses displayed Ask. SL/T1/T2 are illustrative "
            "premium percentages only—not chart-derived, backtested, or validated trade levels. "
            "Quotes change rapidly; confirm the live chain before making any decision."
        )

    if st.session_state.get("v4_option_errors"):
        with st.expander(f"Option lookup warnings ({len(st.session_state['v4_option_errors'])})"):
            st.write(st.session_state["v4_option_errors"])

st.caption(
    "V4.5 adds quality-filtered experimental OB zones and a heuristic ranked shortlist and experimental "
    "price-action levels on top of the V4.2 options integration, not a proven trading system. "
    "Futures OI confirmation and institutional order-flow confirmation is not implemented. "
    "No automated order placement."
)
