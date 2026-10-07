import math
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Crypto Leverage - AAVE (Base)", layout="wide")
st.title("Crypto Leverage (AAVE - Base)")
st.caption(
    "Informational tool only. Not financial advice. Leverage can lead to total loss and liquidation. "
    "Your wallet address is only used for a read-only lookup and is not stored."
)


# ============================================================
# CACHED DATA FETCHERS
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_crypto_technicals(symbol: str):
    """Fetch daily/weekly crypto data used by the market command center.
    Mayer Multiple is intentionally not part of the strategy anymore; valuation
    is handled with 50W/200W structure, Fibonacci location and historical outcomes."""
    result = {
        "price": None,
        "sma200d": None,
        "ema200w": None,
        "dist_ema200w": None,
        "change_24h_pct": None,
        "trend_streak": 0,
        "error": None,
    }
    try:
        stock = yf.Ticker(symbol)

        hist_d = stock.history(period="2y", interval="1d")
        if not hist_d.empty and len(hist_d) >= 200:
            close_d = hist_d["Close"]
            price = close_d.iloc[-1]
            sma200d = close_d.rolling(window=200).mean().iloc[-1]
            result["price"] = price
            result["sma200d"] = sma200d
            if len(close_d) >= 2:
                prev_close = close_d.iloc[-2]
                if prev_close:
                    result["change_24h_pct"] = ((price - prev_close) / prev_close) * 100

        hist_w = stock.history(period="5y", interval="1wk")
        if not hist_w.empty and len(hist_w) >= 200:
            close_w = hist_w["Close"]
            ema200w = close_w.ewm(span=200, adjust=False).mean().iloc[-1]
            result["ema200w"] = ema200w
            if result["price"] and ema200w > 0:
                result["dist_ema200w"] = ((result["price"] - ema200w) / ema200w) * 100

            closes = close_w.tolist()
            streak = 0
            if len(closes) >= 2:
                direction = 1 if closes[-1] > closes[-2] else (-1 if closes[-1] < closes[-2] else 0)
                if direction != 0:
                    streak = direction
                    i = len(closes) - 2
                    while i > 0:
                        step_dir = 1 if closes[i] > closes[i - 1] else (-1 if closes[i] < closes[i - 1] else 0)
                        if step_dir == direction:
                            streak += direction
                            i -= 1
                        else:
                            break
            result["trend_streak"] = streak
    except Exception as e:
        result["error"] = str(e)

    return result


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weekly_chart_data(symbol: str):
    """Weekly close price plus 50-week SMA and 200-week EMA, for charting."""
    try:
        stock = yf.Ticker(symbol)
        hist_w = stock.history(period="5y", interval="1wk")
        if hist_w.empty:
            return None
        close_w = hist_w["Close"]
        df = pd.DataFrame({
            "Close": close_w,
            "SMA50W": close_w.rolling(window=50).mean(),
            "EMA200W": close_w.ewm(span=200, adjust=False).mean(),
        })
        return df
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def fetch_aave_account_data(wallet_address: str, rpc_url: str):
    """Read-only call to AAVE v3 Pool.getUserAccountData(address) on Base.
    Returns a dict of USD figures + ratios, or an 'error' key if it fails.

    NOTE: amounts come back scaled by the price oracle's BASE_CURRENCY_UNIT.
    AAVE's Base market oracle uses USD with 8 decimals, which is what this
    assumes below. If your numbers look 100x/1,000,000x off, that decimal
    assumption is the first thing to check against the AAVE docs.
    """
    try:
        from web3 import Web3
    except ImportError:
        return {"error": "The 'web3' package is not installed. Run: pip install web3"}

    POOL_ADDRESS = Web3.to_checksum_address("0xA238Dd80C259a72e81d7e4664a9801593F98d1c5")
    ABI = [
        {
            "inputs": [{"internalType": "address", "name": "user", "type": "address"}],
            "name": "getUserAccountData",
            "outputs": [
                {"internalType": "uint256", "name": "totalCollateralBase", "type": "uint256"},
                {"internalType": "uint256", "name": "totalDebtBase", "type": "uint256"},
                {"internalType": "uint256", "name": "availableBorrowsBase", "type": "uint256"},
                {"internalType": "uint256", "name": "currentLiquidationThreshold", "type": "uint256"},
                {"internalType": "uint256", "name": "ltv", "type": "uint256"},
                {"internalType": "uint256", "name": "healthFactor", "type": "uint256"},
            ],
            "stateMutability": "view",
            "type": "function",
        }
    ]

    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 15}))
        if not w3.is_connected():
            return {"error": f"Could not connect to RPC endpoint: {rpc_url}"}

        checksum_addr = Web3.to_checksum_address(wallet_address)
        pool = w3.eth.contract(address=POOL_ADDRESS, abi=ABI)
        data = pool.functions.getUserAccountData(checksum_addr).call()

        BASE_UNIT = 1e8  # USD, 8 decimals (see note above)

        total_collateral = data[0] / BASE_UNIT
        total_debt = data[1] / BASE_UNIT
        available_borrows = data[2] / BASE_UNIT
        liq_threshold_pct = data[3] / 100.0  # bps -> %
        ltv_pct = data[4] / 100.0            # bps -> %
        health_factor = data[5] / 1e18

        return {
            "total_collateral": total_collateral,
            "total_debt": total_debt,
            "available_borrows": available_borrows,
            "liq_threshold_pct": liq_threshold_pct,
            "ltv_pct": ltv_pct,
            "health_factor": health_factor,
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# CRYPTO LEVERAGE / MARKET COMMAND CENTER
# ============================================================

st.header("Market Command Center")

def pct_distance(price, level):
    if price is None or level in (None, 0):
        return None
    return (price - level) / level * 100.0

def safe_num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def market_regime(price, sma50w):
    if price is None or sma50w is None:
        return "UNKNOWN"
    return "BULL MARKET" if price >= sma50w else "BEAR MARKET"

def fibonacci_levels(high, low):
    if high is None or low is None or high <= low:
        return {}
    span = high - low
    return {
        "23.6%": high - span * 0.236,
        "38.2%": high - span * 0.382,
        "50.0%": high - span * 0.500,
        "61.8%": high - span * 0.618,
        "78.6%": high - span * 0.786,
    }

def fibonacci_context(close):
    """Use the recent ~4-year weekly range as a transparent Fibonacci context.
    This is a location tool, not a standalone buy signal."""
    if close is None or len(close) < 52:
        return {"high": None, "low": None, "levels": {}, "nearest": None, "distance": None}
    window = close.tail(min(208, len(close)))
    high = float(window.max())
    low = float(window.min())
    levels = fibonacci_levels(high, low)
    price = float(close.iloc[-1])
    if not levels:
        return {"high": high, "low": low, "levels": {}, "nearest": None, "distance": None}
    nearest = min(levels.items(), key=lambda kv: abs(price - kv[1]))
    distance = pct_distance(price, nearest[1])
    return {"high": high, "low": low, "levels": levels, "nearest": nearest[0], "distance": distance}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_btc_probability_history():
    """Fetch enough weekly BTC history to estimate empirical conditional outcomes.
    Outcomes are calculated only from information available at each historical week.
    """
    try:
        stock = yf.Ticker("BTC-USD")
        hist = stock.history(period="10y", interval="1wk")
        if hist.empty or len(hist) < 260:
            return None
        close = hist["Close"].dropna().copy()
        df = pd.DataFrame(index=close.index)
        df["close"] = close
        df["sma50"] = close.rolling(50).mean()
        df["ema200"] = close.ewm(span=200, adjust=False).mean()
        df["ret12"] = close.pct_change(12)
        df["ret26"] = close.pct_change(26)
        df["high52"] = close.rolling(52).max()
        df["dd52"] = close / df["high52"] - 1
        df["weeks_since_high52"] = 0
        last_high_idx = None
        for i in range(len(df)):
            if i == 0 or df["close"].iloc[i] >= df["high52"].iloc[i]:
                last_high_idx = i
            df.iloc[i, df.columns.get_loc("weeks_since_high52")] = i - last_high_idx if last_high_idx is not None else 0
        weekly_direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        streaks = []
        streak = 0
        for d in weekly_direction:
            if d == 0:
                streak = 0
            elif streak == 0 or (streak > 0 and d > 0) or (streak < 0 and d < 0):
                streak = streak + d if streak else d
            else:
                streak = d
            streaks.append(streak)
        df["streak"] = streaks

        # Forward outcomes. These are deliberately concrete and mutually readable:
        # 12W deeper drawdown, 26W recovery, and 26W reclaim of 50W.
        future = close.shift(-1)
        max_fwd12 = pd.concat([close.shift(-i) / close - 1 for i in range(1, 13)], axis=1).min(axis=1)
        max_fwd26 = pd.concat([close.shift(-i) / close - 1 for i in range(1, 27)], axis=1).max(axis=1)
        reclaim50 = []
        for i in range(len(df)):
            future_closes = close.iloc[i + 1:i + 27]
            future_ma = df["sma50"].iloc[i + 1:i + 27]
            reclaim50.append(bool(((future_closes >= future_ma).fillna(False)).any()) if len(future_closes) else False)
        df["deeper12"] = max_fwd12 <= -0.15
        df["recover26"] = max_fwd26 >= 0.20
        df["reclaim50_26"] = reclaim50
        return df.dropna(subset=["sma50", "ema200", "ret12", "ret26", "dd52"])
    except Exception:
        return None

def condition_bucket(row):
    price = row["close"]
    sma50 = row["sma50"]
    ema200 = row["ema200"]
    d50 = (price - sma50) / sma50 if sma50 else 0
    d200 = (price - ema200) / ema200 if ema200 else 0
    if d200 <= -0.20:
        value_zone = "deep"
    elif d200 < 0:
        value_zone = "below200"
    elif d50 < 0:
        value_zone = "bear_above200"
    elif d50 <= 0.10:
        value_zone = "near50"
    else:
        value_zone = "extended"
    if row["streak"] >= 4:
        streak_zone = "green4"
    elif row["streak"] <= -4:
        streak_zone = "red4"
    else:
        streak_zone = "mixed"
    if row["weeks_since_high52"] < 8 and row["ret12"] < -0.10:
        maturity = "early_drop"
    elif row["weeks_since_high52"] >= 20 and row["dd52"] <= -0.25:
        maturity = "mature_drop"
    else:
        maturity = "mid"
    return value_zone, streak_zone, maturity

def empirical_probabilities(current_row, hist):
    """Estimate conditional probabilities using historical BTC analogs.
    Uses progressively broader samples so the UI never presents a tiny sample
    as if it were statistically strong."""
    if hist is None or hist.empty:
        return {"deeper12": None, "recover26": None, "reclaim50_26": None, "n": 0, "level": "No history"}
    target = condition_bucket(current_row)
    buckets = []
    for _, row in hist.iterrows():
        buckets.append(condition_bucket(row))
    h = hist.copy()
    h["_value"], h["_streak"], h["_maturity"] = zip(*buckets)
    v, st, m = target
    filters = [
        (h["_value"] == v) & (h["_streak"] == st) & (h["_maturity"] == m),
        (h["_value"] == v) & (h["_maturity"] == m),
        (h["_value"] == v),
        (h["sma50"].notna()),
    ]
    labels = ["value + streak + maturity", "value + maturity", "value zone", "all history"]
    sample = h.iloc[0:0]
    level = labels[-1]
    for f, label in zip(filters, labels):
        candidate = h.loc[f]
        if len(candidate) >= 12:
            sample = candidate
            level = label
            break
    if sample.empty:
        sample = h
    return {
        "deeper12": float(sample["deeper12"].mean() * 100),
        "recover26": float(sample["recover26"].mean() * 100),
        "reclaim50_26": float(sample["reclaim50_26"].mean() * 100),
        "n": len(sample),
        "level": level,
    }

def decline_maturity(close, sma50w):
    if close is None or len(close) < 52:
        return {"weeks_since_high": 0, "dd52": None, "ret12": None, "ret26": None, "phase": "Unknown"}
    price = float(close.iloc[-1])
    high52 = float(close.tail(52).max())
    weeks_since_high = 0
    for i in range(1, min(52, len(close)) + 1):
        if float(close.iloc[-i]) == high52:
            weeks_since_high = i - 1
            break
    ret12 = float(price / close.iloc[-13] - 1) if len(close) >= 13 else None
    ret26 = float(price / close.iloc[-27] - 1) if len(close) >= 27 else None
    dd52 = price / high52 - 1 if high52 else None
    if weeks_since_high < 8 and ret12 is not None and ret12 < -0.10:
        phase = "EARLY DROP"
    elif weeks_since_high >= 20 and dd52 is not None and dd52 <= -0.25:
        phase = "MATURE BEAR MOVE"
    elif dd52 is not None and dd52 <= -0.15:
        phase = "ESTABLISHED DECLINE"
    else:
        phase = "NORMAL / MIXED"
    return {"weeks_since_high": weeks_since_high, "dd52": dd52, "ret12": ret12, "ret26": ret26, "phase": phase}

def build_current_probability_row(price, sma50w, ema200w, streak, maturity):
    return pd.Series({
        "close": price,
        "sma50": sma50w,
        "ema200": ema200w,
        "ret12": maturity["ret12"] if maturity["ret12"] is not None else 0,
        "ret26": maturity["ret26"] if maturity["ret26"] is not None else 0,
        "dd52": maturity["dd52"] if maturity["dd52"] is not None else 0,
        "weeks_since_high52": maturity["weeks_since_high"],
        "streak": streak,
    })

def buy_intensity(price, sma50w, ema200w, fib, maturity, streak, probs):
    """Return a bounded 0..1 execution intensity.
    This is intentionally an allocation rule, not a probability."""
    if price is None or sma50w is None:
        return 0.5, "INSUFFICIENT DATA"
    d50 = pct_distance(price, sma50w) or 0
    d200 = pct_distance(price, ema200w) if ema200w else None
    intensity = 0.65
    if d50 < 0:
        intensity += min(0.20, abs(d50) / 100 * 0.8)
    elif d50 > 20:
        return 0.0, "STOP / TOO EXTENDED"
    elif d50 > 10:
        intensity -= 0.25
    else:
        intensity -= max(0, d50) * 0.01
    if d200 is not None and d200 < 0:
        intensity += 0.18
    if fib.get("nearest") in {"61.8%", "78.6%"} and abs(fib.get("distance") or 999) <= 4:
        intensity += 0.10
    if streak >= 4:
        intensity -= 0.25
    elif streak <= -4:
        intensity += 0.08
    if maturity["phase"] == "EARLY DROP":
        intensity -= 0.25
    elif maturity["phase"] == "MATURE BEAR MOVE":
        intensity += 0.12
    if probs.get("deeper12") is not None and probs["deeper12"] >= 60:
        intensity -= 0.12
    if probs.get("recover26") is not None and probs["recover26"] >= 65:
        intensity += 0.08
    intensity = max(0.0, min(1.0, intensity))
    if intensity <= 0.10:
        label = "STOP / PRESERVE CASH"
    elif intensity < 0.45:
        label = "LIGHT ACCUMULATION"
    elif intensity < 0.70:
        label = "NORMAL ACCUMULATION"
    elif intensity < 0.88:
        label = "AGGRESSIVE ACCUMULATION"
    else:
        label = "DEEP-VALUE ACCUMULATION"
    return intensity, label

def loop_risk_profile(price, sma50w, ema200w, streak, maturity, probs):
    """Market-risk budget. AAVE headroom is only the hard safety ceiling."""
    below_200 = price is not None and ema200w is not None and price < ema200w
    bear = price is not None and sma50w is not None and price < sma50w
    intensity, intensity_label = buy_intensity(price, sma50w, ema200w, {}, maturity, streak, probs)
    if below_200:
        loops = 2
        base_multiple = 0.50
        mode = "DEEP-VALUE / 2-LOOP ELIGIBLE"
    elif bear:
        loops = 1
        base_multiple = 0.15
        mode = "DEFENSIVE BEAR"
    else:
        loops = 1
        base_multiple = 0.10
        mode = "BULL DISCIPLINE"
    # High probability of further downside suppresses leverage even when value is attractive.
    if probs.get("deeper12") is not None:
        if probs["deeper12"] >= 60:
            base_multiple *= 0.45
        elif probs["deeper12"] >= 45:
            base_multiple *= 0.70
    if intensity < 0.25:
        base_multiple *= 0.35
    if streak >= 4:
        base_multiple *= 0.50
    if maturity["phase"] == "EARLY DROP":
        base_multiple *= 0.60
    if below_200 and intensity >= 0.75 and probs.get("recover26", 0) >= 45:
        base_multiple *= 1.15
    return {
        "below_200": below_200,
        "bear": bear,
        "loops": loops,
        "borrow_multiple": max(0.0, min(base_multiple, 0.75)),
        "mode": mode,
        "intensity": intensity,
        "intensity_label": intensity_label,
    }

def max_borrow_for_loop(collateral, debt, fresh_cash, liq_threshold, ltv, target_hf):
    C = max(0.0, collateral)
    D = max(0.0, debt)
    S = max(0.0, fresh_cash)
    L = max(0.0, liq_threshold)
    V = max(0.0, min(0.9999, ltv))
    H = max(1.0, target_hf)
    hf_den = H - L
    hf_cap = float("inf") if hf_den <= 0 else (L * (C + S) - H * D) / hf_den
    ltv_den = 1.0 - V
    ltv_cap = float("inf") if ltv_den <= 0 else (V * (C + S) - D) / ltv_den
    return max(0.0, min(hf_cap, ltv_cap))

def simulate_sequential_loops(collateral, debt, liq_threshold, ltv, target_hf, fresh_cash, loop_count, total_borrow_budget):
    """Use one total fresh-capital allocation and recalculate after each loop.
    The market-risk budget is split across the permitted loops; AAVE is checked each time."""
    C, D = max(0.0, collateral), max(0.0, debt)
    L, V, H = max(0.0, liq_threshold), max(0.0, min(0.9999, ltv)), max(1.0, target_hf)
    rows = []
    fresh_per_loop = fresh_cash / max(1, loop_count)
    remaining_budget = max(0.0, total_borrow_budget)
    for i in range(loop_count):
        if fresh_per_loop <= 0:
            break
        before_c, before_d = C, D
        max_b = max_borrow_for_loop(before_c, before_d, fresh_per_loop, L, V, H)
        budget_b = remaining_budget if i == loop_count - 1 else total_borrow_budget / loop_count
        borrow = min(max_b, budget_b)
        C += fresh_per_loop + borrow
        D += borrow
        remaining_budget = max(0.0, remaining_budget - borrow)
        hf_after = float("inf") if D <= 0 else (C * L) / D
        ltv_after = 0.0 if C <= 0 else D / C
        rows.append({
            "loop": i + 1, "fresh": fresh_per_loop, "max_borrow": max_b,
            "borrow": borrow, "total_deployed": fresh_per_loop + borrow,
            "collateral_after": C, "debt_after": D,
            "hf_after": hf_after, "ltv_after": ltv_after,
        })
    return rows

# ---------- controls ----------
top_a, top_b, top_c = st.columns([2, 1, 1])
with top_a:
    asset_mode = st.radio("Assets", ["BTC only", "BTC + ETH"], horizontal=True, key="asset_mode_v3")
with top_b:
    target_hf = st.number_input("Target HF", min_value=1.1, max_value=3.0, value=1.8, step=0.1, key="target_hf_v3")
with top_c:
    if st.button("↻ Refresh", key="refresh_v3"):
        fetch_aave_account_data.clear(); fetch_crypto_technicals.clear(); fetch_weekly_chart_data.clear(); fetch_btc_probability_history.clear(); st.rerun()

use_eth = asset_mode == "BTC + ETH"
with st.spinner("Loading market data..."):
    btc = fetch_crypto_technicals("BTC-USD")
    eth = fetch_crypto_technicals("ETH-USD")
assets = [("BTC", btc)] + ([ ("ETH", eth) ] if use_eth else [])
btc_price = btc.get("price")
btc_ema200w = btc.get("ema200w")
btc_streak = btc.get("trend_streak", 0)
btc_change_24h = btc.get("change_24h_pct")
btc_chart = fetch_weekly_chart_data("BTC-USD")
btc_sma50w = None
if btc_chart is not None and not btc_chart.empty:
    btc_sma50w = float(btc_chart["SMA50W"].iloc[-1]) if pd.notna(btc_chart["SMA50W"].iloc[-1]) else None

if btc_price is None or btc_sma50w is None or btc_ema200w is None:
    st.error(
        "Could not load BTC market data from Yahoo Finance (it may be rate-limiting or temporarily down). "
        "Wait a minute and press Refresh."
    )
    if st.button("↻ Retry", key="retry_data"):
        fetch_crypto_technicals.clear(); fetch_weekly_chart_data.clear(); st.rerun()
    st.stop()

maturity = decline_maturity(btc_chart["Close"] if btc_chart is not None else None, btc_sma50w)
fib = fibonacci_context(btc_chart["Close"] if btc_chart is not None else None)
hist_prob = fetch_btc_probability_history()
current_row = build_current_probability_row(btc_price, btc_sma50w, btc_ema200w, btc_streak, maturity)
probs = empirical_probabilities(current_row, hist_prob)
intensity, intensity_label = buy_intensity(btc_price, btc_sma50w, btc_ema200w, fib, maturity, btc_streak, probs)
btc_regime = market_regime(btc_price, btc_sma50w)
risk_profile = loop_risk_profile(btc_price, btc_sma50w, btc_ema200w, btc_streak, maturity, probs)
below_200w = risk_profile["below_200"]
recommended_loops = risk_profile["loops"]

# ---------- COMPACT MARKET OVERVIEW + RECOMMENDATION ----------
st.subheader("Market Overview & Recommendation")
regime_icon = "🟢" if btc_regime == "BULL MARKET" else "🔴" if btc_regime == "BEAR MARKET" else "🟡"
h1, h2, h3, h4, h5 = st.columns(5)
h1.metric("Regime", f"{regime_icon} {btc_regime}")
h2.metric("BTC", f"${btc_price:,.0f}" if btc_price else "N/A")
h3.metric("50W MA", f"${btc_sma50w:,.0f}" if btc_sma50w else "N/A", delta=f"{(pct_distance(btc_price,btc_sma50w) or 0):+.1f}%")
h4.metric("200W EMA", f"${btc_ema200w:,.0f}" if btc_ema200w else "N/A", delta=f"{(pct_distance(btc_price,btc_ema200w) or 0):+.1f}%")
h5.metric("Buy intensity", f"{intensity*100:.0f}%")

p1, p2, p3, p4 = st.columns(4)
p1.metric("Further drop ≤15% / 12W", f"{probs['deeper12']:.0f}%" if probs["deeper12"] is not None else "N/A")
p2.metric("Recovery ≥20% / 26W", f"{probs['recover26']:.0f}%" if probs["recover26"] is not None else "N/A")
p3.metric("Reclaim 50W / 26W", f"{probs['reclaim50_26']:.0f}%" if probs["reclaim50_26"] is not None else "N/A")
p4.metric("Historical analogs", str(probs["n"]))

st.caption(
    f"**How to read the probabilities:** they are empirical BTC history, not a forecast. "
    f"Each historical week is classified using only information known at that time, then we measure what happened afterward. "
    f"Current analog set: **{probs['level']}**, n={probs['n']}. More observations = more confidence; few analogs = use cautiously."
)

d50 = pct_distance(btc_price, btc_sma50w)
d200 = pct_distance(btc_price, btc_ema200w)
fib_text = f"Fib {fib['nearest']} (${fib['levels'][fib['nearest']]:,.0f})" if fib.get("nearest") else "Fib unavailable"
streak_text = f"{abs(btc_streak)} {'green' if btc_streak > 0 else 'red'} week(s)" if btc_streak else "mixed weeks"
recommendation = (
    f"**{intensity_label}.** {btc_regime.title()} · {streak_text} · {maturity['phase'].title()} · {fib_text}. "
    f"{('Below 200W EMA: higher BTC allocation and 2-loop eligibility.' if below_200w else 'Above 200W EMA: keep leverage restrained; 2-loop mode is not active.')}"
)
if btc_streak >= 4:
    recommendation += " **Four+ green weeks: do not blindly execute the scheduled buy; reduce/defer it unless price is in a strong value zone.**"
if maturity["phase"] == "EARLY DROP":
    recommendation += " **The decline appears early: preserve dry powder rather than treating the first large drop as the bottom.**"
st.info(recommendation)

with st.expander("🌊 Elliott Wave lens + interpretation", expanded=False):
    if btc_chart is not None:
        # A transparent wave proxy: recent swing high/low plus current trend position.
        if btc_regime == "BEAR MARKET" and below_200w:
            wave_label = "Possible Wave C / late corrective phase"
            wave_note = "Price is below both long-term trend references. This can be a late correction zone, but Elliott labeling is subjective and should be confirmed by price structure rather than assumed."
        elif btc_regime == "BEAR MARKET":
            wave_label = "Possible ABC correction still developing"
            wave_note = "Below 50W but above 200W: the decline may still have room. This is exactly why the engine avoids max leverage before the 200W value zone."
        else:
            wave_label = "Possible impulsive advance / continuation"
            wave_note = "Above 50W: trend is constructive, but extension risk and green-week streaks matter more for entry timing than wave labels."
        st.markdown(f"**Structure:** {wave_label}")
        st.caption(f"**Interpretation:** {wave_note} Elliott Wave is a secondary lens; the empirical probabilities and price-location rules have priority for sizing.")

with st.expander("📐 Price-location details", expanded=False):
    fcols = st.columns(6)
    for col, (label, level) in zip(fcols, fib.get("levels", {}).items()):
        col.metric(label, f"${level:,.0f}")
    st.caption(
        f"Fibonacci range uses the recent ~208-week high/low as a transparent cycle context. "
        f"Nearest level: **{fib.get('nearest','N/A')}**. This is a location reference, not a standalone signal. "
        f"50W distance: **{d50:+.1f}%** · 200W distance: **{d200:+.1f}%**."
    )

# ---------- CAPITAL ENGINE ----------
cached_aave = st.session_state.get("cl_aave")
st.subheader("Capital Engine")
cap1, cap2, cap3, cap4, cap5 = st.columns(5)
spare_capital = cap1.number_input("Cash available ($)", min_value=0.0, value=0.0, step=100.0, key="spare_capital_v5")
monthly_capital = cap2.number_input("Monthly recurring ($)", min_value=0.0, value=0.0, step=25.0, key="monthly_capital_v5")
dca_frequency = cap3.selectbox("Buy frequency", ["Weekly", "Bi-weekly"], key="dca_frequency_v5")
default_end = date.today() + timedelta(days=90)
schedule_end = cap4.date_input("Deploy cash until", value=default_end, min_value=date.today(), key="schedule_end_v5")
loop_utilization = cap5.number_input("Loop risk budget %", min_value=0.0, max_value=100.0, value=50.0, step=5.0, key="loop_utilization_v5") / 100.0

interval_days = 7 if dca_frequency == "Weekly" else 14
days_remaining = max(1, (schedule_end - date.today()).days)
remaining_purchases = max(1, math.ceil(days_remaining / interval_days))
current_cash_pace = spare_capital / remaining_purchases if spare_capital > 0 else 0.0
recurring_per_purchase = monthly_capital * (interval_days / 30.4375)
base_purchase = current_cash_pace + recurring_per_purchase

# Market-responsive execution. Current cash can be spent only up to what is actually available.
scheduled_from_cash = current_cash_pace * intensity
scheduled_recurring = recurring_per_purchase * intensity
suggested_deploy = scheduled_from_cash + scheduled_recurring
suggested_deploy = min(suggested_deploy, spare_capital + recurring_per_purchase)
if intensity <= 0.10:
    suggested_deploy = 0.0
elif btc_streak >= 4:
    # Keep the scheduled amount as dry powder when the market has already rallied for 4+ weeks.
    suggested_deploy *= 0.35

# Below 200W: target 70% BTC allocation of the currently available cash pool.
deep_value_cash_target = spare_capital * 0.70 if below_200w else None

available_display = spare_capital
b1, b2, b3, b4, b5 = st.columns(5)
b1.metric("Cash now", f"${available_display:,.0f}")
b2.metric("Cash pace / buy", f"${current_cash_pace:,.0f}")
b3.metric("Recurring / buy", f"${recurring_per_purchase:,.0f}")
b4.metric("Suggested BTC buy", f"${suggested_deploy:,.0f}")
b5.metric("Purchases remaining", str(remaining_purchases))

st.caption(
    f"**Pacing:** current cash is spread over the remaining {remaining_purchases} {dca_frequency.lower()} purchase(s). "
    f"Monthly recurring capital is treated as a future flow, not as cash already available. "
    f"If you manually reduce Cash now after a spot buy, the pace automatically falls. "
    f"Below 200W EMA, the strategic target is **70% of current cash in BTC** (currently ${deep_value_cash_target:,.0f})."
    if deep_value_cash_target is not None else
    f"**Pacing:** current cash is spread over the remaining {remaining_purchases} {dca_frequency.lower()} purchase(s). Monthly recurring capital is a separate future flow."
)

if btc_streak >= 4:
    st.warning(f"⏸️ **4+ green weekly candles:** scheduled BTC buy is reduced to 35% of the calculated amount ({'${:,.0f}'.format(suggested_deploy)}). Keep the unused cash as dry powder and reassess after a red/flat week or a better value location.")
elif intensity <= 0.10:
    st.warning("🛑 **STOP buying for now:** price is too extended relative to the 50W structure. Preserve cash rather than chasing.")
elif maturity["phase"] == "EARLY DROP":
    st.info("🟡 **Early decline:** the engine intentionally buys less even though price is falling. The objective is to avoid spending most of the reserve on the way from a major high toward a possible bear-market low.")
elif below_200w:
    st.success(f"🔥 **Below 200W EMA:** deep-value mode. Target BTC allocation from current cash is **70% (${deep_value_cash_target:,.0f})**. Two loops are permitted, but the second loop still requires the first loop to leave enough HF/LTV safety and the market-risk budget to justify it.")
else:
    st.success(f"🟢 **{intensity_label}:** scheduled buying is scaled by price location, trend, weekly streak and historical outcomes rather than buying the same amount every week.")

# ---------- LOOP ENGINE ----------
hf = None
loop_rows = []
max_safe_borrow = 0.0
market_borrow_budget = suggested_deploy * risk_profile["borrow_multiple"] * loop_utilization
if cached_aave and not cached_aave.get("error"):
    collateral = cached_aave["total_collateral"]
    debt = cached_aave["total_debt"]
    liq_threshold = cached_aave["liq_threshold_pct"] / 100.0
    ltv = cached_aave["ltv_pct"] / 100.0
    hf = cached_aave["health_factor"]
    if debt > 0 and hf < 1.5:
        market_borrow_budget = 0.0
        recommended_loops = 0
    elif suggested_deploy > 0 and recommended_loops > 0:
        max_safe_borrow = max_borrow_for_loop(collateral, debt, suggested_deploy, liq_threshold, ltv, target_hf)
        total_borrow_budget = min(max_safe_borrow, market_borrow_budget)
        loop_rows = simulate_sequential_loops(collateral, debt, liq_threshold, ltv, target_hf, suggested_deploy, recommended_loops, total_borrow_budget)

st.subheader("Loop Engine")
if cached_aave and not cached_aave.get("error") and suggested_deploy > 0 and loop_rows:
    total_borrow = sum(r["borrow"] for r in loop_rows)
    total_added = sum(r["total_deployed"] for r in loop_rows)
    final_hf = loop_rows[-1]["hf_after"]
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Loop mode", risk_profile["mode"])
    l2.metric("AAVE max borrow", f"${max_safe_borrow:,.0f}")
    l3.metric("Recommended borrow", f"${total_borrow:,.0f}")
    l4.metric("Final HF", "∞" if final_hf == float("inf") else f"{final_hf:.2f}")
    st.markdown(
        f"**One loop:** buy **${loop_rows[0]['fresh']:,.0f} BTC** → supply it → borrow **${loop_rows[0]['borrow']:,.0f}** → buy BTC with the borrow → supply the new BTC. "
        f"A second loop, when enabled, starts from the new collateral/debt state and recalculates its own HF/LTV ceiling."
    )
    with st.expander("Loop calculations — show every number", expanded=False):
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("Fresh BTC", f"${suggested_deploy:,.0f}")
        q2.metric("Market-risk budget", f"${market_borrow_budget:,.0f}")
        q3.metric("Borrow multiple", f"{risk_profile['borrow_multiple']:.2f}x")
        q4.metric("Risk budget used", f"{loop_utilization:.0%}")
        q5.metric("Target HF", f"{target_hf:.2f}")
        table = []
        for r in loop_rows:
            table.append({
                "Loop": r["loop"],
                "Fresh BTC": f"${r['fresh']:,.0f}",
                "AAVE max": f"${r['max_borrow']:,.0f}",
                "Recommended borrow": f"${r['borrow']:,.0f}",
                "BTC exposure added": f"${r['total_deployed']:,.0f}",
                "HF after": "∞" if r["hf_after"] == float("inf") else f"{r['hf_after']:.2f}",
                "LTV after": f"{r['ltv_after']*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
        st.caption(
            "**Why the recommendation can be far below AAVE headroom:** AAVE headroom answers 'what is technically borrowable while maintaining the target HF/LTV?' The market-risk budget answers 'how much leverage is appropriate for the current market?' The engine always uses the smaller number."
        )
        if below_200w:
            st.info("**2-loop permission:** below the 200W EMA makes the second loop eligible. It is not an automatic instruction to use it. Weak historical recovery odds, an early decline, four green weeks, or a poor HF can still reduce or disable borrowing.")
elif cached_aave and not cached_aave.get("error") and hf is not None and hf < 1.5:
    st.error("🛑 **Leverage disabled:** current Health Factor is below 1.50. Do not add new debt until the safety buffer improves.")
elif suggested_deploy <= 0:
    st.caption("No fresh BTC allocation is currently recommended, so there is no new-capital loop to calculate.")
else:
    st.caption("Load your AAVE position below to calculate the hard HF/LTV ceiling. The market engine will still cap borrowing well below that ceiling when risk is high.")

# ---------- COMPACT CHART ----------
st.subheader("Weekly Price Action")
chart_symbol = "BTC"
if use_eth:
    chart_symbol = st.radio("Chart", ["BTC", "ETH"], horizontal=True, key="chart_symbol_v3")
chart_df = fetch_weekly_chart_data(f"{chart_symbol}-USD")
if chart_df is not None and not chart_df.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Close"], name=f"{chart_symbol} price", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA50W"], name="50W MA", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA200W"], name="200W EMA", line=dict(width=1, dash="dot")))
    fig.update_layout(height=300, margin=dict(l=5, r=5, t=10, b=5), yaxis_title="USD", legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------- RETURN TARGET ----------
st.subheader("🎯 Return Target & Price Scenarios")
r1, r2, r3 = st.columns(3)
desired_multiple = r1.number_input("Desired return (x)", min_value=1.0, value=3.0, step=0.1, key="desired_multiple_v3")
starting_investment = r2.number_input("Starting investment ($)", min_value=0.0, value=10000.0, step=500.0, key="starting_investment_v3")
scenario_horizon = r3.number_input("Scenario horizon (years)", min_value=1.0, value=3.0, step=1.0, key="scenario_horizon_v3")
if btc_price:
    target_price = btc_price * desired_multiple
    # Scenarios are explicitly labeled as stress/sensitivity levels, not forecasts.
    bear_price = btc_price * (0.65 if (probs.get("deeper12") or 0) >= 45 else 0.75)
    base_price = btc_price * (1.35 if btc_regime == "BEAR MARKET" else 1.50)
    bull_price = btc_price * (2.50 if btc_regime == "BEAR MARKET" else 3.00)
    exposure_multiple = 1.0
    if starting_investment > 0 and loop_rows:
        exposure_multiple += sum(r["borrow"] for r in loop_rows) / starting_investment
    scenario_rows = [("🔴 Bear", bear_price), ("🟡 Base", base_price), ("🎯 Target", target_price), ("🟢 Bull", bull_price)]
    cols = st.columns(4)
    for col, (label, price) in zip(cols, scenario_rows):
        implied = price / btc_price
        value = starting_investment * implied * exposure_multiple
        ret = value / starting_investment - 1 if starting_investment else 0
        col.metric(label, f"${price:,.0f}", f"{ret:+.0%}")
    st.caption(
        f"A **{desired_multiple:.1f}x price target** means BTC ≈ **${target_price:,.0f}** from today's ${btc_price:,.0f}. "
        f"Bear/Base/Bull are scenario sensitivities; they are not empirical price forecasts. The displayed return uses the currently recommended loop exposure only and does not model interest, slippage, taxes or liquidation."
    )

# ---------- AAVE POSITION ----------
with st.expander("AAVE position & risk monitor", expanded=False):
    wallet_address = st.text_input("Wallet Address", placeholder="0x...", key="aave_wallet_v3").strip()
    rpc_url = st.text_input("Base RPC URL", value="https://mainnet.base.org", key="aave_rpc_v3").strip()
    fetch_clicked = st.button("Fetch AAVE Position", type="primary", key="fetch_aave_v3")
    if fetch_clicked:
        if not wallet_address:
            st.error("Enter a wallet address first.")
        else:
            with st.spinner("Reading AAVE position..."):
                aave_result = fetch_aave_account_data(wallet_address, rpc_url)
            if aave_result.get("error"):
                st.session_state["cl_aave"] = None
                st.error(f"Could not fetch AAVE data: {aave_result['error']}")
            else:
                st.session_state["cl_aave"] = aave_result
                st.session_state["cl_wallet"] = wallet_address
                st.rerun()
    cached_aave = st.session_state.get("cl_aave")
    if cached_aave and not cached_aave.get("error"):
        aave = cached_aave
        collateral = aave["total_collateral"]
        debt = aave["total_debt"]
        hf = aave["health_factor"]
        liq_thresh = aave["liq_threshold_pct"] / 100.0
        drop_to_liq_pct = max(0.0, (1 - debt / (collateral * liq_thresh)) * 100) if debt > 0 and collateral > 0 else 0.0
        headroom = max(0.0, (collateral * liq_thresh / target_hf) - debt) if debt > 0 and collateral > 0 else 0.0
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Collateral", f"${collateral:,.0f}")
        a2.metric("Debt", f"${debt:,.0f}")
        a3.metric("LTV", f"{aave['ltv_pct']:.1f}%")
        a4.metric("Health Factor", "∞" if debt == 0 else f"{hf:.2f}")
        a5.metric("Liquidation buffer", f"{drop_to_liq_pct:.1f}%")
        if debt > 0 and collateral > 0:
            stress_rows = []
            for drop in [10, 20, 30, 50]:
                new_collateral = collateral * (1 - drop / 100.0)
                new_hf = (new_collateral * liq_thresh) / debt
                stress_rows.append({"Collateral drop": f"-{drop}%", "HF": round(new_hf, 2), "Status": "SAFE" if new_hf >= 1.0 else "LIQUIDATION RISK"})
            st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)
            st.caption(f"Additional borrow headroom at HF ≥ {target_hf:.2f}: **${headroom:,.0f}**. This is a safety ceiling, not a recommendation.")
    else:
        st.caption("No AAVE position loaded.")

# ---------- HOW TO USE ----------
with st.expander("📘 How to use each box", expanded=False):
    st.markdown(
        """
**Market Overview**
- **50W MA / regime:** the simple bull/bear classifier. Below 50W = bear market; above 50W = bull market.
- **200W EMA:** the long-cycle value boundary. Below it, BTC enters deep-value mode and a second loop becomes eligible.
- **Buy intensity:** tells you how much of the scheduled purchase to use. It is an allocation score, not a probability.

**Historical probabilities**
- **Further drop ≤15% / 12W:** percentage of comparable historical weeks where BTC fell at least another 15% at some point in the next 12 weeks.
- **Recovery ≥20% / 26W:** percentage of comparable weeks where BTC gained at least 20% at some point in the next 26 weeks.
- **Reclaim 50W / 26W:** percentage of comparable weeks where BTC crossed back above its 50W MA within 26 weeks.
- **Historical analogs:** sample size. Treat a result based on 12 observations very differently from one based on 80.

**Timing / price location**
- **Green/red streak:** four or more green weeks means don't blindly DCA into strength. Four or more red weeks can justify a small tactical increase, but does not prove the bottom.
- **Decline maturity:** distinguishes an early fall from a mature bear move. This prevents the engine from spending most of the reserve simply because BTC started falling from a major high.
- **Fibonacci:** shows where price sits inside the recent long-term range. It is a confluence tool, not a buy trigger.
- **Elliott:** a secondary structural interpretation. It is subjective, so it should agree with price location and historical evidence before affecting sizing.

**Capital Engine**
- **Cash available:** money you actually have available now. Update it after spot purchases.
- **Monthly recurring:** future contributions. It does not pretend future money is already in your wallet.
- **Weekly / bi-weekly:** controls how quickly your current cash is paced into the market.
- **Deploy cash until:** the date used to spread the current cash reserve across scheduled purchases.
- **Suggested BTC buy:** the scheduled amount after market conditions, green-week timing and price location are applied.
- **Below 200W:** the strategic target becomes 70% of current cash allocated to BTC, while still allowing gradual deployment rather than forcing a one-day purchase.

**Loop Engine**
- **AAVE max borrow:** the hard HF/LTV ceiling. Never treat this as a recommended borrow.
- **Recommended borrow:** the smaller of the AAVE safety ceiling and the market-risk budget.
- **One loop:** buy BTC with fresh money → supply BTC → borrow USDC → buy BTC → supply again.
- **Two loops:** only eligible below the 200W EMA. Loop 2 is recalculated from Loop 1's new collateral and debt state.
- **Target HF:** your minimum desired safety buffer for new borrowing.

**Return Target**
- Enter a desired multiple such as **3x**. The box converts that into the BTC price required to reach the multiple and shows bear/base/bull sensitivity scenarios.
        """
    )

with st.expander("Model notes / limitations", expanded=False):
    st.markdown(
        """
- The bull/bear regime is intentionally simple: price above 50W MA = bull; below = bear.
- The 200W EMA is a valuation/aggression boundary, not a bottom detector.
- Historical probabilities are empirical conditional statistics from BTC weekly history. They are not guarantees and are shown with sample size and the conditioning level used.
- Fibonacci uses a recent ~208-week high/low range. Automatic swing-point selection can be improved later if you want a stricter pivot algorithm.
- Elliott Wave is a subjective lens. The dashboard deliberately does not pretend that automated Elliott labels are objective facts.
- The engine does not fabricate on-chain metrics that are not connected to a live source. They can be added later as additional conditioning variables and backtested in the same framework.
- AAVE headroom is a hard safety ceiling; the market engine can recommend far less.
- Loop calculations are sequential: after each loop, collateral, debt, HF and LTV are recalculated.
- The return scenarios do not model borrow interest, slippage, taxes, liquidation path, BTC/USDC basis risk or protocol parameter changes.
- This dashboard is informational and does not execute trades, borrows or repayments.
        """
    )
