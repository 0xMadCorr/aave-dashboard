import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Crypto Leverage - AAVE (Base)", layout="wide")


# ============================================================
# DATA + CALCULATION HELPERS
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


def pct_distance(price, level):
    if price is None or level in (None, 0):
        return None
    return (price - level) / level * 100.0


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


# ============================================================
# WALLET ASSET BREAKDOWN (which assets are supplied / borrowed on AAVE)
# Addresses/signatures checked against bgd-labs/aave-address-book and aave-v3-origin.
# ============================================================
AAVE_BASE_ADDRESSES_PROVIDER = "0xe20fCBdBfFC4Dd138cE8b2E6FBb6CB49777ad64D"
STABLE_SYMBOLS = {"USDC", "USDBC", "USDT", "DAI", "GHO", "USDS", "SUSDS", "EURC", "LUSD", "CRVUSD"}


def _selector(signature):
    from web3 import Web3
    return "0x" + bytes(Web3.keccak(text=signature)[:4]).hex()


def _addr_word(addr):
    return addr.lower().replace("0x", "").rjust(64, "0")


def _word(hexstr, i):
    return hexstr[2:][64 * i: 64 * (i + 1)]


def _word_addr(hexstr, i):
    return "0x" + _word(hexstr, i)[-40:]


def _rpc_calls(rpc_url, calls):
    """Run many read-only eth_calls. Uses one batched HTTP request per 40 calls;
    falls back to one-by-one if the RPC does not accept batches.
    calls = [(to_address, data_hex)] -> list of hex results (None where a call failed)."""
    import requests
    out = [None] * len(calls)
    for start in range(0, len(calls), 40):
        chunk = calls[start:start + 40]
        payload = [
            {"jsonrpc": "2.0", "id": start + i, "method": "eth_call",
             "params": [{"to": to, "data": data}, "latest"]}
            for i, (to, data) in enumerate(chunk)
        ]
        done = False
        try:
            res = requests.post(rpc_url, json=payload, timeout=20).json()
            if isinstance(res, list):
                for item in res:
                    if isinstance(item, dict) and item.get("result") not in (None, "0x"):
                        out[int(item["id"])] = item["result"]
                done = True
        except Exception:
            pass
        if not done:
            for i, (to, data) in enumerate(chunk):
                try:
                    j = requests.post(
                        rpc_url,
                        json={"jsonrpc": "2.0", "id": 1, "method": "eth_call",
                              "params": [{"to": to, "data": data}, "latest"]},
                        timeout=20,
                    ).json()
                    if j.get("result") not in (None, "0x"):
                        out[start + i] = j["result"]
                except Exception:
                    pass
    return out


@st.cache_data(ttl=120, show_spinner=False)
def fetch_aave_breakdown(wallet_address: str, rpc_url: str):
    """Per-asset supplied / borrowed amounts and USD values for a wallet on AAVE v3 Base.
    Returns {"positions": [...], "error": None} or {"error": "..."}."""
    try:
        from web3 import Web3
        from eth_abi import decode, encode
    except ImportError:
        return {"error": "The 'web3' package is not installed."}
    try:
        user = Web3.to_checksum_address(wallet_address)
        r = _rpc_calls(rpc_url, [
            (AAVE_BASE_ADDRESSES_PROVIDER, _selector("getPoolDataProvider()")),
            (AAVE_BASE_ADDRESSES_PROVIDER, _selector("getPriceOracle()")),
        ])
        if r[0] is None or r[1] is None:
            return {"error": "could not read AAVE's address provider from the RPC"}
        data_provider, oracle = _word_addr(r[0], 0), _word_addr(r[1], 0)

        res = _rpc_calls(rpc_url, [(data_provider, _selector("getAllReservesTokens()"))])[0]
        if res is None:
            return {"error": "could not read the list of AAVE assets"}
        reserves = decode(["(string,address)[]"], bytes.fromhex(res[2:]))[0]
        symbols = [s for s, _ in reserves]
        assets = [a for _, a in reserves]

        price_call = _selector("getAssetsPrices(address[])") + encode(["address[]"], [assets]).hex()
        calls = [(oracle, price_call)]
        for a in assets:
            calls.append((data_provider, _selector("getReserveTokensAddresses(address)") + _addr_word(a)))
            calls.append((a, _selector("decimals()")))
        out = _rpc_calls(rpc_url, calls)
        if out[0] is None:
            return {"error": "could not read asset prices"}
        prices = decode(["uint256[]"], bytes.fromhex(out[0][2:]))[0]

        meta = []
        for i, a in enumerate(assets):
            tok, dec = out[1 + 2 * i], out[2 + 2 * i]
            if tok is None or dec is None:
                continue
            meta.append({
                "symbol": symbols[i], "address": a, "price": prices[i] / 1e8,
                "decimals": int(_word(dec, 0), 16),
                "a_token": _word_addr(tok, 0), "debt_token": _word_addr(tok, 2),
            })
        if not meta:
            return {"error": "no AAVE assets could be read"}

        bal = _selector("balanceOf(address)") + _addr_word(user)
        calls2 = []
        for m in meta:
            calls2.append((m["a_token"], bal))
            calls2.append((m["debt_token"], bal))
        out2 = _rpc_calls(rpc_url, calls2)

        positions = []
        for i, m in enumerate(meta):
            a_raw, d_raw = out2[2 * i], out2[2 * i + 1]
            if a_raw is None or d_raw is None:
                continue
            supplied = int(_word(a_raw, 0), 16) / 10 ** m["decimals"]
            debt = int(_word(d_raw, 0), 16) / 10 ** m["decimals"]
            if supplied <= 0 and debt <= 0:
                continue
            positions.append({
                "symbol": m["symbol"], "address": m["address"], "price": m["price"],
                "supplied": supplied, "supplied_usd": supplied * m["price"],
                "debt": debt, "debt_usd": debt * m["price"],
                "is_stable": m["symbol"].upper() in STABLE_SYMBOLS,
            })
        return {"positions": positions, "error": None}
    except Exception as e:
        return {"error": str(e)}


def liq_price(main_price, vol_value, stable_value, debt, lt):
    """Approx. price of the main collateral asset at which Health Factor reaches 1.
    Assumes all non-stablecoin collateral moves together with the main asset,
    stablecoin collateral stays flat, and the debt is a stablecoin."""
    if not main_price or debt <= 0 or lt <= 0:
        return None
    if vol_value <= 0:
        return 0.0
    factor = (debt / lt - stable_value) / vol_value
    return main_price * max(factor, 0.0)


def liq_text(value, debt):
    if debt <= 0:
        return "None (no debt)"
    if value is None:
        return "N/A"
    if value <= 0:
        return "None (stablecoins cover debt)"
    return f"${value:,.0f}"


def money(x, none_text="N/A"):
    return none_text if x is None else f"${x:,.0f}"


# ============================================================
# PAGE
# ============================================================
top_l, top_r = st.columns([6, 1])
top_l.title("Crypto Leverage (AAVE - Base)")
if top_r.button("↻ Refresh", key="refresh_v2"):
    fetch_aave_account_data.clear(); fetch_aave_breakdown.clear(); fetch_crypto_technicals.clear()
    fetch_weekly_chart_data.clear(); fetch_btc_probability_history.clear()
    st.rerun()
st.caption(
    "Informational tool only. Not financial advice. Leverage can lead to total loss and liquidation. "
    "Your wallet address is only used for a read-only lookup and is not stored. "
    "Light/dark mode: ⋮ menu (top right) → Settings → Theme."
)

# ---------- market data ----------
with st.spinner("Loading market data..."):
    btc = fetch_crypto_technicals("BTC-USD")
    eth = fetch_crypto_technicals("ETH-USD")
    btc_chart = fetch_weekly_chart_data("BTC-USD")

btc_price = btc.get("price")
btc_ema200w = btc.get("ema200w")
btc_streak = btc.get("trend_streak", 0)
btc_sma50w = None
if btc_chart is not None and not btc_chart.empty and pd.notna(btc_chart["SMA50W"].iloc[-1]):
    btc_sma50w = float(btc_chart["SMA50W"].iloc[-1])
market_ok = btc_price is not None and btc_sma50w is not None and btc_ema200w is not None

maturity = {"phase": "Unknown"}
intensity, intensity_label, btc_regime, below_200w, fib = 0.5, "N/A", "UNKNOWN", False, {}
if market_ok:
    maturity = decline_maturity(btc_chart["Close"], btc_sma50w)
    fib = fibonacci_context(btc_chart["Close"])
    probs = empirical_probabilities(
        build_current_probability_row(btc_price, btc_sma50w, btc_ema200w, btc_streak, maturity),
        fetch_btc_probability_history(),
    )
    intensity, intensity_label = buy_intensity(btc_price, btc_sma50w, btc_ema200w, fib, maturity, btc_streak, probs)
    btc_regime = market_regime(btc_price, btc_sma50w)
    below_200w = btc_price < btc_ema200w

# ============================================================
# 1. AAVE POSITION (main input)
# ============================================================
st.header("Your AAVE position")
w1, w2 = st.columns([3, 2])
wallet_address = w1.text_input("Wallet address", placeholder="0x...", key="aave_wallet_v2").strip()
rpc_url = w2.text_input("Base RPC URL", value="https://mainnet.base.org", key="aave_rpc_v2").strip()
if st.button("Fetch AAVE position", type="primary", key="fetch_aave_v2"):
    if not wallet_address:
        st.error("Enter a wallet address first.")
    else:
        with st.spinner("Reading AAVE position..."):
            aave_result = fetch_aave_account_data(wallet_address, rpc_url)
            if aave_result.get("error"):
                st.session_state["cl_aave"] = None
                st.session_state["cl_breakdown"] = None
            else:
                st.session_state["cl_aave"] = aave_result
                st.session_state["cl_breakdown"] = fetch_aave_breakdown(wallet_address, rpc_url)
        if aave_result.get("error"):
            st.error(f"Could not fetch AAVE data: {aave_result['error']}")
        else:
            st.rerun()

cached_aave = st.session_state.get("cl_aave")
position_loaded = bool(cached_aave and not cached_aave.get("error"))
breakdown = st.session_state.get("cl_breakdown") if position_loaded else None
breakdown_ok = bool(breakdown and not breakdown.get("error"))
positions = breakdown["positions"] if breakdown_ok else []
if position_loaded and breakdown and breakdown.get("error"):
    st.warning(f"Could not read your individual assets ({breakdown['error']}). Choose the collateral asset manually below.")

pos_collateral = cached_aave["total_collateral"] if position_loaded else 0.0
pos_debt = cached_aave["total_debt"] if position_loaded else 0.0
supplied_total = sum(p["supplied_usd"] for p in positions)
volatile_supplied = [p for p in positions if not p["is_stable"] and p["supplied_usd"] > 0]
auto_ok = position_loaded and pos_collateral > 0 and supplied_total > 0 and bool(volatile_supplied)

# --- main collateral asset: detected from the wallet, manual BTC/ETH only as a fallback ---
if auto_ok:
    main = max(volatile_supplied, key=lambda p: p["supplied_usd"])
    main_symbol, asset_price = main["symbol"], main["price"]
    vol_share = sum(p["supplied_usd"] for p in volatile_supplied) / supplied_total
    vol_value = pos_collateral * vol_share
    stable_value = pos_collateral - vol_value
else:
    stable_only = bool(position_loaded and breakdown_ok and positions and not volatile_supplied)
    if not position_loaded:
        label = "Asset you plan to buy / hold as collateral"
    elif stable_only:
        label = "Your collateral is all stablecoins. Asset you plan to buy"
    else:
        label = "Collateral asset (could not be detected from your wallet)"
    main_symbol = st.radio(label, ["BTC", "ETH"], horizontal=True, key="coll_asset_v2")
    asset_price = btc_price if main_symbol == "BTC" else eth.get("price")
    vol_value = 0.0 if stable_only else pos_collateral
    stable_value = pos_collateral if stable_only else 0.0

stress_box = None
if position_loaded:
    pos_hf = cached_aave["health_factor"]
    pos_lt = cached_aave["liq_threshold_pct"] / 100.0
    drop_to_liq = (
        max(0.0, (1 - pos_debt / (pos_collateral * pos_lt)) * 100)
        if pos_debt > 0 and pos_collateral > 0 and pos_lt > 0 else 0.0
    )
    a1, a2, a3, a4, a5, a6 = st.columns(6)
    a1.metric("Collateral", f"${pos_collateral:,.0f}")
    a2.metric("Debt", f"${pos_debt:,.0f}")
    a3.metric("LTV", f"{cached_aave['ltv_pct']:.1f}%")
    a4.metric("Health Factor", "∞" if pos_debt == 0 else f"{pos_hf:.2f}")
    a5.metric("Liquidation buffer", f"{drop_to_liq:.1f}%")
    a6.metric(
        f"Liquidation price ({main_symbol})",
        liq_text(liq_price(asset_price, vol_value, stable_value, pos_debt, pos_lt), pos_debt),
    )

    if positions:
        rows = []
        for p in sorted(positions, key=lambda p: -(p["supplied_usd"] + p["debt_usd"])):
            rows.append({
                "Asset": p["symbol"],
                "Supplied": f"{p['supplied']:,.4f}".rstrip("0").rstrip(".") if p["supplied"] else "-",
                "Supplied value": money(p["supplied_usd"]) if p["supplied"] else "-",
                "Share of supplied": f"{p['supplied_usd'] / supplied_total * 100:.0f}%" if supplied_total and p["supplied"] else "-",
                "Borrowed value": money(p["debt_usd"]) if p["debt"] else "-",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if auto_ok:
            st.caption(
                f"**Main collateral asset detected: {main_symbol}** (your largest non-stablecoin collateral). "
                "Liquidation price is an estimate: all your non-stablecoin collateral is assumed to fall together, "
                "stablecoin collateral stays flat, and the debt is treated as a stablecoin."
            )
        if supplied_total > pos_collateral * 1.03 > 0:
            st.caption(
                "Some supplied assets may not be enabled as collateral on AAVE, so the shares above are scaled to AAVE's collateral total."
            )
        volatile_debt = sum(p["debt_usd"] for p in positions if not p["is_stable"])
        if pos_debt > 0 and volatile_debt > 0.05 * pos_debt:
            st.warning("You also borrow a non-stablecoin asset. Its price moves are not included in the liquidation estimate.")
    stress_box = st.container()
else:
    st.caption("Enter your wallet address and press **Fetch AAVE position**.")

# ============================================================
# 2. MONTHLY INVESTMENT + SAFE LOOP
# ============================================================
st.header("This month's investment & safe loop")
i1, i2 = st.columns(2)
invest = i1.number_input("Amount to invest this month ($)", min_value=0.0, value=0.0, step=100.0, key="invest_v2")
hf_keep = i2.number_input("Health factor to keep", min_value=1.1, max_value=5.0, value=1.8, step=0.1, key="hf_keep_v2")

if position_loaded and pos_collateral > 0:
    C, D = pos_collateral, pos_debt
    L, V = cached_aave["liq_threshold_pct"] / 100.0, cached_aave["ltv_pct"] / 100.0
    assumed = False
else:
    st.info(
        "No AAVE position with collateral is loaded, so this treats you as a brand-new position. "
        "The two values below are **assumptions** - check the real numbers for your collateral asset on AAVE."
    )
    ac1, ac2 = st.columns(2)
    L = ac1.number_input("Liquidation threshold %", min_value=1.0, max_value=95.0, value=78.0, step=1.0, key="assumed_lt_v2") / 100.0
    V = ac2.number_input("Max LTV %", min_value=1.0, max_value=95.0, value=73.0, step=1.0, key="assumed_ltv_v2") / 100.0
    C, D = 0.0, 0.0
    assumed = True

C2, D2, vol2, loop_done = C, D, vol_value, False
if C + invest <= 0:
    st.caption("Enter the amount you plan to invest to see the safe loop.")
else:
    max_b = max_borrow_for_loop(C, D, invest, L, V, hf_keep)
    vol2 = vol_value + invest + max_b          # new purchases go into the main asset
    C2, D2, loop_done = stable_value + vol2, D + max_b, True
    hf_after = float("inf") if D2 <= 0 else C2 * L / D2
    ltv_after = D2 / C2 if C2 > 0 else 0.0
    liq_now = liq_price(asset_price, vol_value, stable_value, D, L)
    liq_after = liq_price(asset_price, vol2, stable_value, D2, L)

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Safe borrow", f"${max_b:,.0f}")
    r2.metric(f"Total {main_symbol} to buy", f"${invest + max_b:,.0f}")
    r3.metric("Collateral after", f"${C2:,.0f}")
    r4.metric("Debt after", f"${D2:,.0f}")
    r5.metric("Health Factor after", "∞" if hf_after == float("inf") else f"{hf_after:.2f}")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(f"{main_symbol} price now", money(asset_price))
    s2.metric("Liquidation price now", liq_text(liq_now, D))
    s3.metric("Liquidation price after loop", liq_text(liq_after, D2))
    s4.metric("LTV after", f"{ltv_after*100:.1f}%")

    if D > 0 and C * L / D < hf_keep and max_b <= 0:
        st.warning(
            f"Your current Health Factor ({C * L / D:.2f}) is already below the {hf_keep:.2f} you want to keep, "
            "so there is no safe new borrowing. Adding cash as collateral or repaying debt would raise it."
        )
    elif max_b > 0:
        st.markdown(
            f"**How to do it:** buy **${invest:,.0f} {main_symbol}** and supply it → borrow **${max_b:,.0f}** (stablecoin) → "
            f"buy more {main_symbol} with it → supply that too. You can borrow in several rounds if you prefer; "
            f"the end result is the same. Your Health Factor ends at about **{hf_after:.2f}**."
        )
    else:
        st.info("With this amount and Health Factor there is no extra borrowing room. The cash would simply be added as collateral.")
    st.caption(
        "This is the **maximum** borrow that keeps your chosen Health Factor, not a recommendation. A higher Health Factor is safer. "
        "It ignores borrow interest, fees, slippage and taxes."
        + (" Numbers use the assumed threshold values above." if assumed else "")
    )

    cautions = []
    if market_ok:
        if intensity <= 0.10:
            cautions.append("price is far above its 50-week average (too extended)")
        if btc_streak >= 4:
            cautions.append(f"{btc_streak} green weeks in a row")
        if maturity["phase"] == "EARLY DROP":
            cautions.append("the market looks to be in an early drop")
    if cautions:
        st.warning("Market caution: " + "; ".join(cautions) + ". Consider borrowing less than the maximum, or waiting.")

# ---------- collateral drop stress test (rendered up in the position section) ----------
if stress_box is not None and (D > 0 or D2 > 0):
    with stress_box:
        st.markdown("**Collateral drop stress test** (non-stablecoin collateral falls, stablecoins stay flat)")
        rows = []
        for drop in [10, 20, 30, 50]:
            f = 1 - drop / 100.0
            hf_n = ((vol_value * f + stable_value) * L) / D if D > 0 else None
            row = {
                "Collateral drop": f"-{drop}%",
                f"{main_symbol} price": money(asset_price * f) if asset_price else "N/A",
                "HF now": f"{hf_n:.2f}" if hf_n is not None else "∞",
            }
            last_hf = hf_n
            if loop_done:
                hf_a = ((vol2 * f + stable_value) * L) / D2 if D2 > 0 else None
                row["HF after loop"] = f"{hf_a:.2f}" if hf_a is not None else "∞"
                last_hf = hf_a
            row["Status"] = "LIQUIDATION RISK" if (last_hf is not None and last_hf < 1.0) else "SAFE"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ============================================================
# 3. MARKET OVERVIEW
# ============================================================
st.header("Market overview")
if not market_ok:
    st.warning("Could not load BTC market data from Yahoo Finance (it may be rate-limiting). Wait a minute and press Refresh.")
else:
    regime_icon = "🟢" if btc_regime == "BULL MARKET" else "🔴"
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Regime", f"{regime_icon} {btc_regime}")
    h2.metric("BTC", f"${btc_price:,.0f}")
    h3.metric("50W MA", f"${btc_sma50w:,.0f}", delta=f"{(pct_distance(btc_price, btc_sma50w) or 0):+.1f}%")
    h4.metric("200W EMA", f"${btc_ema200w:,.0f}", delta=f"{(pct_distance(btc_price, btc_ema200w) or 0):+.1f}%")
    h5.metric("Buy intensity", f"{intensity*100:.0f}%")

    fib_text = f"Fib {fib['nearest']} (${fib['levels'][fib['nearest']]:,.0f})" if fib.get("nearest") else "Fib unavailable"
    streak_text = f"{abs(btc_streak)} {'green' if btc_streak > 0 else 'red'} week(s)" if btc_streak else "mixed weeks"
    recommendation = (
        f"**{intensity_label}.** {btc_regime.title()} · {streak_text} · {maturity['phase'].title()} · {fib_text}. "
        + ("Below the 200W EMA: deep-value zone." if below_200w else "Above the 200W EMA.")
    )
    if btc_streak >= 4:
        recommendation += " **Four+ green weeks: don't chase; consider reducing or deferring this month's buy.**"
    if maturity["phase"] == "EARLY DROP":
        recommendation += " **The decline looks early: preserve some cash rather than assuming the first big drop is the bottom.**"
    st.info(recommendation)

# ============================================================
# 4. WEEKLY PRICE ACTION  (no fixed colour template, so it follows light/dark mode)
# ============================================================
st.header("Weekly price action")
chart_symbol = st.radio("Chart", ["BTC", "ETH"], horizontal=True, key="chart_symbol_v2")
chart_df = fetch_weekly_chart_data(f"{chart_symbol}-USD")
if chart_df is not None and not chart_df.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Close"], name=f"{chart_symbol} price", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA50W"], name="50W MA", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA200W"], name="200W EMA", line=dict(width=1, dash="dot")))
    fig.update_layout(
        height=300, margin=dict(l=5, r=5, t=10, b=5), yaxis_title="USD",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
else:
    st.caption("Chart data is not available right now.")
