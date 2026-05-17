"""
Scarica i 7 major cross forex da yfinance, calcola indicatori tecnici
e salva data/forex_latest.json (per la pagina HTML) e CSV individuali
(per Streamlit e backtest).

Uso:
    python data_updater.py
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

CROSSES = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
}

# (base_currency, quote_currency)
PAIRS = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "USDCHF": ("USD", "CHF"),
    "USDCAD": ("USD", "CAD"),
    "AUDUSD": ("AUD", "USD"),
    "NZDUSD": ("NZD", "USD"),
}


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def safe_float(val, decimals: int = 5):
    try:
        v = float(val)
        if pd.isna(v) or np.isinf(v):
            return None
        return round(v, decimals)
    except Exception:
        return None


def download_all() -> dict:
    os.makedirs("data", exist_ok=True)

    strength_raw = {c: 0.0 for c in ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]}
    result = {}

    for name, ticker in CROSSES.items():
        print(f"  Downloading {name} ({ticker})...")
        df = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=True)

        if df.empty:
            print(f"  WARNING: nessun dato per {name}")
            continue

        # Flatten multi-index columns (yfinance >= 0.2.x può restituire multi-index)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Normalizza colonne e pulizia
        df = df.loc[:, ~df.columns.duplicated()]
        needed = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
        if "Volume" not in df.columns:
            df["Volume"] = 0
        df = df[needed + ["Volume"]].copy().dropna(subset=needed)

        # Indicatori tecnici
        df["RSI"]    = compute_rsi(df["Close"])
        df["SMA20"]  = df["Close"].rolling(20).mean()
        df["SMA50"]  = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        df["EMA20"]  = df["Close"].ewm(span=20, adjust=False).mean()
        df["ATR"]    = compute_atr(df["High"], df["Low"], df["Close"])
        df["Pct"]    = df["Close"].pct_change() * 100

        # Salva CSV
        df.index = pd.to_datetime(df.index)
        df.to_csv(f"data/{name}_daily.csv")

        last = df.iloc[-1]
        c  = safe_float(last["Close"])
        s20 = safe_float(last["SMA20"])
        s50 = safe_float(last["SMA50"])

        if c and s20 and s50:
            if c > s20 > s50:
                trend = "LONG"
            elif c < s20 < s50:
                trend = "SHORT"
            else:
                trend = "NEUTRAL"
        else:
            trend = "NEUTRAL"

        # Currency strength: media ultimi 5 giorni
        recent_pct = float(df["Pct"].tail(5).mean())
        base, quote = PAIRS[name]
        strength_raw[base] += recent_pct
        strength_raw[quote] -= recent_pct

        # Ultimi 60 close per sparkline
        hist = df.tail(60)
        history       = [safe_float(v) for v in hist["Close"]]
        history_dates = [str(d.date()) for d in hist.index]

        result[name] = {
            "last":          c,
            "open":          safe_float(last["Open"]),
            "high":          safe_float(last["High"]),
            "low":           safe_float(last["Low"]),
            "change_pct":    safe_float(last["Pct"], 4),
            "rsi":           safe_float(last["RSI"], 2),
            "sma20":         s20,
            "sma50":         s50,
            "sma200":        safe_float(last["SMA200"]),
            "ema20":         safe_float(last["EMA20"]),
            "atr":           safe_float(last["ATR"]),
            "trend":         trend,
            "history":       history,
            "history_dates": history_dates,
        }

    # Normalizza strength 0-100
    vals = list(strength_raw.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi != lo else 1.0
    strength_norm = {k: round((v - lo) / span * 100, 2) for k, v in strength_raw.items()}

    output = {
        "last_update": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "crosses":     result,
        "strength":    strength_norm,
    }

    with open("data/forex_latest.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFatto. {len(result)}/7 cross aggiornati -> data/forex_latest.json")
    return output


def run_backtests() -> dict:
    """Esegue EMA e RSI backtest su tutti i cross e salva data/backtest_results.json."""
    try:
        from backtesting import Backtest, Strategy
        from backtesting.lib import crossover
    except ImportError:
        print("  backtesting non installato, skip backtest")
        return {}

    def _ema(arr, n):
        return pd.Series(arr).ewm(span=n, adjust=False).mean()

    def _rsi(arr, n=14):
        delta = pd.Series(arr).diff()
        g = delta.clip(lower=0).ewm(com=n - 1, min_periods=n).mean()
        l = (-delta.clip(upper=0)).ewm(com=n - 1, min_periods=n).mean()
        return 100 - 100 / (1 + g / l)

    class EMACross(Strategy):
        def init(self):
            self.f = self.I(_ema, self.data.Close, 20)
            self.s = self.I(_ema, self.data.Close, 50)
        def next(self):
            if crossover(self.f, self.s):
                self.position.close(); self.buy()
            elif crossover(self.s, self.f):
                self.position.close(); self.sell()

    class RSIStrat(Strategy):
        def init(self):
            self.rsi = self.I(_rsi, self.data.Close, 14)
        def next(self):
            if self.rsi[-1] < 30 and not self.position.is_long:
                self.position.close(); self.buy()
            elif self.rsi[-1] > 70 and not self.position.is_short:
                self.position.close(); self.sell()

    results = {"ema": {}, "rsi": {}}

    for name in CROSSES:
        path = f"data/{name}_daily.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].tail(504).dropna()
        if len(df) < 60:
            continue

        for strat_name, StratClass in [("ema", EMACross), ("rsi", RSIStrat)]:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bt    = Backtest(df, StratClass, cash=10_000, commission=0.0002, exclusive_orders=True)
                    stats = bt.run()

                results[strat_name][name] = {
                    "return_pct":    round(float(stats["Return [%]"]), 2),
                    "buy_hold_pct":  round(float(stats["Buy & Hold Return [%]"]), 2),
                    "max_dd_pct":    round(float(stats["Max. Drawdown [%]"]), 2),
                    "n_trades":      int(stats["# Trades"]),
                    "win_rate":      round(float(stats["Win Rate [%]"]), 1),
                    "sharpe":        round(float(stats["Sharpe Ratio"]), 2) if stats.get("Sharpe Ratio") else None,
                    "profit_factor": round(float(stats.get("Profit Factor") or 0), 2),
                }
                print(f"  BT {name}/{strat_name}: {results[strat_name][name]['return_pct']:+.2f}%")
            except Exception as exc:
                print(f"  BT {name}/{strat_name}: ERRORE - {exc}")

    output = {
        "last_update":    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "period_days":    504,
        "commission_pct": 0.02,
        "strategies":     results,
    }
    with open("data/backtest_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Backtest salvati -> data/backtest_results.json")
    return output


def build_strength_json(window: int = 10, threshold: float = 15.0) -> dict:
    """Calcola forza valutaria storica, segnali operativi e backtest forza."""
    os.makedirs("data", exist_ok=True)

    closes = {}
    for name in CROSSES:
        path = f"data/{name}_daily.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            closes[name] = df["Close"]

    if not closes:
        print("  Nessun CSV trovato.")
        return {}

    price_df = pd.DataFrame(closes).dropna()
    pct      = price_df.pct_change(window) * 100

    currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]

    # Forza grezza: ogni cross contribuisce alla coppia base/quote
    raw   = pd.DataFrame(0.0, index=pct.index, columns=currencies)
    count = {c: 0 for c in currencies}
    for name, (base, quote) in PAIRS.items():
        if name not in pct.columns:
            continue
        raw[base]  += pct[name]
        raw[quote] -= pct[name]
        count[base]  += 1
        count[quote] += 1
    for c in currencies:
        if count[c] > 0:
            raw[c] /= count[c]

    # Normalizza ogni riga 0-100 (ranking intraday tra le 8 valute)
    row_min   = raw.min(axis=1)
    row_max   = raw.max(axis=1)
    row_range = (row_max - row_min).replace(0, 1.0)
    norm      = raw.subtract(row_min, axis=0).divide(row_range, axis=0).mul(100).round(2)

    # ── Storico completo (tutta la storia disponibile) ───────────────
    hist     = norm.dropna()
    dates    = [str(d.date()) for d in hist.index]
    strength_history = {c: [round(float(v), 2) for v in hist[c]] for c in currencies}

    # ── Serie segnale storica per ogni cross (riusata anche in cross_prices) ──
    def sig_fn(d):
        if pd.isna(d): return 'NEUTRAL'
        return 'SHORT' if d > threshold else ('LONG' if d < -threshold else 'NEUTRAL')

    sig_series_all = {
        name: (norm[base] - norm[quote]).apply(sig_fn)
        for name, (base, quote) in PAIRS.items()
    }

    # ── Segnali correnti + metadati ingresso ─────────────────────────
    last    = norm.iloc[-1]
    signals = {}
    for name, (base, quote) in PAIRS.items():
        b    = float(last[base])
        q    = float(last[quote])
        diff = round(b - q, 1)
        # Contrarian: base troppo forte → SHORT, base troppo debole → LONG
        signal = "SHORT" if diff > threshold else "LONG" if diff < -threshold else "NEUTRAL"

        ss = sig_series_all[name]

        # Ultimo ingresso (ultimo segnale non-NEUTRAL in assoluto)
        non_neutral = ss[ss != 'NEUTRAL']
        last_entry_date   = str(non_neutral.index[-1].date()) if len(non_neutral) > 0 else None
        last_entry_signal = str(non_neutral.iloc[-1])         if len(non_neutral) > 0 else 'NEUTRAL'

        # Da quando è attivo il segnale corrente (ultima transizione a questo segnale)
        transitions = ss[(ss == signal) & (ss.shift(1) != signal)]
        signal_since = str(transitions.index[-1].date()) if len(transitions) > 0 else None

        signals[name] = {
            "base": base, "quote": quote,
            "base_str": round(b, 1), "quote_str": round(q, 1),
            "diff": diff, "signal": signal,
            "last_entry_date":   last_entry_date,
            "last_entry_signal": last_entry_signal,
            "signal_since":      signal_since,
        }
    # Ordina per |diff| decrescente: i segnali più forti in cima
    signals = dict(sorted(signals.items(), key=lambda x: abs(x[1]["diff"]), reverse=True))

    # ── Backtest forza ────────────────────────────────────────────────
    bt_results = {}
    for name, (base, quote) in PAIRS.items():
        path = f"data/{name}_daily.csv"
        if not os.path.exists(path):
            continue
        prices  = pd.read_csv(path, index_col=0, parse_dates=True)["Close"]
        common  = prices.index.intersection(norm.index)
        prices  = prices[common].dropna()
        diff_s  = (norm.loc[common, base] - norm.loc[common, quote]).reindex(prices.index).dropna()
        common2 = prices.index.intersection(diff_s.index)
        prices  = prices[common2]
        diff_s  = diff_s[common2]

        # Contrarian: SHORT quando base è molto forte, LONG quando è molto debole
        pos       = diff_s.apply(lambda d: -1 if d > threshold else (1 if d < -threshold else 0))
        daily_ret = prices.pct_change()
        strat_ret = (pos.shift(1) * daily_ret).dropna()

        total_ret = float((1 + strat_ret).prod() - 1) * 100
        bh_ret    = float(prices.iloc[-1] / prices.iloc[0] - 1) * 100
        cum       = (1 + strat_ret).cumprod()
        max_dd    = float(((cum / cum.cummax()) - 1).min() * 100)
        sharpe    = float(strat_ret.mean() / strat_ret.std() * np.sqrt(252)) if strat_ret.std() > 0 else 0.0
        n_trades  = int((pos.diff() != 0).sum())
        in_pos    = pos.shift(1).reindex(strat_ret.index).fillna(0)
        active    = strat_ret[in_pos != 0]
        win_rate  = float((active > 0).mean() * 100) if len(active) > 0 else 0.0

        bt_results[name] = {
            "return_pct":   round(total_ret, 2),
            "buy_hold_pct": round(bh_ret, 2),
            "max_dd_pct":   round(max_dd, 2),
            "sharpe":       round(sharpe, 2),
            "n_trades":     n_trades,
            "win_rate":     round(win_rate, 1),
        }
        print(f"  ST {name}: {total_ret:+.2f}% | B&H {bh_ret:+.2f}% | Sharpe {sharpe:.2f}")

    # ── Prezzi storici per cross (file separati, caricati on-demand) ──
    os.makedirs("data/cross_prices", exist_ok=True)
    for name in CROSSES:
        path = f"data/{name}_daily.csv"
        if not os.path.exists(path):
            continue
        prices_all = pd.read_csv(path, index_col=0, parse_dates=True)["Close"]
        prices_aligned = prices_all.reindex(hist.index)
        d = 3 if name == "USDJPY" else 5

        # Serie segnali allineata a hist: 1=LONG, -1=SHORT, 0=NEUTRAL
        sig_map  = {'LONG': 1, 'SHORT': -1, 'NEUTRAL': 0}
        sig_hist = sig_series_all[name].reindex(hist.index).fillna('NEUTRAL')
        sig_int  = [sig_map[str(s)] for s in sig_hist]

        price_data = {
            "dates":   dates,
            "prices":  [round(float(v), d) if not pd.isna(v) else None for v in prices_aligned],
            "signals": sig_int,
        }
        with open(f"data/cross_prices/{name}.json", "w") as f:
            json.dump(price_data, f, separators=(",", ":"))
        print(f"  Saved data/cross_prices/{name}.json ({len(dates)} sessioni)")

    output = {
        "last_update":      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "window_days":      window,
        "threshold":        threshold,
        "dates":            dates,
        "strength_history": strength_history,
        "signals":          signals,
        "backtest":         bt_results,
    }
    with open("data/strength_data.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Forza valutaria salvata -> data/strength_data.json ({len(dates)} sessioni)")
    return output


if __name__ == "__main__":
    print("=== Forex Data Updater ===")
    download_all()
    print("\n=== Forza Valutaria & Segnali ===")
    build_strength_json()
