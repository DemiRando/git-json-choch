import backtrader as bt
import pandas as pd
import os
import json
from datetime import datetime, timedelta, UTC
from strategy_resumable import MasterCHOCHStrategy
import dukascopy_python as dkc
from dukascopy_python import INTERVAL_MIN_1
from dukascopy_python.instruments import (
    INSTRUMENT_FX_MAJORS_EUR_USD,
    INSTRUMENT_FX_CROSSES_AUD_NZD,
    INSTRUMENT_FX_CROSSES_GBP_JPY,
    INSTRUMENT_VCCY_LTC_USD
)

# === CONFIGURATION ===
SYMBOLS = {
    'EUR_USD': INSTRUMENT_FX_MAJORS_EUR_USD,
    'AUD_NZD': INSTRUMENT_FX_CROSSES_AUD_NZD,
    'GBP_JPY': INSTRUMENT_FX_CROSSES_GBP_JPY,
    'LTC_USD': INSTRUMENT_VCCY_LTC_USD
}
TIMEFRAME_MINUTES = 1
DATE_FORMAT = '%d.%m.%Y %H:%M:%S.%f'
COLUMN_DATETIME = 'Gmt time'

BOOTSTRAP_START = datetime(2023, 6, 7, tzinfo=UTC)
BOOTSTRAP_END = datetime(2025, 7, 7, tzinfo=UTC)

STATE_DIR = "state"
os.makedirs(STATE_DIR, exist_ok=True)

# === CLEAN OLD TEMP FILES BEFORE START ===
for f in os.listdir('.'):
    if f.endswith('_temp.csv'):
        try:
            os.remove(f)
            print(f"Deleted old temp file: {f}")
        except Exception as e:
            print(f"Failed to delete {f}: {e}")

# === HELPERS ===
def state_path(symbol):
    """Return full path for state file."""
    return os.path.join(STATE_DIR, f"{symbol}_state.json")

def load_state_last_dt(symbol):
    """Load last trade datetime from state JSON if available."""
    state_file = state_path(symbol)
    if not os.path.exists(state_file):
        return None
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        log = state.get('trade_log', [])
        if not log:
            return None
        # Force tz-aware datetime in UTC
        return datetime.strptime(log[-1][0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except Exception as e:
        print(f"{symbol}: Error loading state: {e}")
        return None

def fetch_dukas_to_temp(symbol):
    """Fetch candles into a temporary CSV for either bootstrap or resume mode."""
    state_dt = load_state_last_dt(symbol)

    if not state_dt:
        # No JSON state — bootstrap mode
        start_dt = BOOTSTRAP_START
        end_dt = BOOTSTRAP_END
        print(f"{symbol}: No state found, fetching FULL range {start_dt} → {end_dt}")
    else:
        # Resume mode — fetch from last trade datetime
        start_dt = state_dt
        end_dt = datetime.now(UTC)
        print(f"{symbol}: Resuming from {start_dt} → {end_dt}")

    try:
        df = dkc.fetch(
            instrument=SYMBOLS[symbol],
            interval=INTERVAL_MIN_1,
            offer_side=dkc.OFFER_SIDE_BID,
            start=start_dt,
            end=end_dt
        )
    except Exception as e:
        print(f"{symbol}: Dukascopy fetch failed: {e}")
        return None

    if df.empty:
        print(f"{symbol}: No candles fetched.")
        return None

    # Ensure timestamp column exists
    if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df['timestamp'] = df.index

    # Normalize to UTC tz-aware pandas.Timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Keep Backtrader-friendly structure
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df[COLUMN_DATETIME] = df['timestamp'].dt.strftime(DATE_FORMAT)

    # If resuming, filter only newer data
    if state_dt:
        state_ts = pd.Timestamp(state_dt).tz_convert("UTC")
        df = df[df['timestamp'] > state_ts]

    if df.empty:
        print(f"{symbol}: No new candles after filtering.")
        return None

    # Save as temp CSV
    temp_file = f"{symbol}_temp.csv"
    df[[COLUMN_DATETIME, 'open', 'high', 'low', 'close', 'volume']].to_csv(temp_file, index=False)
    print(f"{symbol}: Temp CSV created with {len(df)} rows.")
    return temp_file

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print(">>> STRATEGY VERSION WITH CSV WARMUP & FALLBACK LOADED")
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MasterCHOCHStrategy, live_mode=False)

    temp_files = []

    for symbol in SYMBOLS:
        temp_file = fetch_dukas_to_temp(symbol)
        if not temp_file:
            continue

        temp_files.append(temp_file)

        data = bt.feeds.GenericCSVData(
            dataname=temp_file,
            dtformat=DATE_FORMAT,
            timeframe=bt.TimeFrame.Minutes,
            compression=TIMEFRAME_MINUTES,
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            openinterest=-1,
            headers=True
        )
        data._name = symbol
        cerebro.adddata(data)

    if temp_files:
        print("Running RESUMABLE BACKTEST on:", [os.path.splitext(f)[0].replace("_temp", "") for f in temp_files])
        cerebro.run()

    # === CLEANUP AFTER RUN ===
    for tf in temp_files:
        try:
            os.remove(tf)
            print(f"Deleted temp file {tf}")
        except Exception as e:
            print(f"Failed to delete {tf}: {e}")
