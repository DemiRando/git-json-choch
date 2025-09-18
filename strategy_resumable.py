import backtrader as bt
import pandas as pd
import json
import os
from emailer import send_email

print(">>> NEW STRATEGY VERSION LOADED")

# Centralized state path (keeps Github workflow & cache intact)
def state_path(pair: str) -> str:
    state_dir = os.path.join(os.path.dirname(__file__), "state")
    os.makedirs(state_dir, exist_ok=True)
    return os.path.join(state_dir, f"{pair}_state.json")


class MasterCHOCHStrategy(bt.Strategy):
    params = dict(
        # swing / zone logic
        swing_lookback=10,

        # risk settings
        risk_reward=0.5,          # from local: honoured in TP calculation
        use_entry_shift=True,
        entry_shift_ratio=0,

        # logging / mode
        log_filename='trade_log.csv',
        live_mode=True,          # keep local default; GitHub workflow env will control runtime

        # === General entry filters (from local analysis) ===
        rsi_min=40.0,             # skip if RSI < 40 (any side)
        atr_norm_max=0.025,       # skip if ATR_norm > 0.025 (any side)
        macd_norm_min=0.0,        # skip if MACD_norm < 0 (any side)

        # === Side-specific refinements ===
        rsi_min_buy=55.0,         # buys require RSI >= 55
        rsi_min_sell=45.0,        # sells require RSI >= 45
    )

    def __init__(self):
        print(">>> NEW STRATEGY VERSION LOADED")
        self.show_queued_alerts = True
        self.data_ctx = {}

        # --- indicators per data feed (keeps richer entry-filter logic) ---
        self.inds = {}
        for d in self.datas:
            name = d._name

            # restore or create state using centralized state path
            state_file = state_path(name)
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                print(f">>> Restored state for {name} from {state_file}")
            else:
                state = {
                    'highs': [], 'lows': [], 'pending_swings': [],
                    'trade_active': False, 'queued_trade': None,
                    'entry_price': None, 'sl': None, 'tp': None, 'is_long': None,
                    'max_favorable': 0.0, 'max_adverse': 0.0,
                    'total_wins': 0, 'total_losses': 0, 'trade_log': [],
                    'last_alerted_event': None,
                    'latest_event': None,
                    'email_status': None,
                    'email_error': None,
                    'last_triggered_trade': None
                }

            # attach indicators for the feed (mirrors local logic)
            self.inds[d] = dict(
                sma50=bt.indicators.SMA(d.close, period=50),
                sma200=bt.indicators.SMA(d.close, period=200),
                ema20=bt.indicators.EMA(d.close, period=20),
                rsi=bt.indicators.RSI(d.close, period=14),
                macd=bt.indicators.MACD(d.close),
                atr=bt.indicators.ATR(d, period=14),
                bb=bt.indicators.BollingerBands(d.close, period=20, devfactor=2),
                adx=bt.indicators.ADX(d, period=14),
            )

            self.data_ctx[name] = state

        # Keep using environment variables for secrets (do NOT hardcode)
        self.sender_email = os.getenv("EMAIL_USER")
        self.app_password = os.getenv("EMAIL_PASS")
        self.recipient_email = os.getenv("EMAIL_TO")

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt}, {txt}')

    def detect_swings_buffered(self, data, ctx):
        lb = self.p.swing_lookback
        if len(data) < (2 * lb + 1):
            return

        highs = list(data.high.get(size=2 * lb + 1))
        lows = list(data.low.get(size=2 * lb + 1))
        if len(highs) != 2 * lb + 1:
            return

        center_high, center_low = highs[lb], lows[lb]
        is_high = all(center_high > h for i, h in enumerate(highs) if i != lb)
        is_low = all(center_low < l for i, l in enumerate(lows) if i != lb)

        time = data.datetime.datetime(-lb)

        if is_high:
            ctx['pending_swings'].append({'type': 'high', 'center_price': center_high, 'center_time': str(time), 'future_bars': []})
        if is_low:
            ctx['pending_swings'].append({'type': 'low', 'center_price': center_low, 'center_time': str(time), 'future_bars': []})

        updated = []
        for swing in ctx['pending_swings']:
            curr = data.high[0] if swing['type'] == 'high' else data.low[0]
            if swing['type'] == 'high' and curr > swing['center_price']:
                continue
            if swing['type'] == 'low' and curr < swing['center_price']:
                continue

            swing['future_bars'].append(curr)
            if len(swing['future_bars']) >= lb:
                label = f"Swing {'High' if swing['type'] == 'high' else 'Low'} Detected at {swing['center_price']}"
                self.log(label, swing['center_time'])
                key = 'highs' if swing['type'] == 'high' else 'lows'
                if not ctx[key] or swing['center_price'] != ctx[key][-1][1]:
                    ctx[key].append((swing['center_time'], swing['center_price']))
            else:
                updated.append(swing)

        ctx['pending_swings'] = updated

    def capture_indicators(self, data):
        """Return normalized indicator snapshot for a given data feed.

        Normalizations:
          - ATR normalized by close price (atr / price)
          - MACD and MACD signal normalized by close price
          - BB %B (between 0 and 1): (close - lower) / (upper - lower)
          - RSI left as-is (0-100)
        """
        i = self.inds[data]
        close = float(data.close[0]) if data.close[0] is not None else 0.0

        # raw pulls (some indicators may be NaN at startup)
        rsi_v = float(i['rsi'][0]) if i['rsi'][0] is not None else float('nan')
        atr_v = float(i['atr'][0]) if i['atr'][0] is not None else float('nan')
        macd_v = float(i['macd'].macd[0]) if i['macd'].macd[0] is not None else float('nan')
        macdsig_v = float(i['macd'].signal[0]) if i['macd'].signal[0] is not None else float('nan')

        # bollinger bands
        try:
            bb_top = float(i['bb'].top[0])
            bb_bot = float(i['bb'].bot[0])
        except Exception:
            bb_top, bb_bot = float('nan'), float('nan')

        # normalize safely (avoid division by zero)
        atr_norm = (atr_v / close) if close and pd.notna(atr_v) else float('nan')
        macd_norm = (macd_v / close) if close and pd.notna(macd_v) else float('nan')
        macdsig_norm = (macdsig_v / close) if close and pd.notna(macdsig_v) else float('nan')

        bbp = float('nan')
        if pd.notna(bb_top) and pd.notna(bb_bot) and (bb_top - bb_bot) != 0:
            bbp = (close - bb_bot) / (bb_top - bb_bot)

        return dict(
            rsi=rsi_v,
            atr_norm=atr_norm,
            macd_norm=macd_norm,
            macdsig_norm=macdsig_norm,
            bbp=bbp
        )

    def _passes_entry_filters(self, ind_vals, is_long):
        # Any NaN critical indicator -> reject (conservative)
        if not pd.notna(ind_vals['rsi']) or not pd.notna(ind_vals['atr_norm']) or not pd.notna(ind_vals['macd_norm']):
            return False, f"INDICATOR_NAN rsi={ind_vals['rsi']}, atr_norm={ind_vals['atr_norm']}, macd_norm={ind_vals['macd_norm']}"

        # General filters
        if ind_vals['rsi'] < self.p.rsi_min:
            return False, f"RSI<{self.p.rsi_min} (rsi={ind_vals['rsi']:.2f})"
        if ind_vals['atr_norm'] > self.p.atr_norm_max:
            return False, f"ATR_norm>{self.p.atr_norm_max} (atr_norm={ind_vals['atr_norm']:.5f})"
        if ind_vals['macd_norm'] < self.p.macd_norm_min:
            return False, f"MACD_norm<{self.p.macd_norm_min} (macd_norm={ind_vals['macd_norm']:.8f})"

        # Side-specific refinements
        if is_long:
            if ind_vals['rsi'] < self.p.rsi_min_buy:
                return False, f"BUY: RSI<{self.p.rsi_min_buy} (rsi={ind_vals['rsi']:.2f})"
        else:
            if ind_vals['rsi'] < self.p.rsi_min_sell:
                return False, f"SELL: RSI<{self.p.rsi_min_sell} (rsi={ind_vals['rsi']:.2f})"

        return True, "OK"

    def next(self):
        for data in self.datas:
            pair = data._name
            ctx = self.data_ctx[pair]
            price = data.close[0]

            self.detect_swings_buffered(data, ctx)

            if not ctx['trade_active'] and ctx['queued_trade']:
                q = ctx['queued_trade']
                valid = (q['is_long'] and price > q['entry_price'] and price > q['sl']) or \
                        (not q['is_long'] and price < q['entry_price'] and price < q['sl'])

                direction = "BUY" if q['is_long'] else "SELL"
                msg = f"""PAIR: {pair}
ACTION: {"TRIGGERED" if valid else "QUEUED"} {direction}
Entry: {q['entry_price']}
SL: {q['sl']}
TP: {q['tp']}"""

                if valid:
                    ctx.update({
                        'entry_price': q['entry_price'], 'sl': q['sl'], 'tp': q['tp'], 'is_long': q['is_long'],
                        'max_favorable': 0.0, 'max_adverse': 0.0, 'trade_active': True
                    })
                    ctx['risk_R'] = abs(q['entry_price'] - q['sl'])  # store risk unit

                    # snapshot normalized indicators at entry
                    ind_vals = self.capture_indicators(data)

                    if self.p.live_mode:
                        self.log(f'TRIGGERED TRADE ALERT:\n{msg}')
                        send_email(f"TRADE ALERT: {direction} {pair}", msg,
                                   self.sender_email, self.app_password, self.recipient_email)
                    else:
                        order = self.buy(data=data, price=q['entry_price']) if q['is_long'] else self.sell(data=data, price=q['entry_price'])
                        # append single-row log with entry indicators and placeholders for exit
                        row = [
                            str(data.datetime.datetime(0)),  # time_entry
                            '',                                # time_exit (fill at exit)
                            direction,                         # Action
                            q['entry_price'], q['sl'], q['tp'], q['rr'],  # entry, sl, tp, rr
                            '',                                # Result (WIN/LOSS) to be set at exit
                            0, 0,                              # Max Drawdown (R), Max Profit (R)
                            '',                                # pnl (optional)
                            ind_vals['rsi'], ind_vals['atr_norm'], ind_vals['macd_norm'], ind_vals['macdsig_norm'], ind_vals['bbp'],
                            '', '', '', '', ''                 # placeholders for exit indicators (indices 16-20)
                        ]
                        ctx['trade_log'].append(row)
                        self.log(f"TRADE EXECUTED: {direction} {pair} at {q['entry_price']}")
                else:
                    if self.show_queued_alerts and self.p.live_mode:
                        self.log(f'QUEUED TRADE ALERT:\n{msg}')
                        send_email(f"SETUP QUEUED: {direction} {pair}", msg,
                                   self.sender_email, self.app_password, self.recipient_email)

                ctx['queued_trade'] = None

            if ctx['trade_active']:
                move = price - ctx['entry_price'] if ctx['is_long'] else ctx['entry_price'] - price
                dd = ctx['entry_price'] - price if ctx['is_long'] else price - ctx['entry_price']
                ctx['max_favorable'] = max(ctx['max_favorable'], move)
                ctx['max_adverse'] = max(ctx['max_adverse'], dd)

                mae_R = ctx['max_adverse'] / ctx['risk_R'] if ctx['risk_R'] else 0
                mfe_R = ctx['max_favorable'] / ctx['risk_R'] if ctx['risk_R'] else 0

                # TP hit
                if (ctx['is_long'] and price >= ctx['tp']) or (not ctx['is_long'] and price <= ctx['tp']):
                    ctx['total_wins'] += 1
                    if not self.p.live_mode:
                        ind_vals = self.capture_indicators(data)  # snapshot indicators at exit
                        # update last trade row with exit time, result and MAE/MFE and exit indicators
                        if ctx['trade_log']:
                            ctx['trade_log'][-1][1] = str(data.datetime.datetime(0))            # time_exit
                            ctx['trade_log'][-1][7:10] = ['WIN', mae_R, mfe_R]                # Result, MAE, MFE
                            ctx['trade_log'][-1][16:21] = [                                   # exit indicators
                                ind_vals['rsi'], ind_vals['atr_norm'], ind_vals['macd_norm'],
                                ind_vals['macdsig_norm'], ind_vals['bbp']
                            ]
                        self.close(data=data)
                    ctx['trade_active'] = False
                    self.log(f"{pair} TP HIT")

                # SL hit
                elif (ctx['is_long'] and price <= ctx['sl']) or (not ctx['is_long'] and price >= ctx['sl']):
                    ctx['total_losses'] += 1
                    if not self.p.live_mode:
                        ind_vals = self.capture_indicators(data)  # snapshot indicators at exit
                        if ctx['trade_log']:
                            ctx['trade_log'][-1][1] = str(data.datetime.datetime(0))            # time_exit
                            ctx['trade_log'][-1][7:10] = ['LOSS', mae_R, mfe_R]               # Result, MAE, MFE
                            ctx['trade_log'][-1][16:21] = [                                   # exit indicators
                                ind_vals['rsi'], ind_vals['atr_norm'], ind_vals['macd_norm'],
                                ind_vals['macdsig_norm'], ind_vals['bbp']
                            ]
                        self.close(data=data)
                    ctx['trade_active'] = False
                    self.log(f"{pair} SL HIT")
                continue

            # --- Swing-based trade queuing (preserves local risk_reward behavior and filters) ---
            if len(ctx['highs']) >= 2 and len(ctx['lows']) >= 2:
                h1_time, h1 = ctx['highs'][-1]
                h0_time, h0 = ctx['highs'][-2]
                l1_time, l1 = ctx['lows'][-1]
                l0_time, l0 = ctx['lows'][-2]

                # evaluate filters BEFORE queuing trade (side-aware)
                ind_vals_now = self.capture_indicators(data)

                if h1 > h0 and price > h1:
                    # Inverted SELL (use side-specific filters)
                    passes, reason = self._passes_entry_filters(ind_vals_now, is_long=False)
                    if not passes:
                        self.log(f"{pair} SKIP SELL (filters): {reason}")
                    else:
                        raw_sl, raw_risk = l1, price - l1
                        if raw_risk < 1e-6: 
                            return
                        entry_shift = raw_risk * self.p.entry_shift_ratio if self.p.use_entry_shift else 0
                        entry = price - entry_shift
                        sl = price + raw_risk
                        risk_dist = abs(entry - sl)
                        tp = entry - (risk_dist * self.p.risk_reward)
                        rr = abs(tp - entry) / risk_dist if risk_dist > 0 else 0.0
                        ctx['queued_trade'] = dict(entry_price=entry, sl=sl, tp=tp, is_long=False, rr=rr)
                        self.log(f"{pair} QUEUED INVERTED SELL -> entry={entry}, sl={sl}, tp={tp}, rr={rr:.3f}")

                elif l1 < l0 and price < l1:
                    # Inverted BUY
                    passes, reason = self._passes_entry_filters(ind_vals_now, is_long=True)
                    if not passes:
                        self.log(f"{pair} SKIP BUY (filters): {reason}")
                    else:
                        raw_sl, raw_risk = h1, h1 - price
                        if raw_risk < 1e-6: 
                            return
                        entry_shift = raw_risk * self.p.entry_shift_ratio if self.p.use_entry_shift else 0
                        entry = price + entry_shift
                        sl = price - raw_risk
                        risk_dist = abs(entry - sl)
                        tp = entry + (risk_dist * self.p.risk_reward)
                        rr = abs(tp - entry) / risk_dist if risk_dist > 0 else 0.0
                        ctx['queued_trade'] = dict(entry_price=entry, sl=sl, tp=tp, is_long=True, rr=rr)
                        self.log(f"{pair} QUEUED INVERTED BUY  -> entry={entry}, sl={sl}, tp={tp}, rr={rr:.3f}")

    def stop(self):
        for pair, ctx in self.data_ctx.items():
            with open(state_path(pair), 'w') as f:
                json.dump(ctx, f, indent=2)
            print(f"Saved state for {pair} to {state_path(pair)}")

            # email logic kept: only send on latest_event difference
            email_sent = False
            email_error = None

            if self.p.live_mode and ctx.get("latest_event"):
                if ctx["latest_event"] != ctx.get("last_alerted_event"):
                    try:
                        send_email(
                            f"TRADE ALERT {pair}",
                            ctx["latest_event"],
                            self.sender_email,
                            self.app_password,
                            self.recipient_email
                        )
                        email_sent = True
                        ctx["last_alerted_event"] = ctx["latest_event"]
                        print(f"✅ Email sent: {ctx['latest_event']}")
                    except Exception as e:
                        email_error = str(e)
                        print(f"❌ Email failed for {pair}: {e}")

            ctx['email_status'] = "Sent" if email_sent else "Failed or Skipped"
            ctx['email_error'] = email_error

            if not self.p.live_mode:
                # write full CSV with richer header (backwards-compatible with Github format)
                df = pd.DataFrame(ctx['trade_log'], columns=[
                    'time_entry', 'time_exit', 'Action', 'Entry', 'SL', 'TP', 'RR',
                    'Result', 'Max Drawdown (R)', 'Max Profit (R)', 'pnl',
                    'RSI_entry', 'ATR_entry_norm', 'MACD_entry_norm', 'MACDsig_entry_norm', 'BBP_entry',
                    'RSI_exit', 'ATR_exit_norm', 'MACD_exit_norm', 'MACDsig_exit_norm', 'BBP_exit'
                ])
                df.to_csv(f"{pair}_log.csv", index=False)

                total = ctx['total_wins'] + ctx['total_losses']
                win_rate = (ctx['total_wins'] / total * 100) if total else 0
                avg_mae_winners = df.loc[df['Result'] == 'WIN', 'Max Drawdown (R)'].mean() if not df.empty else 0
                avg_mfe_losers = df.loc[df['Result'] == 'LOSS', 'Max Profit (R)'].mean() if not df.empty else 0

                print(f'\n== {pair} BACKTEST SUMMARY ==')
                if ctx['last_triggered_trade']:
                    t = ctx['last_triggered_trade']
                    print(f"Status: {t.get('status', 'N/A')}")
                    print(f"Direction: {t.get('direction', 'N/A')}")
                    print(f"Entry: {t.get('entry')} | SL: {t.get('sl')} | TP: {t.get('tp')}")
                    print(f"Date: {t.get('date', '(not recorded)')}")
                else:
                    print("No triggered trades in this run.")
                print(f"Trades: {total}, Wins: {ctx['total_wins']}, Losses: {ctx['total_losses']}, Win Rate: {win_rate:.2f}%")
                print(f"Average Max Drawdown Before Win (R): {avg_mae_winners:.2f}")
                print(f"Average Max Profit Before Loss (R): {avg_mfe_losers:.2f}")
                print(f"Email Status: {ctx['email_status']}")
                if ctx["email_error"]:
                    print(f"Email Error: {ctx['email_error']}")

            else:
                print("===== LIVE MODE COMPLETE =====")
                print(f'{pair} Email Status: {ctx["email_status"]}')
                if ctx["email_error"]:
                    print(f'{pair} Email Error: {ctx["email_error"]}')
