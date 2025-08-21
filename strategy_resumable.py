import backtrader as bt
import pandas as pd
import json
import os
from emailer import send_email

print(">>> NEW STRATEGY VERSION LOADED")

# ✅ Centralized state path (avoids scattered JSONs, no circular import)
def state_path(pair: str) -> str:
    state_dir = os.path.join(os.path.dirname(__file__), "state")
    os.makedirs(state_dir, exist_ok=True)
    return os.path.join(state_dir, f"{pair}_state.json")


class MasterCHOCHStrategy(bt.Strategy):
    params = dict(
        swing_lookback=10,
        risk_reward=1.0,
        use_entry_shift=True,
        entry_shift_ratio=0,
        log_filename='trade_log.csv',
        live_mode=True
    )

    def __init__(self):
        print(">>> NEW STRATEGY VERSION LOADED")
        self.show_queued_alerts = True
        self.data_ctx = {}

        for d in self.datas:
            name = d._name
            state_file = state_path(name)   # ✅ use new helper
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
                    'total_wins': 0, 'total_losses': 0, 'trade_log': []
                }

            self.data_ctx[name] = state

        # ✅ Use GitHub Secrets via environment variables
        self.sender_email = os.getenv("EMAIL_USER")
        self.app_password = os.getenv("EMAIL_PASS")
        self.recipient_email = os.getenv("EMAIL_TO")

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt}, {txt}')

    def detect_swings_buffered(self, data, ctx):
        lb = self.p.swing_lookback
        if len(data) < (2 * lb + 1): return

        highs = list(data.high.get(size=2 * lb + 1))
        lows = list(data.low.get(size=2 * lb + 1))
        if len(highs) != 2 * lb + 1: return

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
            if swing['type'] == 'high' and curr > swing['center_price']: continue
            if swing['type'] == 'low' and curr < swing['center_price']: continue

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
                    if self.p.live_mode:
                        self.log(f'TRIGGERED TRADE ALERT:\n{msg}')
                        send_email(f"TRADE ALERT: {direction} {pair}", msg,
                                   self.sender_email, self.app_password, self.recipient_email)
                    else:
                        order = self.buy(data=data, price=q['entry_price']) if q['is_long'] else self.sell(data=data, price=q['entry_price'])
                        ctx['trade_log'].append([str(data.datetime.datetime(0)), direction, q['entry_price'], q['sl'], q['tp'], '', 0, 0, q['rr']])
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

                # Convert to R units
                mae_R = ctx['max_adverse'] / ctx['risk_R'] if ctx['risk_R'] else 0
                mfe_R = ctx['max_favorable'] / ctx['risk_R'] if ctx['risk_R'] else 0

                if (ctx['is_long'] and price >= ctx['tp']) or (not ctx['is_long'] and price <= ctx['tp']):
                    ctx['total_wins'] += 1
                    if not self.p.live_mode:
                        ctx['trade_log'][-1][5:8] = ['WIN', mae_R, mfe_R]
                        self.close(data=data)
                    ctx['trade_active'] = False
                    self.log(f"{pair} TP HIT")

                elif (ctx['is_long'] and price <= ctx['sl']) or (not ctx['is_long'] and price >= ctx['sl']):
                    ctx['total_losses'] += 1
                    if not self.p.live_mode:
                        ctx['trade_log'][-1][5:8] = ['LOSS', mae_R, mfe_R]
                        self.close(data=data)
                    ctx['trade_active'] = False
                    self.log(f"{pair} SL HIT")
                continue

            # Swing logic (unchanged)
            if len(ctx['highs']) >= 2 and len(ctx['lows']) >= 2:
                h1_time, h1 = ctx['highs'][-1]
                h0_time, h0 = ctx['highs'][-2]
                l1_time, l1 = ctx['lows'][-1]
                l0_time, l0 = ctx['lows'][-2]

                if h1 > h0 and price > h1:
                    raw_sl, raw_risk = l1, price - l1
                    if raw_risk < 1e-6: return
                    entry_shift = raw_risk * self.p.entry_shift_ratio if self.p.use_entry_shift else 0
                    entry, sl, tp = price - entry_shift, price + raw_risk, l1
                    rr = abs(tp - entry) / abs(entry - sl)
                    ctx['queued_trade'] = dict(entry_price=entry, sl=sl, tp=tp, is_long=False, rr=rr)
                    self.log(f"{pair} QUEUED INVERTED SELL")

                elif l1 < l0 and price < l1:
                    raw_sl, raw_risk = h1, h1 - price
                    if raw_risk < 1e-6: return
                    entry_shift = raw_risk * self.p.entry_shift_ratio if self.p.use_entry_shift else 0
                    entry, sl, tp = price + entry_shift, price - raw_risk, h1
                    rr = abs(tp - entry) / abs(entry - sl)
                    ctx['queued_trade'] = dict(entry_price=entry, sl=sl, tp=tp, is_long=True, rr=rr)
                    self.log(f"{pair} QUEUED INVERTED BUY")

    def stop(self):
        for pair, ctx in self.data_ctx.items():
            # Save state for warm resuming
            with open(state_path(pair), 'w') as f:   # ✅ save in state dir
                json.dump(ctx, f, indent=2)
            print(f"Saved state for {pair} to {state_path(pair)}")

            if not self.p.live_mode:
                df = pd.DataFrame(ctx['trade_log'], columns=[
                    'Datetime', 'Action', 'Entry', 'SL', 'TP', 'Result',
                    'Max Drawdown (R)', 'Max Profit (R)', 'RR'])
                df.to_csv(f"{pair}_log.csv", index=False)

                total = ctx['total_wins'] + ctx['total_losses']
                win_rate = (ctx['total_wins'] / total * 100) if total else 0

                # Separate averages for winners and losers
                avg_mae_winners = df.loc[df['Result'] == 'WIN', 'Max Drawdown (R)'].mean() if not df.empty else 0
                avg_mfe_losers = df.loc[df['Result'] == 'LOSS', 'Max Profit (R)'].mean() if not df.empty else 0

                print(f'\n== {pair} BACKTEST SUMMARY ==')
                print(f'Trades: {total}, Wins: {ctx["total_wins"]}, Losses: {ctx["total_losses"]}, Win Rate: {win_rate:.2f}%')
                print(f'Average Max Drawdown Before Win (R): {avg_mae_winners:.2f}')
                print(f'Average Max Profit Before Loss (R): {avg_mfe_losers:.2f}')
            else:
                print("===== LIVE MODE COMPLETE =====")
