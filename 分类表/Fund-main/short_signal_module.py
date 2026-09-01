
import pandas as pd
import numpy as np
from math import floor

# =====================
# Indicators & helpers
# =====================

TAX = 5/1e5
COMMISSION = 1.1/1e5

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(n).mean() / down.rolling(n).mean()
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    tr = pd.concat([
        (df['high'] - df['low']),
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def intraday_vwap(df):
    # expects DateTimeIndex; groups by date
    day = df.index.date
    pv = (df['close']*df['volume']).groupby(day).cumsum()
    vv = df['volume'].groupby(day).cumsum().replace(0, np.nan)
    vwap = pv / vv
    return pd.Series(vwap.values, index=df.index, name="VWAP")

def rolling_zscore(s, win=60):
    m = s.rolling(win).mean()
    sd = s.rolling(win).std(ddof=0).replace(0, np.nan)
    return (s - m) / sd

# =====================
# Signal generation
# =====================

def generate_signals(df,
                     vol_multiplier=1.8,
                     donchian_n=15,
                     atr_n=14,
                     rsi_n=14,
                     z_win=60,
                     atr_low=0.003,  # 0.3%
                     atr_high=0.025  # 2.5%
                     ):
    '''
    df: DataFrame with columns ['open','high','low','close','volume'], DateTimeIndex at 5min freq
    Returns df with columns: EMA5, EMA20, RSI, ATR, VWAP, signal (1/0), etc.
    '''
    df = df.copy()
    df['EMA5'] = ema(df['close'], 5*48)  # 5 periods of 5min in a trading day
    df['EMA20'] = ema(df['close'], 20*48) # 20 periods of 5min in a trading day
    df['RSI'] = rsi(df['close'], rsi_n*48)
    df['ATR'] = atr(df, atr_n*48)
    df['VWAP'] = intraday_vwap(df)
    df['dist_vwap'] = (df['close'] - df['VWAP']).abs()
    df['z_dist'] = rolling_zscore(df['dist_vwap'], z_win)

    # Donchian 突破
    df['don_high'] = df['high'].rolling(donchian_n).max()

    # 量能放大
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_spike'] = df['volume'] > vol_multiplier * df['vol_ma']

    # "回踩 VWAP 后再次向上"
    above = df['close'] > df['VWAP']
    recross_up = (~above & above.shift(-1))
    recross_up = recross_up.shift(1).fillna(False)  # 对齐当前根

    # 波动过滤
    df['atr_pct'] = (df['ATR'] / df['close']).fillna(0)
    vol_filter = (df['atr_pct'] >= atr_low) & (df['atr_pct'] <= atr_high)

    # 方向与动能
    trend_ok = df['EMA5'] > df['EMA20']
    momentum_ok = (df['RSI'] >= 55) & (df['RSI'] <= 75)
    z_ok = df['z_dist'] < 2

    # 两种入场：回踩再上 + 突破
    breakout = df['close'] > df['don_high'].shift(1)
    entry_raw = (trend_ok & momentum_ok & z_ok & vol_filter &
                 df['vol_spike'] & (recross_up | breakout))

    df['signal'] = np.where(entry_raw, 1, 0).astype(int)
    return df

# =====================
# T+1 exits & backtest
# =====================

def apply_t1_exits(df, max_hold_days=2,
                   sl_atr=1.2, trail_init=2.0, trail_atr=1.5, dd_atr=2.5,
                   capital=100000.0, risk_fraction=0.005):
    '''
    Backtest with T+1 exits.
    - capital: starting capital for equity calculation
    - risk_fraction: fraction of capital risked per trade (e.g., 0.5%)
    Shares = floor( risk_capital / (ATR_entry * sl_atr) ), at least 1.
    '''
    df = df.copy()
    df['position'] = 0
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['shares'] = 0
    df['pnl'] = 0.0
    df['holding_days'] = 0
    df['equity'] = np.nan

    in_pos = False
    entry_idx = None
    highest = None
    shares = 0
    equity = capital

    trade_day = pd.Series(df.index.date, index=df.index)

    for i in range(len(df)):
        idx = df.index[i]
        if not in_pos:
            if df['signal'].iat[i] == 1:
                in_pos = True
                entry_idx = i
                entry_price = df['close'].iat[i]
                atr_entry = df['ATR'].iat[i]
                # 计算头寸
                risk_capital = capital * risk_fraction
                # 防止 ATR 为 0
                denom = max(atr_entry * sl_atr, 1e-9)
                shares = min(int(max(1, floor(risk_capital / denom))), int(floor(capital / (entry_price*(1+COMMISSION)))))
                df.at[idx, 'position'] = 1
                df.at[idx, 'entry_price'] = entry_price
                df.at[idx, 'shares'] = shares
                highest = df['high'].iat[i]
                df.at[idx, 'equity'] = equity
            else:
                df.at[idx, 'equity'] = equity
        else:
            # 持仓期间
            df.at[idx, 'position'] = 1
            entry_price = df['entry_price'].iloc[entry_idx]
            atr_val = df['ATR'].iat[i]
            highest = max(highest, df['high'].iat[i])
            df.at[idx, 'entry_price'] = entry_price
            df.at[idx, 'shares'] = shares

            # 次日及以后才允许退出
            can_exit = (trade_day.iat[i] != trade_day.iat[entry_idx])

            exit_now = False
            exit_price = df['close'].iat[i]

            if can_exit:
                # 硬性止损
                if df['low'].iat[i] <= entry_price - sl_atr * atr_val:
                    exit_now = True

                # 首次锁盈 & 追踪止盈
                if not exit_now and (df['high'].iat[i] >= entry_price + trail_init * atr_val):
                    protect = entry_price
                    if df['close'].iat[i] <= protect:
                        exit_now = True

                if not exit_now:
                    # 最高价回撤
                    if df['close'].iat[i] <= (highest - trail_atr * atr_val):
                        exit_now = True

                # 条件出场
                if not exit_now:
                    cond1 = df['close'].iat[i] < df['VWAP'].iat[i]
                    cond2 = df['EMA5'].iat[i] < df['EMA20'].iat[i]
                    cond3 = False
                    if i > 0 and not np.isnan(df['RSI'].iat[i-1]):
                        cond3 = (df['RSI'].iat[i-1] > 70) and (df['RSI'].iat[i] < 65)
                    intraday_dd = highest - df['low'].iat[i]
                    cond4 = intraday_dd >= dd_atr * atr_val
                    if cond1 or cond2 or cond3 or cond4:
                        exit_now = True

                # 时间出场：第 3 个交易日收盘卖出
                hold_days = (pd.Series(trade_day).iloc[entry_idx:i+1].nunique() - 1)
                df.at[idx, 'holding_days'] = hold_days
                if not exit_now and hold_days >= max_hold_days:
                    exit_now = True

            if exit_now:
                df.at[idx, 'exit_price'] = exit_price
                trade_pnl = ( exit_price*(1-TAX-COMMISSION) - entry_price*(1+COMMISSION) ) * shares
                df.at[idx, 'pnl'] = trade_pnl
                equity += trade_pnl
                df.at[idx, 'equity'] = equity
                in_pos = False
                entry_idx = None
                highest = None
                shares = 0
            else:
                df.at[idx, 'equity'] = equity

    # 完成后填充未赋值的 equity
    df['equity'] = df['equity'].ffill().fillna(capital)
    df['trade_pnl'] = df['pnl'].where(df['exit_price'].notna(), 0.0)
    return df

# =====================
# Metrics
# =====================

def _max_drawdown(series):
    '''Max drawdown on equity curve (series).'''
    roll_max = series.cummax()
    dd = series / roll_max - 1.0
    return dd.min(), dd

def evaluate_performance(df, capital=100000.0):
    '''
    Expects df after apply_t1_exits (with equity and trade_pnl).
    Returns summary dict and trades DataFrame.
    '''
    out = {}
    eq = df['equity'].dropna()
    if eq.empty:
        return {'message':'no equity data'}, pd.DataFrame()

    # Max DD
    mdd, dd_series = _max_drawdown(eq)
    out['max_drawdown'] = float(mdd)

    # Daily returns (group by date on equity last value)
    daily_eq = eq.groupby(eq.index.date).last()
    daily_ret = daily_eq.pct_change().dropna()
    if len(daily_ret) > 1 and daily_ret.std(ddof=0) > 0:
        sharpe = daily_ret.mean() / daily_ret.std(ddof=0) * np.sqrt(252)
    else:
        sharpe = np.nan
    out['sharpe'] = float(sharpe) if pd.notna(sharpe) else None

    # Total return
    total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1.0
    out['total_return'] = float(total_ret)

    # Trades
    exits = df['exit_price'].notna()
    trades = df.loc[exits, ['entry_price', 'exit_price', 'pnl', 'shares', 'holding_days']].copy()
    trades['win'] = trades['pnl'] > 0
    out['num_trades'] = int(len(trades))
    out['win_rate'] = float(trades['win'].mean()) if len(trades)>0 else None
    out['avg_win'] = float(trades.loc[trades['win'], 'pnl'].mean()) if trades['win'].any() else None
    out['avg_loss'] = float(trades.loc[~trades['win'], 'pnl'].mean()) if (~trades['win']).any() else None

    return out, trades

# =====================
# Simple grid search
# =====================

def grid_search(df,
                vol_m_list=(1.3,1.5,1.8),
                don_list=(15,20,30),
                trail_init_list=(1.5,2.0,2.5),
                trail_atr_list=(1.0,1.5,2.0),
                capital=100000.0,
                risk_fraction=0.005):
    rows = []
    for vm in vol_m_list:
        for dn in don_list:
            for ti in trail_init_list:
                for ta in trail_atr_list:
                    gdf = generate_signals(df, vol_multiplier=vm, donchian_n=dn)
                    bdf = apply_t1_exits(gdf, trail_init=ti, trail_atr=ta,
                                         capital=capital, risk_fraction=risk_fraction)
                    perf, _ = evaluate_performance(bdf, capital=capital)
                    row = {
                        'vol_multiplier': vm,
                        'donchian_n': dn,
                        'trail_init': ti,
                        'trail_atr': ta,
                        **perf
                    }
                    rows.append(row)
    res = pd.DataFrame(rows).sort_values(['total_return','sharpe'], ascending=[False, False])
    return res
