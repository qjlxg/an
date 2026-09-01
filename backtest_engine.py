import os
import pandas as pd
import glob
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 实战优化配置 ---
DATA_DIR = 'fund_data'
HOLD_DAYS = [10, 20, 40, 60]  # 增加到60天（一个季度）观察长效
MIN_SCORE_THRESHOLD = 75      
STOP_LOSS = -0.07             # 中线放宽止损，防止在底部被洗出
# 量比逻辑：改为“不放量杀跌”
VOL_LIMIT_UPPER = 1.1         # 不超过均量的1.1倍
VOL_LIMIT_LOWER = 0.4         # 不低于0.4倍，防止僵尸股

def calculate_tech(df):
    df = df.sort_values('日期').copy()
    # RSI & KDJ
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    low_9 = df['收盘'].rolling(9).min()
    high_9 = df['收盘'].rolling(9).max()
    rsv = (df['收盘'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    # 乖离与量比
    df['MA20'] = df['收盘'].rolling(20).mean()
    df['BIAS_20'] = (df['收盘'] - df['MA20']) / df['MA20'] * 100
    df['V_MA5'] = df['成交量'].shift(1).rolling(5).mean()
    df['VOL_RATIO'] = df['成交量'] / df['V_MA5']
    return df

def run_single_backtest(file_path):
    trades = []
    try:
        code = re.search(r'(\d{6})', os.path.basename(file_path)).group(1)
        df = pd.read_csv(file_path)
        if len(df) < 300: return []
        df = calculate_tech(df)
        
        for i in range(20, len(df) - max(HOLD_DAYS)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            score = 0
            if row['RSI'] < 35: score += 30
            if row['J'] < 5: score += 30
            if row['BIAS_20'] < -4: score += 40
            
            # 核心改进：取消极致缩量，改为“不放量且不极端缩量”
            is_vol_safe = VOL_LIMIT_LOWER < row['VOL_RATIO'] < VOL_LIMIT_UPPER
            
            # 右侧确认信号
            if score >= MIN_SCORE_THRESHOLD and is_vol_safe and row['J'] > prev['J']:
                buy_price = row['收盘']
                res = {'代码': code, '日期': row['日期']}
                for d in HOLD_DAYS:
                    period_df = df.iloc[i+1 : i+d+1]
                    # 检查是否触发止损
                    if (period_df['最低'].min() - buy_price) / buy_price <= STOP_LOSS:
                        res[f'{d}日收益%'] = STOP_LOSS * 100
                    else:
                        res[f'{d}日收益%'] = round((df.iloc[i+d]['收盘'] - buy_price) / buy_price * 100, 2)
                trades.append(res)
    except: pass
    return trades

def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    all_trades = []
    with ProcessPoolExecutor() as executor:
        for result in executor.map(run_single_backtest, files):
            all_trades.extend(result)
            
    if all_trades:
        res_df = pd.DataFrame(all_trades).sort_values('日期')
        summary = []
        for d in HOLD_DAYS:
            col = f'{d}日收益%'
            win_rate = (res_df[col] > 0).mean() * 100
            avg_ret = res_df[col].mean()
            # 统计时间跨度
            total_days = (pd.to_datetime(res_df['日期'].max()) - pd.to_datetime(res_df['日期'].min())).days
            monthly_signals = len(res_df) / (total_days / 30) if total_days > 0 else 0
            
            summary.append({
                '周期': f'{d}天', 
                '月均信号': round(monthly_signals, 1),
                '胜率%': round(win_rate, 2), 
                '平均收益%': round(avg_ret, 2)
            })
        print(pd.DataFrame(summary).to_string(index=False))

if __name__ == "__main__":
    main()
