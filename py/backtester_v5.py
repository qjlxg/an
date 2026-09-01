import pandas as pd
import glob
import os
import numpy as np
import logging
import math
import pytz
from datetime import datetime

# --- 配置参数 (模拟 V4.4 策略设定) ---
FUND_DATA_DIR = 'fund_data'
BACKTEST_START_DATE = '2020-01-01'  # 回测起始日期
BACKTEST_END_DATE = '2024-12-31'    # 回测结束日期
INITIAL_CAPITAL = 100000.0          # 初始总资金 (包含基础仓位和预备金)
BUY_AMOUNT_PER_TRADE = 10000.0      # 每次买入金额 (模拟网格补仓金额)
REPORT_FILE_NAME = 'fund_backtest_v5_report.md' # V5.0 报告名称

# --- V4.4 策略核心纪律配置 ---
RSI_BUY_THRESHOLD = 30.0    # RSI(6) <= 30 时买入 (质量过滤)
GRID_STEP_PERCENT = 0.04    # 价格相对平均成本下跌 4% 时触发信号过滤 (Level 1 触发)
TREND_RATIO_MIN = 0.95      # MA50/MA250 必须大于等于 0.95 (风控过滤)
STOP_LOSS_PERCENT = 0.08    # 止损阈值 (8%低于平均成本)
STOP_PROFIT_PERCENT = 0.15  # 止盈阈值 (15%高于平均成本)

# --- 指标计算辅助函数 ---

def calculate_technical_indicators(df):
    """ 计算V4.4所需的RSI(6)和MA趋势指标 """
    df_asc = df.copy()
    if 'value' not in df_asc.columns or len(df_asc) < 60:
        return {'RSI(6)': np.nan, 'MA50/MA250': np.nan, 'MA50/MA250趋势': '数据不足'}

    delta = df_asc['value'].diff()

    # 1. RSI (6) - V4.4 核心信号
    gain_6 = (delta.where(delta > 0, 0)).rolling(window=6, min_periods=1).mean()
    loss_6 = (-delta.where(delta < 0, 0)).rolling(window=6, min_periods=1).mean()
    rs_6 = gain_6 / loss_6.replace(0, np.nan) 
    df_asc['RSI_6'] = 100 - (100 / (1 + rs_6))
    rsi_6_latest = df_asc['RSI_6'].iloc[-1]
    
    # 2. 移动平均线和趋势分析 (V4.4 趋势风控)
    df_asc['MA50'] = df_asc['value'].rolling(window=50, min_periods=1).mean()
    df_asc['MA250'] = df_asc['value'].rolling(window=250, min_periods=1).mean() 
    
    ma50_latest = df_asc['MA50'].iloc[-1]
    ma250_latest = df_asc['MA250'].iloc[-1]
    
    ma50_to_ma250 = np.nan
    trend_direction = '数据不足'
    
    if len(df_asc) >= 250 and ma250_latest and ma250_latest != 0:
        ma50_to_ma250 = ma50_latest / ma250_latest
        
        # MA50/MA250 趋势方向判断
        recent_ratio = (df_asc['MA50'] / df_asc['MA250']).tail(20).dropna()
        if len(recent_ratio) >= 5:
            # 使用线性拟合的斜率来判断趋势
            slope = np.polyfit(np.arange(len(recent_ratio)), recent_ratio.values, 1)[0]
            if slope > 0.001: trend_direction = '向上'
            elif slope < -0.001: trend_direction = '向下'
            else: trend_direction = '平稳'

    return {
        'RSI(6)': round(rsi_6_latest, 2) if not math.isnan(rsi_6_latest) else np.nan,
        'MA50/MA250': round(ma50_to_ma250, 2) if not math.isnan(ma50_to_ma250) else np.nan, 
        'MA50/MA250趋势': trend_direction,
    }

def calculate_max_drawdown(series):
    """ 计算最大回撤 """
    if series.empty: return 0.0
    rolling_max = series.cummax()
    drawdown = (rolling_max - series) / rolling_max
    return drawdown.max()

# --- V5.0 新增指标功能函数 ---

def calculate_recovery_days(equity_series):
    """ V5.0: 计算从最大回撤谷底恢复到历史高点所需的天数。 """
    if equity_series.empty:
        return np.nan

    # 1. 找到历史最高点
    rolling_max = equity_series.cummax()
    
    # 2. 找到最大回撤的起点和谷底 (基于 drawdown series)
    drawdown = (rolling_max - equity_series) / rolling_max
    max_drawdown_index = drawdown.idxmax()

    # 如果最大回撤点在最后一天，则尚未修复
    if max_drawdown_index == equity_series.index[-1]:
        return np.nan 

    max_dd_peak = rolling_max.loc[:max_drawdown_index].max()
    
    # 3. 找到恢复到或超过历史最高点的第一天
    recovery_period = equity_series.loc[max_drawdown_index:]
    
    # 找到第一个大于或等于 max_dd_peak 的日期
    recovery_date = recovery_period[recovery_period >= max_dd_peak].index.min()
    
    if pd.isna(recovery_date):
        return np.nan # 尚未修复
    
    # 4. 计算天数 (日期差)
    trough_date = pd.to_datetime(max_drawdown_index)
    recovery_date = pd.to_datetime(recovery_date)
    
    # 结果为两个 datetime.date 之间的差值 (days)
    return (recovery_date - trough_date).days

# --- V5.0 核心回测逻辑 ---

def run_backtest_v5(df_fund, fund_code):
    """
    对单只基金运行 V4.4 网格补仓策略，并计算 V5.0 指标。
    """
    df = df_fund.copy()
    
    # 1. 筛选回测周期并计算指标
    df = df[(df['date'] >= BACKTEST_START_DATE) & (df['date'] <= BACKTEST_END_DATE)].copy()
    if df.empty or len(df) < 250:
        logging.warning(f"基金 {fund_code} 数据不足 250 条，跳过 V5.0 回测。")
        return None

    # V5.0 关键：对每一天的数据运行指标计算，避免未来函数
    df_tech = pd.DataFrame([calculate_technical_indicators(df.iloc[:i+1]) for i in range(len(df))])
    df = pd.concat([df.reset_index(drop=True), df_tech], axis=1)
    
    df = df.dropna(subset=['RSI(6)']).reset_index(drop=True)
    if df.empty: return None

    # 2. 初始化回测变量
    initial_capital = INITIAL_CAPITAL
    cash = initial_capital
    shares = 0.0        
    avg_cost_per_share = 0.0 
    
    # V5.0 指标跟踪
    take_profit_count = 0 
    stop_loss_count = 0
    
    trade_log = []
    equity_values = []
    
    # 3. 逐日回测
    for index, row in df.iterrows():
        current_date = row['date']
        current_value = row['value']
        current_rsi_6 = row['RSI(6)']
        ma_ratio = row['MA50/MA250']
        trend_dir = row['MA50/MA250趋势']
        
        market_value = shares * current_value
        total_equity = cash + market_value
        equity_values.append(total_equity)

        # --- 卖出判断 (止盈/止损) ---
        if shares > 0:
            current_holding_cost = shares * avg_cost_per_share
            current_profit_ratio = (market_value - current_holding_cost) / current_holding_cost
            
            # 止损信号:
            if current_profit_ratio <= -STOP_LOSS_PERCENT:
                sale_amount = market_value
                cash += sale_amount
                shares = 0.0
                avg_cost_per_share = 0.0
                stop_loss_count += 1 # V5.0 记录止损次数
                trade_log.append({'Date': current_date, 'Action': 'SELL (Stop Loss)'})
                continue 

            # 止盈信号:
            if current_profit_ratio >= STOP_PROFIT_PERCENT:
                sale_amount = market_value
                cash += sale_amount
                shares = 0.0
                avg_cost_per_share = 0.0
                take_profit_count += 1 # V5.0 记录止盈次数
                trade_log.append({'Date': current_date, 'Action': 'SELL (Take Profit)'})
                continue 
        
        # --- V4.4 买入判断 (网格 & 信号 & 趋势) ---
        
        # 1. 初始建仓（模拟任务驱动，仅执行一次，占总资金的约 10%）
        if shares == 0 and cash >= BUY_AMOUNT_PER_TRADE:
            buy_shares = BUY_AMOUNT_PER_TRADE / current_value
            shares += buy_shares
            avg_cost_per_share = current_value
            cash -= BUY_AMOUNT_PER_TRADE
            trade_log.append({'Date': current_date, 'Action': 'BUY (Initial)'})
            continue 
            
        # 2. 网格补仓（信号驱动）
        if shares > 0 and cash >= BUY_AMOUNT_PER_TRADE:
            
            # 2.1. 趋势安全垫过滤 (Level 3)
            if trend_dir == '向下' or ma_ratio < TREND_RATIO_MIN or math.isnan(ma_ratio):
                continue

            # 2.2. 价格到位 (网格触发 - Level 1)
            current_drop_from_avg = (avg_cost_per_share - current_value) / avg_cost_per_share
            if current_drop_from_avg < GRID_STEP_PERCENT:
                continue 

            # 2.3. 质量过滤 (RSI(6) 极值 - Level 2)
            if current_rsi_6 <= RSI_BUY_THRESHOLD:
                
                # 触发买入
                buy_shares = BUY_AMOUNT_PER_TRADE / current_value
                total_buy_cost = shares * avg_cost_per_share + BUY_AMOUNT_PER_TRADE
                shares += buy_shares
                avg_cost_per_share = total_buy_cost / shares
                cash -= BUY_AMOUNT_PER_TRADE
                
                trade_log.append({'Date': current_date, 'Action': 'BUY (Grid)'})

    # --- 最终结算与性能指标计算 ---
    
    final_equity = cash + shares * df['value'].iloc[-1] if not df.empty else initial_capital
    if equity_values:
        equity_values[-1] = final_equity
    
    df_equity = pd.Series(equity_values, index=df['date'])
    df_equity = df_equity.replace(0, np.nan).dropna()
    
    total_return = (final_equity - initial_capital) / initial_capital
    max_drawdown = calculate_max_drawdown(df_equity)
    
    years = (df_equity.index[-1] - df_equity.index[0]).days / 365.25 if not df_equity.empty else 0
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    daily_returns = df_equity.pct_change().dropna()
    annual_volatility = daily_returns.std() * np.sqrt(252)
    risk_free_rate = 0.02
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else np.nan

    # V5.0 新增指标计算
    total_sells = take_profit_count + stop_loss_count
    win_rate = take_profit_count / total_sells if total_sells > 0 else np.nan
    max_dd_recovery_days = calculate_recovery_days(df_equity)

    return {
        '基金代码': fund_code,
        '最终资产': round(final_equity, 2),
        '总收益率': round(total_return, 4),
        '最大回撤': round(max_drawdown, 4),
        '年化收益率': round(annual_return, 4),
        '夏普比率': round(sharpe_ratio, 2),
        '策略胜率': round(win_rate, 4),
        '最大回撤修复期 (天)': max_dd_recovery_days,
        '买入次数': len([t for t in trade_log if 'BUY' in t['Action']]),
        '止盈次数': take_profit_count,
        '止损次数': stop_loss_count,
    }

# --- 数据加载和报告生成函数 ---

def load_fund_data(filepath, fund_code):
    """ 加载和清洗数据 """
    try:
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='gbk')
    except Exception as e:
        logging.error(f"加载基金 {filepath} 失败: {e}")
        return None

    # 兼容您提供的文件格式
    if 'date' not in df.columns or 'net_value' not in df.columns:
        return None
        
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
    df = df.rename(columns={'net_value': 'value'})
    
    if len(df) < 250:
         return None
         
    return df

def generate_backtest_report(df_results):
    """ 生成 V5.0 回测报告 Markdown 文件 """
    report_parts = []
    
    report_parts.extend([
        f"# V5.0 网格策略回测报告 (穿越牛熊指标) ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n",
        f"**回测周期:** {BACKTEST_START_DATE} 至 {BACKTEST_END_DATE}\n",
        f"**策略:** V4.4 双重过滤网格 (V5.0 指标评估)\n",
        f"**买入信号 (需同时满足):**\n",
        f"1. **价格到位 (Level 1)**: 相对平均成本下跌 $\\ge {GRID_STEP_PERCENT*100:.0f}\\%$\n",
        f"2. **质量过滤 (Level 2)**: RSI(6) $\\le {RSI_BUY_THRESHOLD:.0f}$\n",
        f"3. **趋势过滤 (风控)**: MA50/MA250 $\\ge {TREND_RATIO_MIN:.2f}$ 且趋势非 '向下'\n",
        f"**风控:** 止损 $\\le -{STOP_LOSS_PERCENT*100:.0f}\\%$；止盈 $\\ge {STOP_PROFIT_PERCENT*100:.0f}\\%$；每次补仓 $\\yen {BUY_AMOUNT_PER_TRADE:.0f}$。\n\n",
        f"## 📊 总体性能指标 (按夏普比率降序排列)\n\n"
    ])

    TABLE_HEADER = "| 基金代码 | **夏普比率** | **年化收益率** | **策略胜率** | **修复期 (天)** | 最大回撤 | 总交易次数 |\n"
    TABLE_SEPARATOR = "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    report_parts.append(TABLE_HEADER)
    report_parts.append(TABLE_SEPARATOR)

    for index, row in df_results.iterrows():
        total_trades = int(row['买入次数']) + int(row['止盈次数']) + int(row['止损次数'])
        # 格式化修复期 (如果是 NaN，显示 ---)
        recovery_days = f"{int(row['最大回撤修复期 (天)'])}" if not pd.isna(row['最大回撤修复期 (天)']) else '---'
        
        report_parts.append(
            f"| `{row['基金代码']}` | **{row['夏普比率']:.2f}** | **{row['年化收益率']:.2%}** | **{row['策略胜率']:.2%}** | "
            f"{recovery_days} | {row['最大回撤']:.2%} | {total_trades} |\n"
        )
        
    with open(REPORT_FILE_NAME, 'w', encoding='utf-8') as f:
        f.write("".join(report_parts))
        
    logging.info(f"V5.0 回测完成，报告已保存到 {REPORT_FILE_NAME}")


def main_backtester():
    """ V5.0 回测主函数 """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info("--- V5.0 网格策略回测脚本启动 (穿越牛熊指标评估) ---")
    
    # 确保 fund_data 目录存在
    if not os.path.exists(FUND_DATA_DIR):
        logging.error(f"目录 '{FUND_DATA_DIR}' 不存在。请创建该目录并放入基金数据文件。")
        return

    csv_files = glob.glob(os.path.join(FUND_DATA_DIR, '*.csv'))
    if not csv_files:
        logging.error(f"在目录 '{FUND_DATA_DIR}' 中未找到CSV文件。")
        return

    results = []
    
    for filepath in csv_files:
        fund_code = os.path.splitext(os.path.basename(filepath))[0]
        logging.info(f"开始回测基金: {fund_code}...")
        
        df_fund = load_fund_data(filepath, fund_code)
        if df_fund is not None:
            backtest_result = run_backtest_v5(df_fund, fund_code)
            if backtest_result:
                results.append(backtest_result)
    
    if results:
        # 按夏普比率降序排序
        df_results = pd.DataFrame(results).sort_values(by='夏普比率', ascending=False)
        generate_backtest_report(df_results)
    else:
        logging.info("没有基金数据满足 V5.0 回测要求 (数据需 > 250 条)。")

if __name__ == '__main__':
    main_backtester()
    print("V5.0 回测脚本执行完毕。")
