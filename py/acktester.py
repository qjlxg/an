# backtester.py

import pandas as pd
import glob
import os
import numpy as np
import logging
import math
from datetime import datetime

# --- 配置参数 (基于原脚本进行回测优化) ---
FUND_DATA_DIR = 'fund_data'
EXTREME_RSI_THRESHOLD_P1 = 29.0  # 买入信号 RSI 阈值
STOP_LOSS_PERCENT = 0.08         # 止损阈值 (8%)
STOP_PROFIT_PERCENT = 0.15       # 止盈阈值 (15%)
BACKTEST_START_DATE = '2020-01-01' # 回测起始日期
BACKTEST_END_DATE = '2024-12-31'   # 回测结束日期
INITIAL_CAPITAL = 100000.0       # 初始资金 (元)
BUY_AMOUNT_PER_TRADE = 10000.0   # 每次买入金额 (元)
REPORT_FILE_NAME = 'fund_backtest_report.md'

# --- 复用原脚本的技术指标计算函数 (简化版，仅保留必要逻辑) ---
# 警告: 实际回测中，这些函数应从 analyzer.py 中导入。这里为独立脚本演示，直接复制关键函数。

def calculate_technical_indicators(df):
    """ 计算RSI(14)和当日涨跌幅，用于回测信号。 """
    df_asc = df.copy()

    if 'value' not in df_asc.columns or len(df_asc) < 60:
        df_asc['RSI_14'] = np.nan
        df_asc['Daily_Drop'] = np.nan
        return df_asc

    delta = df_asc['value'].diff()

    # 1. RSI (14)
    gain_14 = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss_14 = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs_14 = gain_14 / loss_14.replace(0, np.nan) 
    df_asc['RSI_14'] = 100 - (100 / (1 + rs_14))
    
    # 2. 当日涨跌幅
    df_asc['Daily_Drop'] = df_asc['value'].pct_change()
        
    return df_asc

def calculate_max_drawdown(series):
    """ 计算最大回撤 """
    if series.empty: return 0.0
    rolling_max = series.cummax()
    drawdown = (rolling_max - series) / rolling_max
    return drawdown.max()

# --- 核心回测逻辑 ---

def run_backtest(df_fund, fund_code):
    """
    对单只基金运行回测策略。
    策略：RSI(14) <= EXTREME_RSI_THRESHOLD_P1 时，买入固定金额。
          达到止盈或止损时，卖出所有持仓。
    """
    df = df_fund.copy()
    
    # 1. 筛选回测周期
    df = df[(df['date'] >= BACKTEST_START_DATE) & (df['date'] <= BACKTEST_END_DATE)].copy()
    if df.empty:
        logging.warning(f"基金 {fund_code} 在回测周期内没有数据。")
        return None

    # 2. 计算所需指标
    df = calculate_technical_indicators(df)
    df = df.dropna(subset=['RSI_14']).reset_index(drop=True)
    if df.empty: return None

    # 3. 初始化回测变量
    initial_capital = INITIAL_CAPITAL
    cash = initial_capital
    shares = 0.0        # 持有份额
    avg_cost_per_share = 0.0 # 平均持仓成本（每份额）
    
    trade_log = []
    equity_values = []
    
    # 4. 逐日回测
    for index, row in df.iterrows():
        current_date = row['date']
        current_value = row['value']
        current_rsi = row['RSI_14']
        
        # 计算当前总资产 (净值 * 份额 + 现金)
        market_value = shares * current_value
        total_equity = cash + market_value
        equity_values.append(total_equity)

        # --- 卖出判断 (止盈/止损) ---
        if shares > 0:
            # 当前持仓成本
            current_holding_cost = shares * avg_cost_per_share
            # 当前收益率: (现值 - 成本) / 成本
            current_profit_ratio = (market_value - current_holding_cost) / current_holding_cost
            
            # 止损信号: 跌幅 >= 8% (STOP_LOSS_PERCENT)
            if current_profit_ratio <= -STOP_LOSS_PERCENT:
                sale_amount = market_value
                cash += sale_amount
                trade_log.append({
                    'Date': current_date, 'Action': 'SELL (Stop Loss)', 
                    'Shares': shares, 'Value': current_value,
                    'Gain_Ratio': current_profit_ratio, 'Equity': total_equity
                })
                shares = 0.0
                avg_cost_per_share = 0.0
                continue # 完成交易，跳过当日买入判断

            # 止盈信号: 涨幅 >= 15% (STOP_PROFIT_PERCENT)
            if current_profit_ratio >= STOP_PROFIT_PERCENT:
                sale_amount = market_value
                cash += sale_amount
                trade_log.append({
                    'Date': current_date, 'Action': 'SELL (Take Profit)', 
                    'Shares': shares, 'Value': current_value,
                    'Gain_Ratio': current_profit_ratio, 'Equity': total_equity
                })
                shares = 0.0
                avg_cost_per_share = 0.0
                continue # 完成交易，跳过当日买入判断
        
        # --- 买入判断 (RSI极值) ---
        # 条件：RSI 超卖 AND 仍有现金 AND 当前没有持仓 (简化：一次性买入，卖出后才能再次买入)
        if current_rsi <= EXTREME_RSI_THRESHOLD_P1 and cash >= BUY_AMOUNT_PER_TRADE and shares == 0:
            buy_shares = BUY_AMOUNT_PER_TRADE / current_value
            
            # 更新成本和份额
            total_buy_cost = shares * avg_cost_per_share + BUY_AMOUNT_PER_TRADE
            shares += buy_shares
            avg_cost_per_share = total_buy_cost / shares
            cash -= BUY_AMOUNT_PER_TRADE
            
            trade_log.append({
                'Date': current_date, 'Action': 'BUY', 
                'Shares': buy_shares, 'Value': current_value,
                'RSI': current_rsi, 'Equity': total_equity
            })

    # --- 最终结算 ---
    # 如果回测结束时仍有持仓，则以最后一日净值清仓
    final_equity = cash + shares * df['value'].iloc[-1]
    equity_values[-1] = final_equity # 修正最后一天的总资产
    
    # 5. 性能指标计算
    df_equity = pd.Series(equity_values, index=df['date'])
    df_equity = df_equity.replace(0, np.nan).dropna() # 避免初始0值影响计算
    
    total_return = (final_equity - initial_capital) / initial_capital
    max_drawdown = calculate_max_drawdown(df_equity)
    
    # 简化年化收益率和夏普比率计算 (假设 252 个交易日)
    years = (df_equity.index[-1] - df_equity.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # 计算每日收益率并计算夏普比率 (假设无风险利率为 2%)
    daily_returns = df_equity.pct_change().dropna()
    annual_volatility = daily_returns.std() * np.sqrt(252)
    risk_free_rate = 0.02
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else np.nan

    return {
        '基金代码': fund_code,
        '起始资金': initial_capital,
        '最终资产': round(final_equity, 2),
        '总收益率': round(total_return, 4),
        '最大回撤': round(max_drawdown, 4),
        '年化收益率': round(annual_return, 4),
        '夏普比率': round(sharpe_ratio, 2),
        '交易次数': len([t for t in trade_log if t['Action'] != 'BUY']) # 只统计卖出次数
    }

# --- 数据加载与主控函数 ---

def load_fund_data(filepath, fund_code):
    """ 加载和清洗数据 (与 analyzer.py 逻辑相似) """
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='gbk')
    except Exception as e:
        logging.error(f"加载基金 {filepath} 失败: {e}")
        return None

    if 'date' not in df.columns or 'net_value' not in df.columns:
        return None
        
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
    df = df.rename(columns={'net_value': 'value'})
    
    if len(df) < 250: # 至少需要一年的数据进行有效回测
         logging.warning(f"基金 {fund_code} 数据不足 250 条，跳过回测。")
         return None
         
    return df

def main_backtester():
    """ 回测主函数 """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info("--- 基金超卖回测脚本启动 ---")
    
    csv_files = glob.glob(os.path.join(FUND_DATA_DIR, '*.csv'))
    if not csv_files:
        logging.error(f"在目录 '{FUND_DATA_DIR}' 中未找到CSV文件。请确保数据已放置。")
        return

    results = []
    
    for filepath in csv_files:
        fund_code = os.path.splitext(os.path.basename(filepath))[0]
        logging.info(f"开始回测基金: {fund_code}...")
        
        df_fund = load_fund_data(filepath, fund_code)
        if df_fund is not None:
            backtest_result = run_backtest(df_fund, fund_code)
            if backtest_result:
                results.append(backtest_result)
    
    if results:
        df_results = pd.DataFrame(results).sort_values(by='总收益率', ascending=False)
        generate_backtest_report(df_results)
    else:
        logging.info("没有基金数据满足回测要求。")

def generate_backtest_report(df_results):
    """ 生成回测报告 Markdown 文件 """
    report_parts = []
    
    report_parts.extend([
        f"# 基金超卖策略回测报告 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n",
        f"**回测周期:** {BACKTEST_START_DATE} 至 {BACKTEST_END_DATE}\n",
        f"**策略:** RSI(14) $\\le {EXTREME_RSI_THRESHOLD_P1:.0f}$ 时买入 $\\yen {BUY_AMOUNT_PER_TRADE:.0f}$。\n",
        f"**风控:** 止损 $\\le -{STOP_LOSS_PERCENT*100:.0f}\\%$；止盈 $\\ge {STOP_PROFIT_PERCENT*100:.0f}\\%$。\n\n",
        f"## 📊 总体性能指标\n\n"
    ])

    TABLE_HEADER = "| 基金代码 | 最终资产 (¥) | **总收益率** | **年化收益率** | 最大回撤 | 夏普比率 | 交易次数 |\n"
    TABLE_SEPARATOR = "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    report_parts.append(TABLE_HEADER)
    report_parts.append(TABLE_SEPARATOR)

    for index, row in df_results.iterrows():
        # 突出显示收益率最高的基金
        gain_display = f"**{row['总收益率']:.2%}**"
        annual_gain_display = f"**{row['年化收益率']:.2%}**"
        
        report_parts.append(
            f"| `{row['基金代码']}` | {row['最终资产']:.2f} | {gain_display} | {annual_gain_display} | "
            f"{row['最大回撤']:.2%} | {row['夏普比率']:.2f} | {int(row['交易次数'])} |\n"
        )
        
    with open(REPORT_FILE_NAME, 'w', encoding='utf-8') as f:
        f.write("".join(report_parts))
        
    logging.info(f"回测完成，报告已保存到 {REPORT_FILE_NAME}")


if __name__ == '__main__':
    # 注意：运行此脚本前，您需要创建 'fund_data' 目录并放入 CSV 数据文件。
    main_backtester()
    print("回测脚本执行完毕。")
