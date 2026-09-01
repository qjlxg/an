import pandas as pd
import numpy as np
import os
from datetime import datetime
from sell_decision import load_config, calculate_indicators, get_big_market_status, decide_sell

# --- 回测配置 ---
START_DATE = '2022-01-01'  # 回测开始日期
END_DATE = datetime.now().strftime('%Y-%m-%d') # 回测结束日期

# --- 历史回测核心逻辑 ---
def run_backtest(fund_code, initial_cost_nav, params, fund_df, big_market_data, big_trend_df):
    
    # 过滤回测日期范围
    fund_df = fund_df[(fund_df['date'] >= START_DATE) & (fund_df['date'] <= END_DATE)].copy()
    if fund_df.empty:
        print(f"警告: 基金 {fund_code} 在回测期内无数据。")
        return None

    # 初始化持仓和交易记录
    trade_log = []
    current_cost_nav = initial_cost_nav
    current_shares = 1000 # 假设初始投入1000份
    initial_cash = current_shares * initial_cost_nav
    current_peak_nav = initial_cost_nav # 初始峰值

    # 循环模拟每日决策
    for i in range(1, len(fund_df)):
        current_date = fund_df.iloc[i]['date']
        
        # 1. 提取当前日期之前的所有历史数据
        historical_data = fund_df.iloc[:i+1].copy()
        
        # 2. 计算截至当日的指标
        # ⚠️ 注意：每次迭代都重新计算所有指标，这是模拟真实交易的必要开销
        historical_data = calculate_indicators(
            historical_data, 
            params.get('rsi_window', 14), 
            params.get('ma_window', 50), 
            params.get('bb_window', 20), 
            params.get('adx_window', 14)
        )
        
        latest_nav_value = historical_data.iloc[-1]['net_value']
        
        # 3. 模拟当日持仓状态
        # 模拟当前持仓状态 (回测期间，成本净值是动态的)
        current_peak_nav = max(current_peak_nav, latest_nav_value)
        
        value = current_shares * latest_nav_value
        cost = current_shares * current_cost_nav
        profit = value - cost
        profit_rate = (profit / cost) * 100 if cost > 0 else 0
        
        holding = {
            'value': value,
            'cost_nav': current_cost_nav,
            'shares': current_shares,
            'latest_net_value': latest_nav_value,
            'profit': profit,
            'profit_rate': profit_rate,
            'current_peak': current_peak_nav
        }
        
        # 4. 获取大盘当日状态 (回测时，大盘趋势判断应基于该日数据)
        # ⚠️ 简化处理：回测中我们直接使用 big_market_data 的当日指标
        big_market_latest = big_market_data[big_market_data['date'] == current_date].iloc[-1] if current_date in big_market_data['date'].values else big_market_data.iloc[-1] # 使用最近的
        big_trend = big_trend_df[big_trend_df['date'] == current_date].iloc[-1]['trend'] if current_date in big_trend_df['date'].values else '中性' # 假设趋势
        
        # 5. 做出决策
        decision_result = decide_sell(fund_code, holding, historical_data, params, big_market_latest, big_market_data, big_trend)
        decision = decision_result['decision']

        # 6. 记录交易日志 (简化处理：只记录卖出/暂停信号，不执行复杂的份额增减)
        if '卖' in decision or '暂停' in decision:
            trade_log.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                'Net_Value': round(latest_nav_value, 4),
                'Cost_Nav': round(current_cost_nav, 4),
                'Profit_Rate(%)': round(profit_rate, 2),
                'Decision': decision,
                'RSI': decision_result['rsi'],
                'MACD': decision_result['macd_signal'],
                'BB': decision_result['bb_pos'],
                'Big_Trend': big_trend,
                'Reason': decision # 简化，Decision即为Reason
            })

    # 7. 最终回测报告 (只返回交易日志)
    return pd.DataFrame(trade_log)

def main():
    print(f"--- 历史回测模块启动 ({START_DATE} 至 {END_DATE}) ---")
    
    # 1. 加载配置和参数
    params, holdings_config = load_config()
    
    # 2. 预加载大盘数据 (用于趋势判断)
    big_market_data, _, _ = get_big_market_status(params)
    
    # 预计算大盘趋势DF (简化回测循环中的趋势获取)
    big_trend_df = big_market_data[['date']].copy()
    big_trend_df['net_value'] = big_market_data['net_value']
    big_trend_df['ma50'] = big_market_data['ma50']
    big_trend_df['rsi'] = big_market_data['rsi']
    big_trend_df['trend'] = np.where(
        (big_trend_df['rsi'] > 50) & (big_trend_df['net_value'] > big_trend_df['ma50']), '强势',
        np.where(
            (big_trend_df['rsi'] < 50) & (big_trend_df['net_value'] < big_trend_df['ma50']), '弱势',
            '中性'
        )
    )
    
    # 3. 运行回测
    all_results = {}
    fund_data_dir = 'fund_data/'

    for code, cost_nav in holdings_config.items():
        fund_file = os.path.join(fund_data_dir, f"{code}.csv")
        if os.path.exists(fund_file):
            fund_df = pd.read_csv(fund_file, parse_dates=['date']).sort_values('date').reset_index(drop=True)
            print(f"开始回测基金: {code} (初始成本净值: {cost_nav})")
            
            # 使用初始成本净值作为回测的起点
            initial_cost_nav = float(cost_nav)
            
            backtest_log = run_backtest(code, initial_cost_nav, params, fund_df, big_market_data, big_trend_df)
            
            if backtest_log is not None and not backtest_log.empty:
                all_results[code] = backtest_log
            else:
                print(f"基金 {code} 没有产生交易信号。")

    # 4. 汇总并输出结果
    if all_results:
        # 合并所有基金的回测日志
        final_df = pd.concat([df.assign(Fund_Code=code) for code, df in all_results.items()], ignore_index=True)
        final_df = final_df.sort_values(['Fund_Code', 'Date'])
        
        output_filename = 'backtest_decision_log.csv'
        final_df.to_csv(output_filename, index=False, encoding='utf_8_sig')
        print(f"\n✅ 历史回测结果已生成 CSV 文件: {output_filename}")
    else:
        print("\n⚠️ 所有基金回测均失败或未产生交易信号，未生成结果文件。")

if __name__ == '__main__':
    main()
