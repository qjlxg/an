import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed 
import io 

# -------------------------------------------------------------------
# 基金适用的技术指标 (V2.9 改进：信号频率控制, 择时定投模拟)
# -------------------------------------------------------------------

def calculate_indicators(df, risk_free_rate_daily=0.0):
    """
    计算适用于基金数据的技术指标 (SMA, RSI, MACD, 波动率, 夏普比率, MDD)。
    【V2.8 修正】：对 RSI 和 MACD 严格设置最小数据窗口，数据不足时为 NaN。
    """
    
    df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
    df.dropna(subset=['net_value'], inplace=True)
    df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date']) # 确保日期是 datetime，便于定投模拟

    # 1. 收益率 (百分比表示)
    df['daily_return'] = df['net_value'].pct_change() * 100

    # 2. 简单移动平均线 (SMA)
    df['SMA_5'] = df['net_value'].rolling(window=5, min_periods=1).mean()
    df['SMA_20'] = df['net_value'].rolling(window=20, min_periods=1).mean()
    df['SMA_60'] = df['net_value'].rolling(window=60, min_periods=1).mean()
    
    # 3. 相对强弱指标 (RSI)
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = np.where(loss == 0, np.inf, gain / loss)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI'] = calculate_rsi(df['net_value'], window=14)
    df.loc[df.index < 13, 'RSI'] = np.nan 
    
    # 4. MACD
    exp1 = df['net_value'].ewm(span=12, adjust=False).mean()
    exp2 = df['net_value'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df.loc[df.index < 25, ['MACD', 'Signal_Line', 'MACD_Hist']] = np.nan
    
    # 5. 波动率 (20日标准差，年化)
    df['volatility'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)
    
    # 6. 夏普比率 (年化)
    df['sharpe_ratio'] = (df['daily_return'].rolling(window=252).mean() * 252 - risk_free_rate_daily * 252) / (df['daily_return'].rolling(window=252).std() * np.sqrt(252))
    df.loc[df['daily_return'].rolling(window=252).count() < 252, 'sharpe_ratio'] = np.nan
    
    # 7. 最大回撤 (MDD)
    df['cum_return'] = df['net_value'] / df['net_value'].iloc[0] 
    df['peak'] = df['net_value'].cummax()
    df['drawdown'] = (df['net_value'] - df['peak']) / df['peak']
    
    # 8. 相对历史位置
    max_value = df['net_value'].max()
    min_value = df['net_value'].min()
    range_value = max_value - min_value
    df['relative_position'] = np.where(range_value == 0, 0.5, (df['net_value'] - min_value) / range_value)
    
    # 填充 NaN 值
    cols_to_fill = ['SMA_5', 'SMA_20', 'SMA_60', 'volatility', 'peak', 'drawdown', 'relative_position'] 
    for col in cols_to_fill:
        df[col].fillna(method='bfill', inplace=True)
        df[col].fillna(0, inplace=True) 
    
    df[['RSI', 'MACD', 'Signal_Line']].fillna(0, inplace=True)

    return df

def generate_all_signals(df):
    """
    根据技术指标生成买入/卖出信号和原始评分。
    【V2.9 修正】：增加超低位抄底组合权重 (+8)，提高买入阈值到 6。
    """
    df['signal_score'] = 0
    
    # --- 0. 极值超低位抄底 (V2.9 新增，最高权重) ---
    # RSI <= 20 且 相对历史位置 <= 15% (极低位，权重 +8)
    df.loc[(df['RSI'] <= 20) & (df['relative_position'] <= 0.15), 'signal_score'] += 8
    
    # --- 1. 趋势与动量指标 (MACD/SMA) ---
    df.loc[(df['MACD'].shift(1) < df['Signal_Line'].shift(1)) & 
           (df['MACD'] >= df['Signal_Line']), 'signal_score'] += 2 
    df.loc[(df['MACD'].shift(1) > df['Signal_Line'].shift(1)) & 
           (df['MACD'] <= df['Signal_Line']), 'signal_score'] -= 2

    df.loc[(df['SMA_5'] > df['SMA_20']) & (df['SMA_20'] > df['SMA_60']), 'signal_score'] += 3
    df.loc[(df['SMA_5'] < df['SMA_20']) & (df['SMA_20'] < df['SMA_60']), 'signal_score'] -= 3
    
    # --- 2. 超买/超卖指标 (RSI) ---
    df.loc[df['RSI'] <= 25, 'signal_score'] += 4
    df.loc[df['RSI'] >= 75, 'signal_score'] -= 4
    
    # --- 3. 抄底/逃顶指标 ---
    df.loc[df['relative_position'] <= 0.2, 'signal_score'] += 6 
    df.loc[df['relative_position'] >= 0.8, 'signal_score'] -= 4
    df['latest_drawdown'] = df['drawdown'] * 100 
    df.loc[df['latest_drawdown'] <= -20, 'signal_score'] += 5 
    
    # --- 4. 风险调整指标 (夏普比率) ---
    df.loc[df['sharpe_ratio'] >= 1.0, 'signal_score'] += 2
    df.loc[df['sharpe_ratio'] < 0, 'signal_score'] -= 2

    # 交易信号 (买入/卖出)
    # 【V2.9 修正】：买入阈值提高到 6
    df['action_signal'] = np.where(df['signal_score'] >= 6, '买入', 
                                    np.where(df['signal_score'] <= -4, '卖出', '持有'))
    
    return df

def backtest_strategy(df, transaction_cost=0.001, stop_loss=-5.0, take_profit=10.0, position_increment=0.5):
    """
    (V2.8 修正) 基于评分信号的分批择时回测。
    - 每次买入信号，仓位增加 position_increment (默认 50%)。
    - 止损/止盈/策略卖出，仓位清零。
    """
    
    df['daily_return'] = pd.to_numeric(df['daily_return'], errors='coerce')
    
    positions = [] 
    trades = [] 
    
    position_size = 0.0 
    entry_value = 0.0 
    entry_date = None 

    for i in range(len(df)):
        date = df.loc[i, 'date']
        net_value = df.loc[i, 'net_value']
        signal_score = df.loc[i, 'signal_score']
        action_signal = df.loc[i, 'action_signal'] # V2.9: 使用 action_signal

        action = None
        exit_reason = None
        
        # --- 1. 持仓检查：止损/止盈/清仓 ---
        if position_size > 0:
            return_since_entry = (net_value / entry_value - 1) * 100
            
            if return_since_entry <= stop_loss:
                exit_reason = 'STOP_LOSS'
                action = '卖出 (止损)'
            elif return_since_entry >= take_profit:
                exit_reason = 'TAKE_PROFIT'
                action = '卖出 (止盈)'
            elif action_signal == '卖出': # V2.9: 触发卖出信号
                exit_reason = 'STRATEGY_SELL'
                action = '卖出 (信号)'
            
            if exit_reason:
                exit_value = net_value
                
                # 记录清仓交易事件
                trades.append({
                    'date': date, 'type': 'SELL', 'size_change': -position_size, 
                    'net_value': exit_value, 'reason': exit_reason,
                    'is_complete_trade': True,
                    'entry_date': entry_date, 'exit_date': date, 
                    'entry_net_value': entry_value, 'pnl_since_entry': return_since_entry
                })
                
                # 重置仓位状态
                position_size = 0.0
                entry_value = 0.0
                entry_date = None
        
        # --- 2. 无/不满仓检查：建仓/加仓 ---
        if position_size < 1.0:
            if action_signal == '买入': # V2.9: 触发买入信号 (score >= 6)
                increment = min(position_increment, 1.0 - position_size) 
                
                if position_size == 0.0:
                    entry_value = net_value 
                    entry_date = date
                
                position_size += increment
                action = f'买入 (+{increment*100:.0f}%)'
                
                # 记录加仓交易事件
                trades.append({
                    'date': date, 'type': 'BUY', 'size_change': increment, 
                    'net_value': net_value, 'reason': 'STRATEGY_BUY',
                    'is_complete_trade': False,
                    'entry_date': entry_date, 'exit_date': None, 
                    'entry_net_value': entry_value, 'pnl_since_entry': 0.0
                })
        
        positions.append(position_size)

    df['position'] = positions
    
    # 计算回测收益率
    df['strategy_return'] = df['daily_return'] * df['position'].shift(1) 
    df['strategy_return'].fillna(0, inplace=True) 

    trades_df = pd.DataFrame(trades)
    complete_trades_df = trades_df[trades_df['is_complete_trade']].copy()
    
    if not complete_trades_df.empty:
        # PNL = (退出净值 / 首次建仓净值 - 1) * 100 - 交易成本 (双边)
        complete_trades_df['pnl'] = (complete_trades_df['net_value'] / complete_trades_df['entry_net_value'] - 1) * 100 - 2 * transaction_cost * 100
        
        win_rate = (complete_trades_df['pnl'] > 0).sum() / len(complete_trades_df)
        avg_return = complete_trades_df['pnl'].mean()
        trades_count = len(complete_trades_df)
        
        total_pnl = (df['strategy_return']/100 + 1).prod() - 1
        
    else:
        win_rate = 0.0
        avg_return = 0.0
        total_pnl = 0.0
        trades_count = 0

    return win_rate, avg_return, total_pnl, trades_count, trades_df, stop_loss, take_profit

def simulate_monthly_invest(df, monthly_amount=5000):
    """
    【V2.9 新增】模拟基于信号的每月择时定投。
    只在每月出现 '买入' (score >= 6) 或 '强烈买入' (score >= 6) 信号时投入。
    """
    
    # 确保 'date' 是索引
    df = df.set_index('date')
    
    # 找出每月最后一个交易日的数据
    monthly_signals = df.resample('M').last().dropna(subset=['net_value'])
    
    total_invested = 0.0
    total_units = 0.0
    
    for _, row in monthly_signals.iterrows():
        # V2.9: 只有在 '买入' 或 '强烈买入' 信号时才投入
        if row['action_signal'] == '买入': 
            
            # 投入金额
            invest_amount = monthly_amount 
            
            # 购买份额
            units_purchased = invest_amount / row['net_value']
            
            total_invested += invest_amount
            total_units += units_purchased
            
    if total_invested == 0:
        return np.nan # 如果从未买入，返回 NaN

    # 最终价值 = 总份额 * 结束净值
    final_value = total_units * df['net_value'].iloc[-1]
    
    # 计算总收益率 (不考虑交易成本的简化模型)
    return (final_value / total_invested - 1) * 100


def generate_signal_and_score(df, monthly_invest_return):
    """
    根据最新数据点提取最终评分和信号
    【V2.9 修正】：接收 monthly_invest_return
    """
    if df.empty:
        return {'score': 0, 'signal': '数据不足', 'fund_name': 'N/A', 'latest_date': 'N/A',
                'latest_net_value': 0.0, 'latest_daily_return': 0.0, 'sharpe_ratio': np.nan,
                'relative_position': np.nan, 'latest_mdd': np.nan, 'current_drawdown': np.nan,
                'volatility': np.nan, 'monthly_invest_return': np.nan} 
    
    latest = df.iloc[-1]
    
    score = latest['signal_score']
    
    if score >= 8:
        signal = '超强抄底买入' # V2.9 对应新增的 +8 信号
    elif score >= 6:
        signal = '强烈买入' # V2.9 信号阈值
    elif score >= 4:
        signal = '可分批买入' # 仍有一定积极信号，但未达行动阈值
    elif score >= 1:
        signal = '观察/持有'
    elif score <= -6:
        signal = '强烈卖出/规避'
    elif score <= -4:
        signal = '卖出/规避' # V2.9 信号阈值
    elif score <= -1:
        signal = '弱卖出/规避'
    else:
        # 中间分数，根据趋势指标调整建议
        if latest['SMA_5'] > latest['SMA_20']:
            signal = '持有/观察'
        elif latest['SMA_5'] < latest['SMA_20']:
            signal = '等待回调'
        else:
            signal = '观察'

    current_drawdown = latest['drawdown'] * 100 
    latest_mdd = df['drawdown'].min() * 100
    
    start_date = df.iloc[0]['date'].strftime('%Y-%m-%d')
    end_date = latest['date'].strftime('%Y-%m-%d')
    time_span = f"{start_date} 至 {end_date}"
    
    cumulative_return = (latest['net_value'] / df.iloc[0]['net_value'] - 1) * 100
    
    return {
        'score': score,
        'signal': signal,
        'fund_name': latest['fund_name'],
        'latest_date': latest['date'].strftime('%Y-%m-%d'),
        'latest_net_value': latest['net_value'],
        'latest_daily_return': latest['daily_return'],
        'sharpe_ratio': latest['sharpe_ratio'],
        'relative_position': latest['relative_position'],
        'latest_mdd': latest_mdd,
        'current_drawdown': current_drawdown,
        'time_span': time_span,
        'cumulative_return': cumulative_return,
        'volatility': latest['volatility'], 
        'monthly_invest_return': monthly_invest_return, # V2.9 新增
        'sma5': latest['SMA_5'],
        'sma20': latest['SMA_20'],
        'sma60': latest['SMA_60'],
        'stop_loss': 5.0, 
        'take_profit': 10.0 
    }

def analyze_single_fund(file_path, risk_free_rate_daily_percent, transaction_cost):
    """
    分析单个基金数据的完整流程。
    【V2.9 修正】：增加定投模拟。
    """
    try:
        df = pd.read_csv(file_path)
        base_name = os.path.basename(file_path)
        fund_code = os.path.splitext(base_name)[0]
        fund_name = fund_code 
        df['fund_name'] = fund_name
        
        df['date'] = pd.to_datetime(df['date'])
        
        if df.shape[0] < 252: # 提高最小数据要求以计算夏普比率和定投
            return None 

        df = calculate_indicators(df, risk_free_rate_daily=risk_free_rate_daily_percent)
        df = generate_all_signals(df)
        
        # --- 运行回测和定投模拟 ---
        
        # 1. 择时回测 (分批)
        win_rate, avg_return, total_pnl, trades_count, trades_df, sl, tp = backtest_strategy(
            df, 
            transaction_cost=transaction_cost,
            stop_loss=-5.0, 
            take_profit=10.0,
            position_increment=0.5
        )
        
        # 2. 择时定投模拟 (V2.9 新增)
        # 复制 df 是因为 simulate_monthly_invest 内部会设置 index
        monthly_invest_return = simulate_monthly_invest(df.copy(), monthly_amount=5000)

        # 3. 生成最终报告结果
        analysis_result = generate_signal_and_score(df, monthly_invest_return)
        analysis_result['win_rate'] = win_rate
        analysis_result['avg_return'] = avg_return
        analysis_result['total_pnl'] = total_pnl
        analysis_result['trades'] = trades_count 
        analysis_result['stop_loss'] = -sl
        analysis_result['take_profit'] = tp
        analysis_result['fund_name'] = fund_name 

        return analysis_result
    
    except Exception as e:
        # 打印错误信息以便调试，但不中断进程
        # print(f"Error processing {file_path}: {e}")
        return None

def extract_fund_codes_from_markdown(markdown_content):
    """
    从 market_monitor_report.md 的内容中提取第一列的基金代码。
    """
    try:
        lines = markdown_content.split('\n')
        table_start_index = -1
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.startswith('|') and '-' in stripped_line and ':' in stripped_line and not re.search('[a-zA-Z]', stripped_line):
                table_start_index = i + 1
                break
        
        if table_start_index == -1:
            return []
            
        fund_codes = []
        for line in lines[table_start_index:]:
            line = line.strip()
            if not line.startswith('|') or line.startswith('#'):
                continue
            if re.search(r'^-+$', line.replace('|', '').strip()):
                continue
            
            parts = [p.strip() for p in line.split('|')]
            
            if len(parts) > 1:
                code = parts[1].strip()
                if code.isdigit(): 
                    fund_codes.append(code)
                    
        return fund_codes

    except Exception as e:
        return []


def main_analysis(fund_data_dir='fund_data/', report_path='market_monitor_report.md', risk_free_rate_annual=0.03, transaction_cost=0.001):
    """
    (V2.9 改进) 主分析函数：引入定投模拟，提高买入信号阈值。
    """
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
    except FileNotFoundError:
        print(f"错误：未找到报告文件 '{report_path}'。请确保文件存在。")
        return
    except Exception as e:
        print(f"读取报告文件 '{report_path}' 时发生错误: {e}")
        return

    fund_codes = extract_fund_codes_from_markdown(markdown_content)
    
    if not fund_codes:
        print("错误：未从报告中提取到任何有效的基金代码。")
        return

    fund_files = [os.path.join(fund_data_dir, f"{code}.csv") for code in fund_codes]
    # 过滤掉不存在的 CSV 文件
    fund_files = [f for f in fund_files if os.path.exists(f)]
    
    if not fund_files:
        # 只有在 fund_data_dir 存在，但没有对应文件时才警告
        if os.path.isdir(fund_data_dir):
            print(f"警告：根据报告中的基金代码，未在 '{fund_data_dir}' 目录下找到任何 CSV 文件。请确保数据已下载。")
        else:
            print(f"错误：基金数据目录 '{fund_data_dir}' 不存在。请检查路径。")
        return

    print(f"从报告中提取到 {len(fund_codes)} 个基金代码，其中 {len(fund_files)} 个有对应数据文件，开始并行分析...")

    analysis_results = []
    risk_free_rate_daily_percent = (risk_free_rate_annual / 252) * 100
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(
                analyze_single_fund,
                file_path,
                risk_free_rate_daily_percent,
                transaction_cost
            ): file_path
            for file_path in fund_files
        }
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                analysis_results.append(result)

    if not analysis_results:
        print("所有文件分析失败或数据不足，未生成报告。")
        return
        
    # 排序逻辑保持不变 (抄底优先)
    buy_signals = [x for x in analysis_results if x['score'] >= 6] # V2.9: 筛选 >= 6
    buy_signals.sort(key=lambda x: (x['relative_position'], x['current_drawdown'], -x['score'])) 
    other_signals = [x for x in analysis_results if x['score'] < 6] # V2.9: 筛选 < 6
    other_signals.sort(key=lambda x: (-x['score'], x['relative_position']))
    final_sorted_results = buy_signals + other_signals

    # --- 输出到 Markdown 文件 ---
    current_date = datetime.now()
    output_dir = current_date.strftime('%Y%m')
    output_filename = current_date.strftime('%Y%m%d%H%M%S.md')
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # V2.9 修正：移除 f-string 中的 \mathbf{...} 反斜杠，使用 **粗体** 代替
        f.write(f"# 基金投资分析报告 - {current_date.strftime('%Y年%m月%d日 %H:%M:%S')} (**V2.9 - 信号严格化/择时定投**)\n\n")
        f.write(f"**分析时间:** {current_date.strftime('%Y-%m-%d %H:%M:%S')} (UTC+0)\n")
        f.write(f"**总计分析基金数:** {len(analysis_results)} 个 (来自报告中提取的 {len(fund_codes)} 个代码)\n")
        # 修正：将 \mathbf{\ge 6} 替换为 **>= 6**
        f.write(f"**版本优化:** **买入信号阈值提高至 >= 6**；**新增择时定投模拟**；增加超强抄底组合 **+8** 权重。\n")
        f.write(f"**指标依据:** SMA, MACD, RSI(14), 年化波动率, 夏普比率, MDD。\n")
        
        if analysis_results:
             sl = analysis_results[0]['stop_loss']
             tp = analysis_results[0]['take_profit']
             # 修正：将 \mathbf{5.00} 替换为 **5.00**
             f.write(f"**回测策略 (择时):** 基于评分模型的分批择时回测 (买入 **>= 6**，增仓 50%)。\n")
             f.write(f"**回测风控:** **硬性止损 {sl:.2f}%%，硬性止盈 {tp:.2f}%%。**\n\n") # 使用 f-string 格式化
        
        # 修正：将 \mathbf{\ge 6} 替换为 **>= 6**
        f.write("## 基金综合排序及建议 (【评分 **>= 6** 且低位】基金优先排序)\n\n")
        
        # 【V2.9 报告列修正】：增加 择时定投收益
        f.write("| 排名 | 基金名称 | **评分** | **投资建议** | 相对历史位置 | MDD深度(%) | 最新回撤(%) | 最新净值 | 最新日涨幅(%) | 年化波动率(%) | 年化夏普比率 | **回测胜率(%)** | **交易次数** | **回测平均净收益率(%)** | **择时定投收益(%)** |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        for i, result in enumerate(final_sorted_results):
            rel_pos_percent = f"{result['relative_position'] * 100:.2f}%"
            sharpe_ratio = f"{result['sharpe_ratio']:.2f}" if not np.isnan(result['sharpe_ratio']) else "N/A"
            avg_return = f"{result['avg_return']:.2f}" if result['avg_return'] != 0 else "0.00"
            mdd = f"{result['latest_mdd']:.2f}" if not np.isnan(result['latest_mdd']) else "N/A"
            cd = f"{result['current_drawdown']:.2f}" if not np.isnan(result['current_drawdown']) else "N/A"
            
            win_rate_percent = f"{result['win_rate'] * 100:.2f}"
            trades_count = result['trades']
            volatility_percent = f"{result['volatility']:.2f}" if not np.isnan(result['volatility']) else "N/A"
            
            monthly_ret_percent = f"{result['monthly_invest_return']:.2f}" if not np.isnan(result['monthly_invest_return']) else "N/A"

            f.write(
                f"| {i+1} "
                f"| `{result['fund_name']}` "
                f"| **{result['score']}** "
                f"| **{result['signal']}** "
                f"| {rel_pos_percent} "
                f"| {mdd} "
                f"| {cd} "
                f"| {result['latest_net_value']:.4f} "
                f"| {result['latest_daily_return']:.2f} "
                f"| {volatility_percent} " 
                f"| {sharpe_ratio} "
                f"| **{win_rate_percent}** " 
                f"| **{trades_count}** "
                f"| {avg_return} "
                f"| **{monthly_ret_percent}** |\n"
            )

        f.write("\n## 附录：指标详情与说明\n\n")
        f.write("### 投资建议说明\n")
        # 修正：将 \mathbf{...} 替换为 **粗体**，将 \to 替换为 ->
        f.write("- **【重要】列表已优先展示** **评分 >= 6** **的基金，并按** **相对历史位置 -> 最新回撤深度 -> 评分** **的顺序排序。**\n")
        f.write("- **回测策略 (择时)**：采用**分批建仓**模型，每次买入信号（评分 $\\ge 6$）增仓 50%。\n")
        f.write("- **择时定投收益(%)**: **【新】** 模拟每月检查信号，只有在出现 **'买入'** 或 **'强烈买入'** 信号时才投入，计算其总收益率。\n")
        f.write("- **年化波动率(%)**: 风险水平指标。\n")
        f.write("- **年化夏普比率**: 风险调整后的年化收益。\n")
        # 修正：将 \mathbf{%.2f} 替换为 **%.2f**
        f.write("- **回测平均净收益率**: 每笔完整择时交易所获得的平均净收益，已扣除 **%.2f**%% 的往返交易成本。\n" % (transaction_cost*2*100))
        f.write("\n### 数据范围\n")
        for result in final_sorted_results:
            f.write(f"- `{result['fund_name']}`: 数据日期 {result['latest_date']}，范围 {result['time_span']} (累计涨幅: {result['cumulative_return']:.2f}%) (SMA: {result['sma5']:.4f}/{result['sma20']:.4f}/{result['sma60']:.4f})\n")

        
    print(f"\n分析完成! 报告已输出到: {output_path}")

# --- 脚本执行入口 ---
if __name__ == "__main__":
    main_analysis(report_path='market_monitor_report.md', risk_free_rate_annual=0.03, transaction_cost=0.001)