import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed # 引入并行处理模块

# -------------------------------------------------------------------
# 基金适用的技术指标 (V2.6 抄底强化：权重调整, MDD计算)来源:recommended_cn_funds.csv 
# -------------------------------------------------------------------

def calculate_indicators(df, risk_free_rate_daily=0.0, annualize_sharpe=False):
    """
    计算适用于基金数据的技术指标 (SMA, RSI, MACD, 波动率, 夏普比率, MDD)。
    与 V2.5 逻辑保持一致，新增 MDD 计算。
    """
    
    # 确保 'net_value' 是数值类型并按日期排序
    df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
    df.dropna(subset=['net_value'], inplace=True)
    df = df.sort_values(by='date', ascending=True).reset_index(drop=True)

    # 1. 收益率 (百分比表示)
    df['daily_return'] = df['net_value'].pct_change() * 100

    # 2. 简单移动平均线 (SMA)
    df['SMA_5'] = df['net_value'].rolling(window=5, min_periods=1).mean()
    df['SMA_20'] = df['net_value'].rolling(window=20, min_periods=1).mean()
    df['SMA_60'] = df['net_value'].rolling(window=60, min_periods=1).mean()
    
    # 【新增】SMA_250 用于长期趋势过滤
    df['SMA_250'] = df['net_value'].rolling(window=250, min_periods=1).mean() 

    # 3. 相对强弱指数 (RSI) - 周期14
    window = 14
    delta = df['net_value'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    RS = gain / loss.replace(0, np.nan)
    
    df['RSI'] = 100 - (100 / (1 + RS))
    df['RSI'] = df['RSI'].fillna(50.0)
    
    # 4. 移动平均收敛/发散指标 (MACD)
    exp1 = df['net_value'].ewm(span=12, adjust=False, min_periods=12).mean()
    exp2 = df['net_value'].ewm(span=26, adjust=False, min_periods=26).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

    # 5. 波动率 (20日标准差)
    df['Volatility'] = df['daily_return'].rolling(window=20, min_periods=1).std()

    # 6. 夏普比率 (基于20日收益率，考虑无风险利率)
    daily_excess_return = df['daily_return'].rolling(window=20, min_periods=1).mean() - risk_free_rate_daily
    df['Sharpe_Ratio'] = (daily_excess_return / 
                          df['Volatility']).replace([np.inf, -np.inf], np.nan)

    if annualize_sharpe:
        df['Sharpe_Ratio'] = df['Sharpe_Ratio'] * np.sqrt(252)
        
    # 7. 【V2.6 新增】最大回撤 (Max Drawdown - MDD)
    # 计算累计最高净值 (高水位线)
    df['hwm'] = df['net_value'].expanding().max()
    # 计算当前回撤
    df['drawdown'] = (df['net_value'] / df['hwm'] - 1) * 100
    # 计算最大回撤 (在观察期内)
    df['mdd'] = df['drawdown'].expanding().min()
    
    # 提取最新的最大回撤深度 (百分比，负值)
    df['latest_mdd'] = df['mdd'].iloc[-1]
    
    # 【V2.6 新增】最新回撤深度 (最新值，百分比，负值)
    df['current_drawdown'] = df['drawdown'].iloc[-1]

    return df

def generate_all_signals(df):
    """
    【V2.6 抄底强化】调整评分权重，强化相对位置和MDD，弱化趋势指标。
    """
    
    # 初始化评分列
    df['score'] = 0
    
    # 1. 均线趋势 (SMA) - 【权重弱化】(V2.5 +/- 4 -> V2.6 +/- 2)
    # SMA_5 > SMA_20 (+1), SMA_5 < SMA_20 (-1)
    df['score'] += np.where(df['SMA_5'] > df['SMA_20'], 1, 0)
    df['score'] -= np.where(df['SMA_5'] < df['SMA_20'], 1, 0)
    
    # SMA_20 > SMA_60 (+1), SMA_20 < SMA_60 (-1)
    df['score'] += np.where(df['SMA_20'] > df['SMA_60'], 1, 0)
    df['score'] -= np.where(df['SMA_20'] < df['SMA_60'], 1, 0)

    # 2. MACD 动量 - 【权重弱化】(V2.5 +/- 2 -> V2.6 +/- 1)
    # MACD_Hist > 0 (+1), MACD_Hist < 0 (-1)
    df['score'] += np.where(df['MACD_Hist'] > 0, 1, 0)
    df['score'] -= np.where(df['MACD_Hist'] < 0, 1, 0)
    
    # 3. RSI 超买/超卖 - 评分不变 (+/- 2)
    # V2.4 优化：RSI < 25 (+2), RSI > 75 (-2) 
    df['score'] += np.where(df['RSI'] < 25, 2, 0)
    df['score'] -= np.where(df['RSI'] > 75, 2, 0)

    # 4. 净值相对位置 - 【权重最大化】以增加逆向信号的强度 (V2.5 +/- 3 -> V2.6 +/- 6)
    df['min_value'] = df['net_value'].expanding().min()
    df['max_value'] = df['net_value'].expanding().max()
    df['range'] = df['max_value'] - df['min_value']
    
    # 相对位置计算 (向量化)
    df['relative_position_dynamic'] = np.where(
        df['range'] > 0,
        (df['net_value'] - df['min_value']) / df['range'],
        0.5 # 极值相等或数据不足时设为中性
    )
    
    # 评分基于相对位置 (向量化)
    # V2.6 抄底强化：< 0.3 (+6), > 0.8 (-6)
    df['score'] += np.where(df['relative_position_dynamic'] < 0.3, 6, 0)
    df['score'] -= np.where(df['relative_position_dynamic'] > 0.8, 6, 0)
    
    # 5. 【V2.6 新增】最大回撤 (MDD) 评分 - 权重强化 (+5)
    # 只有当最新回撤（Current Drawdown）超过历史最大回撤（MDD）的 70% 时，才加分。
    # 目标是抄底深度回撤，而不是仅仅最大回撤大。
    
    # 提取最新的历史最大回撤和当前回撤
    latest_mdd_percent = df['latest_mdd'].iloc[-1]
    current_drawdown_percent = df['current_drawdown'].iloc[-1]
    
    # 仅在有足够数据且最大回撤深度超过阈值时才计算
    if (len(df) > 60 and 
        not np.isnan(latest_mdd_percent) and 
        latest_mdd_percent < -5.0 # 至少有 5% 的回撤深度才考虑
       ):
        
        # 假设：如果当前回撤深度 > 历史最大回撤深度的 70% (即回撤还在底部区域)
        # 例如: MDD = -30%, 当前回撤 = -25% ( > 0.7*(-30%)) -> 处于深位
        if current_drawdown_percent < latest_mdd_percent * 0.7: 
             # 处于历史深度回撤区域，加分
             # V2.6 抄底强化：+5分 (或 +10分，取决于V2.6的完整MDD逻辑)
             df['score'].iloc[-1] += 5 
        # Note: 由于向量化困难且评分是基于历史回撤的，这里仅对【最新日】进行评分调整

    # 6. 波动率和夏普比率调整 (分位数依赖动态历史数据，使用优化的循环结构)
    start_index = 60 if len(df) > 60 else 1 
    
    for i in range(start_index, len(df)):
        current_df = df.iloc[:i+1]
        latest_volatility = current_df.iloc[-1]['Volatility']
        latest_sharpe = current_df.iloc[-1]['Sharpe_Ratio']
        
        # 波动率调整 - 权重不变 (+/- 1)
        volatility_series = current_df['Volatility'].dropna()
        if len(volatility_series) > 10 and not np.isnan(latest_volatility):
            q25_vol = volatility_series.quantile(0.25)
            q75_vol = volatility_series.quantile(0.75)
            if latest_volatility < q25_vol:
                df.loc[i, 'score'] += 1 # 低波动率加分
            elif latest_volatility > q75_vol:
                df.loc[i, 'score'] -= 1 # 高波动率减分

        # 夏普比率调整 - 权重强化不变 (+/- 4)
        sharpe_series = current_df['Sharpe_Ratio'].dropna()
        if len(sharpe_series) > 10 and not np.isnan(latest_sharpe):
            q25_sharpe = sharpe_series.quantile(0.25)
            q75_sharpe = sharpe_series.quantile(0.75)
            if latest_sharpe > q75_sharpe:
                df.loc[i, 'score'] += 4 # 高风险调整收益加分
            elif latest_sharpe < q25_sharpe:
                df.loc[i, 'score'] -= 4 # 低风险调整收益减分
                
    # 7. 生成最终信号 (向量化) - 阈值不变
    # V2.4 优化买入阈值: >= 4, 强烈买入 >= 8 
    df['signal'] = np.select(
        [
            df['score'] >= 8, # 强烈买入：从 6 提高到 8
            (df['score'] >= 4) & (df['score'] < 8), # 买入：从 2 提高到 4
            df['score'] <= -8, # 强烈卖出：从 -6 降低到 -8
            (df['score'] <= -4) & (df['score'] > -8) # 卖出：从 -2 降低到 -4
        ],
        [
            "强烈买入",
            "买入",
            "强烈卖出",
            "卖出"
        ],
        default="观望"
    )
    
    # 8. 【V2.6 新增】MDD 评分调整后的【最新日】信号二次检查 (如果 MDD 触发加分，则优先加仓)
    if (len(df) > start_index and 
        df['score'].iloc[-1] >= 4 and # 至少是买入信号
        df['score'].iloc[-1] > df['score'].iloc[-2] and # 相比前一日评分增加
        current_drawdown_percent < latest_mdd_percent * 0.7 # 处于深度回撤区域
       ):
        # 如果评分已经够高，且是深度回撤，确保信号是买入/强烈买入
        if df['score'].iloc[-1] >= 8:
            df['signal'].iloc[-1] = "强烈买入"
        elif df['score'].iloc[-1] >= 4:
            df['signal'].iloc[-1] = "买入"


    # 对于数据不足的部分 (如前60天), 信号设为 "数据不足"
    df.loc[df.index < start_index, 'signal'] = "数据不足"
    df.loc[df.index < start_index, 'score'] = 0.0
    
    # 清理中间列
    df.drop(columns=['min_value', 'max_value', 'range', 'hwm', 'drawdown'], errors='ignore', inplace=True)

    return df


def backtest_strategy(df, transaction_cost=0.001, stop_loss_percent=10.0, take_profit_percent=20.0):
    """
    【V2.4 优化】基于预先计算好的信号进行回测，引入硬性止损和止盈机制。
    (V2.6 逻辑不变)
    """
    signals = []
    returns = []
    position = 0  # 0: 无持仓, 1: 持仓
    buy_price = 0
    buy_date = None # 记录买入日期，用于止损/止盈检查
    
    # 总交易成本：买入+卖出，以收益率百分比表示
    round_trip_cost_percent = transaction_cost * 100 * 2
    
    # 从有足够计算指标数据的点开始回测
    start_index = df[df['signal'] != '数据不足'].index.min()
    start_index = start_index if not pd.isna(start_index) else 1 
    
    # 增加硬性止损/止盈的百分比阈值
    stop_loss_threshold = 1.0 - (stop_loss_percent / 100.0) # 例如 0.9
    take_profit_threshold = 1.0 + (take_profit_percent / 100.0) # 例如 1.2

    for i in range(start_index, len(df)):
        signal = df.iloc[i]['signal']
        current_price = df.iloc[i]['net_value']
        current_date = df.iloc[i]['date']

        # --- 止损/止盈检查 (优先级最高) ---
        if position == 1:
            sell_reason = None
            if current_price <= buy_price * stop_loss_threshold:
                sell_reason = 'Stop Loss'
            elif current_price >= buy_price * take_profit_threshold:
                sell_reason = 'Take Profit'
            
            # 如果触发止损或止盈，则立即平仓
            if sell_reason:
                position = 0
                sell_price = current_price
                
                # 计算净收益率并扣除双边交易成本
                gross_return = (sell_price - buy_price) / buy_price * 100
                trade_return = gross_return - round_trip_cost_percent
                
                returns.append(trade_return)
                signals.append(('Sell (Hard Exit: ' + sell_reason + ')', current_date, sell_price, buy_date))
                
                buy_date = None # 清空买入日期
                continue

        # --- 信号操作 ---
        if signal.startswith("买入"):
            if position == 0:
                position = 1
                buy_price = current_price
                buy_date = current_date
                signals.append(('Buy', current_date, buy_price, buy_date))
        
        elif signal.startswith("卖出"):
            if position == 1:
                position = 0
                sell_price = current_price
                
                # 计算净收益率并扣除双边交易成本
                gross_return = (sell_price - buy_price) / buy_price * 100
                trade_return = gross_return - round_trip_cost_percent
                
                returns.append(trade_return)
                signals.append(('Sell (Signal)', current_date, sell_price, buy_date))
                
                buy_date = None # 清空买入日期

    # 如果回测结束时仍有持仓，则以最后一日价格结算
    if position == 1 and len(df) > 0:
        sell_price = df.iloc[-1]['net_value']
        current_date = df.iloc[-1]['date']
        
        # 结算最后一次交易，同样扣除双边交易成本
        gross_return = (sell_price - buy_price) / buy_price * 100
        trade_return = gross_return - round_trip_cost_percent
        
        returns.append(trade_return)
        signals.append(('Sell (Final)', current_date, sell_price, buy_date))

    # 计算胜率和平均收益率
    num_trades = len(returns)
    win_rate = len([r for r in returns if r > 0]) / num_trades if num_trades > 0 else 0
    avg_return = np.mean(returns) if returns else 0
    total_return = sum(returns) if returns else 0

    return win_rate, avg_return, total_return, signals

def generate_signal_and_score(df):
    """
    从已计算好的指标中提取最新信号、评分和相对位置 (V2.6 新增 MDD)
    """
    if df.empty or 'score' not in df.columns or df.iloc[-1].isnull().all():
        # V2.6 新增 latest_mdd, current_drawdown 默认值
        return 0, "数据不足/指标未计算", 0.5, 0.0, 0.0, 0.0, 0.0, 0.0

    latest = df.iloc[-1]
    
    score = latest['score']
    signal_zh = latest['signal']
    
    # 确保 relative_position_dynamic 存在且非 NaN
    relative_position = latest.get('relative_position_dynamic', 0.5) if not pd.isna(latest.get('relative_position_dynamic')) else 0.5
    
    # V2.6 新增 MDD 相关指标
    latest_mdd = latest.get('latest_mdd', 0.0) if not pd.isna(latest.get('latest_mdd')) else 0.0
    current_drawdown = latest.get('current_drawdown', 0.0) if not pd.isna(latest.get('current_drawdown')) else 0.0


    # 添加英文翻译，用于报告输出
    if signal_zh == "强烈买入": signal_full = "强烈买入 (Strong Buy) / 强烈加仓"
    elif signal_zh == "买入": signal_full = "买入 (Buy) / 加仓"
    elif signal_zh == "强烈卖出": signal_full = "强烈卖出 (Strong Sell) / 强烈减仓"
    elif signal_zh == "卖出": signal_full = "卖出 (Sell) / 减仓"
    else: signal_full = "观望 (Hold) / 保持仓位"


    return score, signal_full, relative_position, latest['SMA_5'], latest['SMA_20'], latest['SMA_60'], latest_mdd, current_drawdown


def analyze_single_fund(file_path, risk_free_rate_daily_percent, transaction_cost):
    """
    (V2.6 优化) 用于并行处理的单个基金分析核心逻辑，新增 MDD 返回。
    """
    fund_name = re.sub(r'\.csv$', '', os.path.basename(file_path))
    
    # 【V2.4 新增参数】
    STOP_LOSS = 10.0
    TAKE_PROFIT = 20.0
    
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        
        # --- 严格输入校验 ---
        required_columns = ['date', 'net_value']
        if not all(col in df.columns for col in required_columns):
            print(f"跳过 {fund_name}: 缺少必要列 {required_columns}")
            return None
        if df['date'].isnull().any():
            print(f"跳过 {fund_name}: 日期列包含空值")
            return None
            
        if len(df) < 60:
            print(f"跳过 {fund_name}: 数据不足60天。")
            return None
        
        # 1. 计算所有技术指标 (包含夏普比率年化和MDD)
        df_indicators = calculate_indicators(
            df, 
            risk_free_rate_daily=risk_free_rate_daily_percent, 
            annualize_sharpe=True
        )
        
        # 2. 逐日生成信号和评分 (V2.6 抄底强化 - 权重调整)
        df_signals = generate_all_signals(df_indicators)

        # 3. 提取最新的信号和数据用于报告
        latest_data = df_signals.iloc[-1]
        # V2.6 新增 MDD 相关指标
        score, signal, rel_pos, sma5, sma20, sma60, latest_mdd, current_drawdown = generate_signal_and_score(df_signals)
        
        start_date = df_indicators['date'].min().strftime('%Y-%m-%d')
        end_date = df_indicators['date'].max().strftime('%Y-%m-%d')
        
        initial_value = df_indicators.iloc[0]['net_value']
        latest_value = latest_data['net_value']
        cumulative_return = ((latest_value / initial_value) - 1) * 100
        
        # 4. 回测策略 (使用指定的交易成本、止损止盈)
        win_rate, avg_return, total_return, _ = backtest_strategy(
            df_signals, 
            transaction_cost=transaction_cost,
            stop_loss_percent=STOP_LOSS,     # V2.4 引入
            take_profit_percent=TAKE_PROFIT  # V2.4 引入
        )

        return {
            'fund_name': fund_name,
            'score': score,
            'signal': signal,
            'latest_net_value': latest_value,
            'latest_date': end_date,
            'time_span': f"{start_date} 至 {end_date}",
            'cumulative_return': cumulative_return,
            'latest_daily_return': latest_data.get('daily_return', np.nan),
            'relative_position': rel_pos,
            'latest_mdd': latest_mdd,           # V2.6 新增
            'current_drawdown': current_drawdown, # V2.6 新增
            'sma5': sma5,
            'sma20': sma20,
            'sma60': sma60,
            'volatility': latest_data.get('Volatility', np.nan),
            'sharpe_ratio': latest_data.get('Sharpe_Ratio', np.nan),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'stop_loss': STOP_LOSS,
            'take_profit': TAKE_PROFIT
        }

    except Exception as e:
        # 为了调试，暂时只打印错误，不返回 None
        print(f"处理文件 {fund_name} 时发生错误: {e}")
        return None


def main_analysis(fund_data_dir='fund_data/', risk_free_rate_annual=0.03, transaction_cost=0.001):
    """
    (V2.6 抄底强化) 主分析函数：并行处理，**按评分+抄底指标筛选优先排序**。
    """
    
    fund_files = glob.glob(os.path.join(fund_data_dir, '*.csv'))
    
    if not fund_files:
        print(f"错误：未在 '{fund_data_dir}' 目录下找到任何 CSV 文件。请检查您的仓库结构。")
        return

    print(f"找到 {len(fund_files)} 个基金文件，开始并行分析...")

    analysis_results = []
    
    # 将年化无风险利率转换为日化百分比
    risk_free_rate_daily_percent = (risk_free_rate_annual / 252) * 100
    
    # 使用 ProcessPoolExecutor 进行并行处理
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        
        # 提交所有文件的分析任务
        futures = {
            executor.submit(
                analyze_single_fund,
                file_path,
                risk_free_rate_daily_percent,
                transaction_cost
            ): file_path
            for file_path in fund_files
        }
        
        # 收集结果
        for future in as_completed(futures):
            result = future.result()
            if result:
                analysis_results.append(result)

    # --- 后续处理 (排序和报告生成) ---
    if not analysis_results:
        print("所有文件分析失败或数据不足，未生成报告。")
        return
        
    # 【V2.6 核心优化】：自定义排序逻辑：
    # 1. 筛选出【评分在 +4 以上】的基金 (即买入/强烈买入)
    buy_signals = [x for x in analysis_results if x['score'] >= 4]
    
    # 2. 对这些基金进行【抄底优先排序】：
    #    主要按 'relative_position'（相对历史位置）从低到高排序 (越低越好)
    #    其次按 'current_drawdown'（最新回撤深度）从低到高排序 (越深越好，因为是负值)
    #    再其次按 'score'（评分）从高到低排序
    buy_signals.sort(key=lambda x: (x['relative_position'], x['current_drawdown'], -x['score'])) 
    
    # 3. 将剩余的基金按原 V2.5 逻辑排序 (评分优先)
    other_signals = [x for x in analysis_results if x['score'] < 4]
    other_signals.sort(key=lambda x: (-x['score'], x['relative_position']))
    
    # 4. 合并最终的排序列表
    final_sorted_results = buy_signals + other_signals

    # --- 输出到 Markdown 文件 ---
    
    current_date = datetime.now()
    output_dir = current_date.strftime('%Y%m')
    output_filename = current_date.strftime('%Y%m%d%H%M%S.md')
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 报告生成逻辑... 
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# 基金投资分析报告 - {current_date.strftime('%Y年%m月%d日 %H:%M:%S')} (**V2.6 - 深度抄底优先**)\n\n")
        f.write(f"**分析时间:** {current_date.strftime('%Y-%m-%d %H:%M:%S')} (UTC+0)\n")
        f.write(f"**总计分析基金数:** {len(analysis_results)} 个\n")
        f.write(f"**版本优化:** **权重偏向低位抄底** (相对位置 $\\mathbf{+6}$，MDD $\\mathbf{+5}$，趋势权重降级)。\n")
        f.write(f"**指标依据:** SMA(5/20/60), MACD(12/26/9), RSI(14 - 阈值 25/75), 波动率(20日), 年化夏普比率, **MDD**。\n")
        
        if analysis_results:
             sl = analysis_results[0]['stop_loss']
             tp = analysis_results[0]['take_profit']
             f.write(f"**回测策略:** 基于评分模型的全历史回测 (已扣除双边交易成本 {transaction_cost*2*100:.2f}%)。\n")
             f.write(f"**回测风控:** **硬性止损 $\\mathbf{{%.2f}}$%%，硬性止盈 $\\mathbf{{%.2f}}$%%。**\n\n" % (sl, tp))
        
        # 【V2.6 报告标题优化】：反映抄底优先排序
        f.write("## 基金综合排序及建议 (【评分 $\\mathbf{\\ge 4}$ 且低位】基金优先排序)\n\n")
        
        # V2.6 报告列新增 MDD
        f.write("| 排名 | 基金名称 | **评分** | **投资建议** | 相对历史位置 | MDD深度(%) | 最新回撤(%) | 最新净值 | 最新日涨幅(%) | 年化夏普比率 | **回测平均净收益率(%)** |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        for i, result in enumerate(final_sorted_results):
            rel_pos_percent = f"{result['relative_position'] * 100:.2f}%"
            sharpe_ratio = f"{result['sharpe_ratio']:.2f}" if not np.isnan(result['sharpe_ratio']) else "N/A"
            avg_return = f"{result['avg_return']:.2f}" if result['avg_return'] != 0 else "0.00"
            mdd = f"{result['latest_mdd']:.2f}" if not np.isnan(result['latest_mdd']) else "N/A"
            cd = f"{result['current_drawdown']:.2f}" if not np.isnan(result['current_drawdown']) else "N/A"
            
            # 【V2.6 报告列顺序优化】：评分和抄底指标提前，去掉波动率和胜率
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
                f"| {sharpe_ratio} "
                f"| {avg_return} |\n"
            )

        f.write("\n## 附录：指标详情与说明\n\n")
        f.write("### 投资建议说明\n")
        f.write("- **【重要】列表已优先展示** $\\mathbf{\\text{评分} \ge 4}$ **的基金，并按** $\\mathbf{\\text{相对历史位置} \to \text{最新回撤深度} \to \text{评分}}$ **的顺序排序。**\n")
        f.write("- **相对历史位置**: 权重 $\\mathbf{+6}$。最新净值在历史最高和最低之间的相对位置 (0%为历史最低，100%为历史最高)。\n")
        f.write("- **MDD 深度 / 最新回撤**: 权重 $\\mathbf{+5}$。最大回撤/当前回撤越深（负值越大），在评分中加分越多，反映抄底机会。\n")
        f.write("- **年化夏普比率**: 权重 $\\mathbf{+4}$。反映**风险调整后的年化收益**。无风险利率假设为 $\\mathbf{%.2f}$%%。\n" % (risk_free_rate_annual*100))
        f.write("- **回测平均净收益率**: 模型基于历史数据的实际交易效果，已扣除 $\\mathbf{%.2f}$%% 的往返交易成本。**包含了止损/止盈对结果的优化**。\n" % (transaction_cost*2*100))
        f.write("\n### 数据范围\n")
        for result in final_sorted_results:
            f.write(f"- `{result['fund_name']}`: 数据日期 {result['latest_date']}，范围 {result['time_span']} (累计涨幅: {result['cumulative_return']:.2f}%) (SMA: {result['sma5']:.4f}/{result['sma20']:.4f}/{result['sma60']:.4f})\n")

        
    print(f"\n分析完成! 报告已输出到: {output_path}")

# --- 新增的执行入口 ---
if __name__ == "__main__":
    main_analysis(risk_free_rate_annual=0.03, transaction_cost=0.001)