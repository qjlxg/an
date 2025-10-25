import pandas as pd
import glob
import os
import numpy as np

# --- 配置参数 (双重筛选条件) ---
FUND_DATA_DIR = 'fund_data'
MIN_CONSECUTIVE_DROP_DAYS = 3 # 连续下跌天数的阈值 (用于30日)
MIN_MONTH_DRAWDOWN = 0.06      # 1个月回撤的阈值 (6%)
# 高弹性筛选的最低回撤阈值 (例如 10%)
HIGH_ELASTICITY_MIN_DRAWDOWN = 0.10 
# 【新增】当日跌幅的最低阈值 (例如 3%)
MIN_DAILY_DROP_PERCENT = 0.03 
REPORT_BASE_NAME = 'fund_warning_report' 

# --- 新增函数：计算技术指标 ---
def calculate_technical_indicators(df):
    """
    计算基金净值的RSI(14)、MACD、MA50，并判断布林带位置。
    要求df必须按日期降序排列。
    """
    if 'value' not in df.columns or len(df) < 50:
        # MACD和MA50至少需要较长数据
        return {
            'RSI': np.nan, 'MACD信号': '数据不足', '净值/MA50': np.nan, 
            '布林带位置': '数据不足', '最新净值': df['value'].iloc[0] if not df.empty else np.nan,
            '当日跌幅': np.nan # 新增当日跌幅
        }
    
    # 确保我们使用按时间升序的副本进行计算
    df_asc = df.iloc[::-1].copy()
    
    # 1. RSI (14) - 计算不变
    delta = df_asc['value'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_asc['RSI'] = 100 - (100 / (1 + rs))
    rsi_latest = df_asc['RSI'].iloc[-1]
    
    # 2. MACD, MA50, 布林带 - 计算不变
    ema_12 = df_asc['value'].ewm(span=12, adjust=False).mean()
    ema_26 = df_asc['value'].ewm(span=26, adjust=False).mean()
    df_asc['MACD'] = ema_12 - ema_26
    df_asc['Signal'] = df_asc['MACD'].ewm(span=9, adjust=False).mean()
    macd_latest = df_asc['MACD'].iloc[-1]
    signal_latest = df_asc['Signal'].iloc[-1]
    macd_prev = df_asc['MACD'].iloc[-2]
    signal_prev = df_asc['Signal'].iloc[-2]

    if macd_latest > signal_latest and macd_prev < signal_prev:
        macd_signal = '金叉'
    elif macd_latest < signal_latest and macd_prev > signal_prev:
        macd_signal = '死叉'
    else:
        macd_signal = '观察'

    df_asc['MA50'] = df_asc['value'].rolling(window=50).mean()
    ma50_latest = df_asc['MA50'].iloc[-1]
    value_latest = df_asc['value'].iloc[-1]
    net_to_ma50 = value_latest / ma50_latest if ma50_latest else np.nan

    df_asc['MA20'] = df_asc['value'].rolling(window=20).mean()
    df_asc['StdDev'] = df_asc['value'].rolling(window=20).std()
    upper_latest = df_asc['MA20'].iloc[-1] + (df_asc['StdDev'].iloc[-1] * 2)
    lower_latest = df_asc['MA20'].iloc[-1] - (df_asc['StdDev'].iloc[-1] * 2)

    if value_latest > upper_latest:
        bollinger_pos = '上轨上方'
    elif value_latest < lower_latest:
        bollinger_pos = '下轨下方'
    elif value_latest > df_asc['MA20'].iloc[-1]:
        bollinger_pos = '中轨上方'
    else:
        bollinger_pos = '中轨下方/中轨'
        
    # 5. 【新增】计算当日跌幅 (T日 vs T-1日)
    daily_drop = 0.0
    if len(df_asc) >= 2:
        value_t_minus_1 = df_asc['value'].iloc[-2]
        if value_t_minus_1 > 0:
            daily_drop = (value_t_minus_1 - value_latest) / value_t_minus_1

    return {
        'RSI': round(rsi_latest, 2), 
        'MACD信号': macd_signal, 
        '净值/MA50': round(net_to_ma50, 2), 
        '布林带位置': bollinger_pos,
        '最新净值': round(value_latest, 4),
        '当日跌幅': round(daily_drop, 4) # 新增
    }

# --- 其他不变的辅助函数 (extract_fund_codes, calculate_consecutive_drops, calculate_max_drawdown) ---
# ... (保持不变)

def extract_fund_codes(report_content):
    codes = set()
    lines = report_content.split('\n')
    in_table = False
    for line in lines:
        if line.strip().startswith('|') and '---' in line and ':' in line: 
            in_table = True
            continue
        if in_table and line.strip() and line.count('|') >= 8: 
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 11: 
                fund_code = parts[2]
                action_signal = parts[10]
                if action_signal.startswith('买入信号'): # 仅筛选买入信号（RSI < 35且当日大跌）
                    try:
                        if fund_code.isdigit():
                            codes.add(fund_code)
                    except ValueError:
                        continue 
    return list(codes)

def calculate_consecutive_drops(series):
    if series.empty or len(series) < 2:
        return 0
    drops = (series < series.shift(1)).iloc[1:] 
    drops_int = drops.astype(int)
    max_drop_days = 0
    current_drop_days = 0
    for val in drops_int:
        if val == 1:
            current_drop_days += 1
        else:
            max_drop_days = max(max_drop_days, current_drop_days)
            current_drop_days = 0
    max_drop_days = max(max_drop_days, current_drop_days)
    return max_drop_days

def calculate_max_drawdown(series):
    if series.empty:
        return 0.0
    rolling_max = series.cummax()
    drawdown = (rolling_max - series) / rolling_max
    mdd = drawdown.max()
    return mdd

# --- 修正后的生成报告函数（新增极度超卖精选列表并置于首位） ---
def generate_report(results, timestamp_str):
    now_str = timestamp_str

    if not results:
        return (
            f"# 基金预警报告 ({now_str} UTC+8)\n\n"
            f"## 分析总结\n\n"
            f"**恭喜，在过去一个月内，没有发现同时满足 '连续下跌{MIN_CONSECUTIVE_DROP_DAYS}天以上' 和 '1个月回撤{MIN_MONTH_DRAWDOWN*100:.0f}%以上' 的基金。**\n\n"
            f"---\n"
            f"分析数据时间范围: 最近30个交易日 (通常约为1个月)。"
        )

    # 1. 主列表处理 (所有预警基金)
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='最大回撤', ascending=False).reset_index(drop=True)
    df_results.index = df_results.index + 1 
    
    total_count = len(df_results)
    
    report = f"# 基金预警报告 ({now_str} UTC+8)\n\n"
    
    # --- 增加总结部分 ---
    report += f"## 分析总结\n\n"
    report += f"本次分析共发现 **{total_count}** 只基金同时满足以下两个预警条件（基于最近30个交易日）：\n"
    report += f"1. **连续下跌**：净值连续下跌 **{MIN_CONSECUTIVE_DROP_DAYS}** 天以上。\n"
    report += f"2. **高回撤**：近 1 个月内最大回撤达到 **{MIN_MONTH_DRAWDOWN*100:.0f}%** 以上。\n\n"
    report += f"**新增分析维度：近一周（5日）连跌天数、当日跌幅、关键技术指标（RSI, MACD等）和基于RSI的行动提示。**\n"
    report += f"---"
    
    # 2. 【新增】极度超卖精选列表筛选 (最严格的条件)
    # 条件：最大回撤 >= 10% 且 近一周连跌天数 == 1 且 RSI < 35 且 当日跌幅 >= 3%
    df_elastic = df_results[
        (df_results['最大回撤'] >= HIGH_ELASTICITY_MIN_DRAWDOWN) & 
        (df_results['近一周连跌'] == 1)
    ].copy() 

    # 筛选出满足 RSI < 35 AND 当日跌幅 >= 3% 的技术共振基金
    df_strict_elastic = df_elastic[
        ((df_elastic['RSI'] < 35)) &
        (df_elastic['当日跌幅'] >= MIN_DAILY_DROP_PERCENT)
    ].copy()
    
    # 3. 生成【技术共振】极度超卖精选列表报告部分 (置于最前)
    if not df_strict_elastic.empty:
        df_strict_elastic = df_strict_elastic.sort_values(by=['当日跌幅', 'RSI'], ascending=[False, True]).reset_index(drop=True)
        df_strict_elastic.index = df_strict_elastic.index + 1
        
        strict_count = len(df_strict_elastic)
        
        report += f"\n## **🥇【最高优先级】当日恐慌买入信号** ({strict_count}只)\n\n"
        
        report += f"此列表是所有预警中，**最具操作性的极度超跌标的**。需同时满足：\n"
        report += f"1. 长期超跌 ($\ge$ {HIGH_ELASTICITY_MIN_DRAWDOWN*100:.0f}%) + 低位企稳 (连跌1天)\n"
        report += f"2. 中期超卖 (RSI < 35)\n"
        report += f"3. **当日恐慌性大跌 ($\ge$ {MIN_DAILY_DROP_PERCENT*100:.0f}%)**\n\n"
        
        report += f"| 排名 | 基金代码 | 最大回撤 (1M) | **当日跌幅** | 连跌 (1M) | RSI(14) | MACD信号 | 净值/MA50 | 试水买价 (跌3%) | 行动提示 |\n"
        report += f"| :---: | :---: | ---: | ---: | ---: | ---: | :---: | ---: | :---: | :---: |\n"  

        for index, row in df_strict_elastic.iterrows():
            latest_value = row.get('最新净值', 1.0)
            trial_price = latest_value * 0.97
            
            # 强化行动提示：满足当日大跌条件后，RSI < 35 的提示改为更积极的“买入信号”
            action_prompt = '买入信号 (RSI超卖 + 当日大跌)'
            if row['RSI'] < 30:
                action_prompt = '买入信号 (RSI极度超卖 + 当日大跌)'

            report += f"| {index} | `{row['基金代码']}` | **{row['最大回撤']:.2%}** | **{row['当日跌幅']:.2%}** | {row['最大连续下跌']} | {row['RSI']:.2f} | {row['MACD信号']} | {row['净值/MA50']:.2f} | {trial_price:.4f} | **{action_prompt}** |\n"
        
        report += "\n---\n"
    else:
        report += f"\n## **🥇【最高优先级】当日恐慌买入信号**\n\n"
        report += f"**恭喜，没有发现同时满足所有严格筛选条件（高回撤、低位企稳、RSI < 35 和 当日跌幅 $\ge$ {MIN_DAILY_DROP_PERCENT*100:.0f}%）的基金。** 市场恐慌度不足。\n\n"
        report += "\n---\n"

    # 4. 生成【扩展列表】高弹性精选列表报告部分 (原列表，作为观察池)
    # 此列表现在是所有满足 10%回撤 + 连跌1天 的基金 (RSI < 35 或 当日大跌 的已在上一个列表出现)
    if not df_elastic.empty:
        # 从 df_elastic 中排除 df_strict_elastic 中已有的基金，以避免重复
        funds_to_exclude = df_strict_elastic['基金代码'].tolist()
        df_extended_elastic = df_elastic[~df_elastic['基金代码'].isin(funds_to_exclude)].copy()

        if not df_extended_elastic.empty:
            df_extended_elastic = df_extended_elastic.sort_values(by='最大回撤', ascending=False).reset_index(drop=True)
            df_extended_elastic.index = df_extended_elastic.index + 1
            
            extended_count = len(df_extended_elastic)
            
            report += f"\n## **🥈【扩展列表】高弹性观察池** ({extended_count}只)\n\n"
            
            report += f"此列表包含：**最大回撤 $\ge$ {HIGH_ELASTICITY_MIN_DRAWDOWN*100:.0f}%** 且 **近一周连跌天数 = 1** 的所有基金（排除了已进入最高优先级列表的）。\n\n"
            
            # 表格新增 当日跌幅
            report += f"| 排名 | 基金代码 | 最大回撤 (1M) | **当日跌幅** | 连跌 (1M) | RSI(14) | MACD信号 | 净值/MA50 | 试水买价 (跌3%) | 行动提示 |\n"
            report += f"| :---: | :---: | ---: | ---: | ---: | ---: | :---: | ---: | :---: | :---: |\n"  

            for index, row in df_extended_elastic.iterrows():
                latest_value = row.get('最新净值', 1.0)
                trial_price = latest_value * 0.97
                
                report += f"| {index} | `{row['基金代码']}` | **{row['最大回撤']:.2%}** | {row['当日跌幅']:.2%} | {row['最大连续下跌']} | {row['RSI']:.2f} | {row['MACD信号']} | {row['净值/MA50']:.2f} | {trial_price:.4f} | {row['行动提示']} |\n"
            
            report += "\n---\n"
        else:
            report += f"\n## **🥈【扩展列表】高弹性观察池**\n\n"
            report += f"所有满足 **最大回撤 $\ge$ {HIGH_ELASTICITY_MIN_DRAWDOWN*100:.0f}%** 且 **近一周连跌天数 = 1** 的基金，均已进入 **最高优先级列表**。\n\n"
            report += "\n---\n"

    # 5. 原有预警基金列表 (所有符合条件的基金)
    report += f"\n## 所有预警基金列表 (共 {total_count} 只，按最大回撤降序排列)\n\n"
    
    # 表格新增 当日跌幅
    report += f"| 排名 | 基金代码 | 最大回撤 (1M) | **当日跌幅** | 连跌 (1M) | 连跌 (1W) | RSI(14) | MACD信号 | 净值/MA50 | 布林带位置 |\n"
    report += f"| :---: | :---: | ---: | ---: | ---: | ---: | :---: | ---: | :---: | :---: |\n"  

    for index, row in df_results.iterrows():
        report += f"| {index} | `{row['基金代码']}` | **{row['最大回撤']:.2%}** | {row['当日跌幅']:.2%} | {row['最大连续下跌']} | {row['近一周连跌']} | {row['RSI']:.2f} | {row['MACD信号']} | {row['净值/MA50']:.2f} | {row['布林带位置']} |\n"
    
    report += "\n---\n"
    report += f"分析数据时间范围: 最近30个交易日 (通常约为1个月)。\n"
    
    # 6. 新增行动策略总结 (已修复所有转义错误)
    report += f"\n## **高弹性策略执行纪律**\n\n"
    report += f"**1. 建仓与最大加仓（逆向原则）：**\n"
    report += f"    * **最高优先级：** 仅当基金出现在 **🥇【最高优先级】当日恐慌买入信号** 列表中时，才应考虑立即建仓。\n"
    report += f"    * **试水建仓:** 当'行动提示'为 **'买入信号'** 时，立即投入 **小额资金（例如 1/5 观察仓位）**进行建仓。\n"
    report += f"    * **最大加仓:** 当基金在试水后，累计跌幅达到您的金字塔原则 **(例如从试水价下跌 5%)** 且 **RSI < 20** 时，执行**最大额加仓**（如 **1000** 元），实现快速降低成本。\n"
    report += f"**2. 波段止盈与清仓信号（顺势原则）：**\n"
    report += f"    * **确认反弹/止盈警惕:** 当目标基金的 **MACD 信号从 '观察/死叉' 变为 '金叉'** 时，表明反弹趋势确立，此时应视为 **分批止盈** 的警惕信号，而不是加仓。应在 **+5%** 止盈线出现时，果断赎回 **50%** 份额。\n"
    report += f"    * **趋势反转/清仓:** 当 **MACD 信号从 '金叉' 变为 '死叉'** 或 **净值跌破 MA50 (净值/MA50 < 1.0)** 且您的**平均成本已实现 5% 利润**时，应考虑**清仓止盈**。\n"
    report += f"**3. 风险控制（严格止损）：**\n"
    report += f"    * 为所有买入的基金设置严格的止损线。建议从买入平均成本价开始计算，一旦跌幅达到 **8%-10%**，应**立即**卖出清仓，避免深度套牢。\n"
    
    return report


# --- 原有函数：在分析时计算技术指标和行动提示 (RSI提示已修正) ---
def analyze_all_funds(target_codes=None): 
    """
    遍历基金数据目录，分析每个基金，并返回符合条件的基金列表。
    """
    if target_codes:
        csv_files = [os.path.join(FUND_DATA_DIR, f'{code}.csv') for code in target_codes]
        csv_files = [f for f in csv_files if os.path.exists(f)]
        
        if not csv_files:
            print(f"警告：在目录 '{FUND_DATA_DIR}' 中未找到目标基金对应的 CSV 文件。")
            return []
    else:
        csv_files = glob.glob(os.path.join(FUND_DATA_DIR, '*.csv'))
        if not csv_files:
            print(f"警告：在目录 '{FUND_DATA_DIR}' 中未找到任何 CSV 文件。")
            return []


    print(f"找到 {len(csv_files)} 个基金数据文件，开始分析...")
    
    qualifying_funds = []
    
    for filepath in csv_files:
        try:
            fund_code = os.path.splitext(os.path.basename(filepath))[0]
            
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            # 确保主df按日期降序排列
            df = df.sort_values(by='date', ascending=False).reset_index(drop=True) 
            df = df.rename(columns={'net_value': 'value'})
            
            if len(df) < 30:
                continue
            
            df_recent_month = df.head(30)
            df_recent_week = df.head(5)
            
            # 1. 连续下跌和回撤指标
            max_drop_days_month = calculate_consecutive_drops(df_recent_month['value'])
            mdd_recent_month = calculate_max_drawdown(df_recent_month['value'])
            max_drop_days_week = calculate_consecutive_drops(df_recent_week['value'])

            # 2. 技术指标 (使用完整的df进行计算)
            tech_indicators = calculate_technical_indicators(df)
            rsi_val = tech_indicators.get('RSI', np.nan)
            daily_drop_val = tech_indicators.get('当日跌幅', 0.0)

            # 3. 行动提示逻辑 (针对高弹性精选标准)
            action_prompt = '不适用 (非高弹性精选)'
            
            # 只有满足 10%回撤 和 连跌1天 基础条件时，才触发行动提示逻辑
            if mdd_recent_month >= HIGH_ELASTICITY_MIN_DRAWDOWN and max_drop_days_week == 1:
                
                if not np.isnan(rsi_val):
                    # 【最高优先级】 RSI极度超卖 + 当日大跌
                    if rsi_val < 30 and daily_drop_val >= MIN_DAILY_DROP_PERCENT:
                        action_prompt = '买入信号 (RSI极度超卖 + 当日大跌)'
                    
                    # 【次高优先级】 RSI超卖 + 当日大跌
                    elif rsi_val < 35 and daily_drop_val >= MIN_DAILY_DROP_PERCENT:
                        action_prompt = '买入信号 (RSI超卖 + 当日大跌)'
                        
                    # 【次级观察】 满足所有条件，但当日未大跌
                    elif rsi_val < 35:
                         action_prompt = '考虑试水建仓 (RSI超卖)'
                    
                    # 仅满足回撤和连跌1天，RSI未超卖
                    else: 
                        action_prompt = '高回撤观察 (RSI未超卖)'


            if max_drop_days_month >= MIN_CONSECUTIVE_DROP_DAYS and mdd_recent_month >= MIN_MONTH_DRAWDOWN:
                fund_data = {
                    '基金代码': fund_code,
                    '最大回撤': mdd_recent_month,  
                    '最大连续下跌': max_drop_days_month,
                    '近一周连跌': max_drop_days_week,
                    # --- 整合技术指标 ---
                    'RSI': tech_indicators['RSI'],
                    'MACD信号': tech_indicators['MACD信号'],
                    '净值/MA50': tech_indicators['净值/MA50'],
                    '布林带位置': tech_indicators['布林带位置'],
                    '最新净值': tech_indicators['最新净值'],
                    '当日跌幅': daily_drop_val, # 新增
                    # --- 整合行动提示 ---
                    '行动提示': action_prompt
                }
                qualifying_funds.append(fund_data)

        except Exception as e:
            print(f"处理文件 {filepath} 时发生错误: {e}")
            continue

    return qualifying_funds


if __name__ == '__main__':
    
    # 0. 获取当前时间戳和目录名
    try:
        # 使用 Asia/Shanghai 时区（UTC+8）
        now = pd.Timestamp.now(tz='Asia/Shanghai') 
        timestamp_for_report = now.strftime('%Y-%m-d %H:%M:%S')
        timestamp_for_filename = now.strftime('%Y%m%d_%H%M%S')
        DIR_NAME = now.strftime('%Y%m') 
    except Exception:
        # 异常处理，使用默认时间戳
        timestamp_for_report = pd.Timestamp.now().strftime('%Y-%m-d %H:%M:%S')
        timestamp_for_filename = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        DIR_NAME = pd.Timestamp.now().strftime('%Y%m')
        
    # 1. 创建目标目录
    os.makedirs(DIR_NAME, exist_ok=True)
        
    # 2. 生成带目录和时间戳的文件名
    REPORT_FILE = os.path.join(DIR_NAME, f"{REPORT_BASE_NAME}_{timestamp_for_filename}.md")

    # 3. 读取并解析 market_monitor_report.md 文件
    try:
        with open('market_monitor_report.md', 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        # 提取目标基金代码 (只提取具有明确买入信号的基金)
        target_funds = extract_fund_codes(report_content)
        
        print(f"已从报告中提取 {len(target_funds)} 个 '买入信号' 的基金代码。")
        
    except FileNotFoundError:
        print("警告：未找到 market_monitor_report.md 文件，将分析 FUND_DATA_DIR 目录下的所有文件。")
        target_funds = None

    # 4. 执行分析，只针对目标基金
    results = analyze_all_funds(target_codes=target_funds)
    
    # 5. 生成 Markdown 报告
    report_content = generate_report(results, timestamp_for_report)
    
    # 6. 写入报告文件
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"分析完成，报告已保存到 {REPORT_FILE}")
