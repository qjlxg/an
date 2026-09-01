import pandas as pd
import glob
import os
import numpy as np
from datetime import datetime
import pytz
import logging
import math

# --- 配置参数 (完整保留) ---# 只有近一个月最大回撤达到 $15\%$ 或更高的基金，才会进入报告中的高优先级超卖判断。
FUND_DATA_DIR = 'fund_data'
MIN_CONSECUTIVE_DROP_DAYS = 3#表示最小连续下跌天数为 3 天
MIN_MONTH_DRAWDOWN = 0.10  #(即 10%) 是基础回撤条件，所有基金只有达到这个回撤才会被纳入报告。
HIGH_ELASTICITY_MIN_DRAWDOWN = 0.15  # (即 15%) 是用于定义 "高弹性" 基金的更严格回撤条件，只有达到这个条件的基金才会进入报告中的第一和第二优先级 (P1, P2) 筛选，并触发行动提示中的超卖判断。
MIN_DAILY_DROP_PERCENT = 0.03  # 当日大跌的定义 (3%)
REPORT_BASE_NAME = 'fund_warning_report'

# --- 核心阈值调整 (完整保留) --
EXTREME_RSI_THRESHOLD_P1 = 29.0 
STRONG_RSI_THRESHOLD_P2 = 35.0

# --- 设置日志 (函数配置 1/13) ---
def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fund_analysis.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

# --- 验证数据 (函数配置 2/13) ---
def validate_fund_data(df, fund_code):
    """验证基金数据的完整性和质量"""
    if df.empty: return False, "数据为空"
    if 'value' not in df.columns: return False, "缺少净值列"
    # 【已保留】最小数据要求为 60
    if len(df) < 60: return False, f"数据不足60条，当前只有{len(df)}条"
    if (df['value'] <= 0).any(): return False, "存在无效净值(<=0)"
    return True, "数据有效"

# --- 布林带计算 (函数配置 3/13) ---
def calculate_bollinger_bands(series, window=20):
    """计算布林带位置"""
    if len(series) < window:
        return "数据不足"
    
    df_temp = pd.DataFrame({'value': series.values})
    df_temp['MA20'] = df_temp['value'].rolling(window=window).mean()
    df_temp['STD20'] = df_temp['value'].rolling(window=window).std()
    
    # 确保没有除以零
    if df_temp['STD20'].iloc[-1] == 0:
        return "波动极小"
        
    df_temp['Upper Band'] = df_temp['MA20'] + (df_temp['STD20'] * 2)
    df_temp['Lower Band'] = df_temp['MA20'] - (df_temp['STD20'] * 2)
    
    latest_value = df_temp['value'].iloc[-1]
    latest_lower = df_temp['Lower Band'].iloc[-1]
    latest_upper = df_temp['Upper Band'].iloc[-1]
    
    if pd.isna(latest_lower) or pd.isna(latest_upper):
        return "数据不足"
        
    if latest_value <= latest_lower:
        return "**下轨下方**" 
    elif latest_value >= latest_upper:
        return "**上轨上方**" 
    else:
        # 归一化位置
        range_band = latest_upper - latest_lower
        if range_band == 0:
             return "轨道中间" 
             
        position = (latest_value - latest_lower) / range_band
        if position < 0.2:
            return "下轨附近"
        elif position > 0.8:
            return "上轨附近"
        else:
            return "轨道中间"

# --- 技术指标计算 (函数配置 4/13) ---
def calculate_technical_indicators(df):
    """
    计算基金净值的完整技术指标 (RSI(14), RSI(6), MACD, MA, 趋势等)
    假设 df 是按时间升序排列的（最新值在末尾）。
    """
    df_asc = df.copy()

    try:
        # 这里的判断也从 250 降低到 60，以兼容 MA50 和 RSI
        if 'value' not in df_asc.columns or len(df_asc) < 60:
            return {
                'RSI(14)': np.nan, 
                'RSI(6)': np.nan, # 新增RSI(6)
                'MACD信号': '数据不足', 
                '净值/MA50': np.nan,
                '净值/MA250': np.nan, 
                'MA50/MA250': np.nan, 
                'MA50/MA250趋势': '数据不足',
                '布林带位置': '数据不足', 
                '最新净值': df_asc['value'].iloc[-1] if not df_asc.empty else np.nan,
                '当日跌幅': np.nan
            }

        delta = df_asc['value'].diff()

        # 1. RSI (14) - 原有逻辑
        gain_14 = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss_14 = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs_14 = gain_14 / loss_14.replace(0, np.nan) 
        df_asc['RSI_14'] = 100 - (100 / (1 + rs_14))
        rsi_14_latest = df_asc['RSI_14'].iloc[-1]
        
        # 1.b RSI (6) - 新增逻辑
        gain_6 = (delta.where(delta > 0, 0)).rolling(window=6, min_periods=1).mean()
        loss_6 = (-delta.where(delta < 0, 0)).rolling(window=6, min_periods=1).mean()
        rs_6 = gain_6 / loss_6.replace(0, np.nan) 
        df_asc['RSI_6'] = 100 - (100 / (1 + rs_6))
        rsi_6_latest = df_asc['RSI_6'].iloc[-1]
        
        # 2. MACD (简化为信号判断)
        ema_12 = df_asc['value'].ewm(span=12, adjust=False).mean()
        ema_26 = df_asc['value'].ewm(span=26, adjust=False).mean()
        df_asc['MACD'] = ema_12 - ema_26
        df_asc['Signal'] = df_asc['MACD'].ewm(span=9, adjust=False).mean()
        macd_latest = df_asc['MACD'].iloc[-1]
        signal_latest = df_asc['Signal'].iloc[-1]
        macd_prev = df_asc['MACD'].iloc[-2] if len(df_asc) >= 2 else np.nan
        signal_prev = df_asc['Signal'].iloc[-2] if len(df_asc) >= 2 else np.nan
        macd_signal = '观察'
        if not np.isnan(macd_prev) and not np.isnan(signal_prev):
            if macd_latest > signal_latest and macd_prev < signal_prev: macd_signal = '金叉'
            elif macd_latest < signal_latest and macd_prev > signal_prev: macd_signal = '死叉'

        # 3. 移动平均线和趋势分析
        df_asc['MA50'] = df_asc['value'].rolling(window=50, min_periods=1).mean()
        # MA250 计算仍然保留，数据不足时会自动产生 NaN
        df_asc['MA250'] = df_asc['value'].rolling(window=250, min_periods=1).mean() 
        
        ma50_latest = df_asc['MA50'].iloc[-1]
        ma250_latest = df_asc['MA250'].iloc[-1]
        value_latest = df_asc['value'].iloc[-1]
        
        net_to_ma50 = value_latest / ma50_latest if ma50_latest and ma50_latest != 0 else np.nan
        
        # 只有在数据足够时才计算 MA250 相关指标
        if len(df_asc) < 250:
            net_to_ma250 = np.nan
            ma50_to_ma250 = np.nan
            trend_direction = '数据不足'
        else:
            net_to_ma250 = value_latest / ma250_latest if ma250_latest and ma250_latest != 0 else np.nan
            ma50_to_ma250 = ma50_latest / ma250_latest if ma250_latest and ma250_latest != 0 else np.nan
        
            # 4. MA50/MA250 趋势方向判断
            trend_direction = '数据不足'
            recent_ratio = (df_asc['MA50'] / df_asc['MA250']).tail(20).dropna()
            if len(recent_ratio) >= 5:
                slope = np.polyfit(np.arange(len(recent_ratio)), recent_ratio.values, 1)[0]
                if slope > 0.001: trend_direction = '向上'
                elif slope < -0.001: trend_direction = '向下'
                else: trend_direction = '平稳'
        
        # 5. 当日涨跌幅 (最新一天涨跌幅)
        daily_drop = 0.0
        if len(df_asc) >= 2:
            value_t_minus_1 = df_asc['value'].iloc[-2]
            if value_t_minus_1 > 0:
                # 标准涨跌幅：(现值 - 前值) / 前值。负值代表跌幅，正值代表涨幅。
                daily_drop = (value_latest - value_t_minus_1) / value_t_minus_1
                
        # 6. 布林带位置 (调用了 calculate_bollinger_bands)
        bollinger_position = calculate_bollinger_bands(df_asc['value'])

        return {
            'RSI(14)': round(rsi_14_latest, 2) if not math.isnan(rsi_14_latest) else np.nan, # 键名更新
            'RSI(6)': round(rsi_6_latest, 2) if not math.isnan(rsi_6_latest) else np.nan,   # 新增
            'MACD信号': macd_signal,
            '净值/MA50': round(net_to_ma50, 2) if not math.isnan(net_to_ma50) else np.nan,
            '净值/MA250': round(net_to_ma250, 2) if not math.isnan(net_to_ma250) else np.nan, 
            'MA50/MA250': round(ma50_to_ma250, 2) if not math.isnan(ma50_to_ma250) else np.nan, 
            'MA50/MA250趋势': trend_direction,
            '布林带位置': bollinger_position, 
            '最新净值': round(value_latest, 4) if not math.isnan(value_latest) else np.nan,
            '当日跌幅': round(daily_drop, 4) 
        }

    except Exception as e:
        logging.error(f"计算技术指标时发生错误: {e}")
        return {
            'RSI(14)': np.nan, 
            'RSI(6)': np.nan, # 新增RSI(6)
            'MACD信号': '计算错误', 
            '净值/MA50': np.nan,
            '净值/MA250': np.nan, 
            'MA50/MA250': np.nan, 
            'MA50/MA250趋势': '计算错误',
            '布林带位置': '计算错误',
            '最新净值': np.nan,
            '当日跌幅': np.nan
        }

# --- 连续下跌计算 (函数配置 5/13) ---
def calculate_consecutive_drops(series):
    """
    计算净值序列中最大的连续下跌天数 (t < t-1)
    假设 series 是按时间升序排列的（最新值在末尾）。
    """
    try:
        if series.empty or len(series) < 2: return 0
        
        # 1. 直接使用 series (已是升序)
        series_asc = series
        
        # 2. 标记每一天相对于前一天是否下跌（当前值 < 前值）
        # diff() 计算 t - t-1。如果结果 < 0，则代表下跌。
        # drops 是布尔数组，True 代表下跌。
        drops = (series_asc.diff() < 0).values
        
        max_drop_days = 0
        current_drop_days = 0
        
        # 从第二个元素开始遍历 (第一个 diff 是 NaN，已经是 False)
        for is_dropped in drops:
            if is_dropped: # 如果是下跌 (t < t-1)
                current_drop_days += 1
                max_drop_days = max(max_drop_days, current_drop_days)
            else: # 如果是上涨或持平 (t >= t-1)
                current_drop_days = 0
        
        return max_drop_days
    except Exception as e:
        logging.error(f"计算连续下跌天数时发生错误: {e}")
        return 0

# --- 最大回撤计算 (函数配置 6/13) ---
def calculate_max_drawdown(series):
    """
    计算最大回撤
    假设 series 是按时间升序排列的（最新值在末尾）。
    """
    try:
        if series.empty: return 0.0
        
        # 1. 计算累计最高点
        rolling_max = series.cummax()
        
        # 2. 最大回撤 = (最高点 - 当前点) / 最高点
        drawdown = (rolling_max - series) / rolling_max
        return drawdown.max()
    except Exception as e:
        logging.error(f"计算最大回撤时发生错误: {e}")
        return 0.0

# --- 行动提示生成 (函数配置 7/13) ---
def get_action_prompt(rsi_val, daily_drop_val, mdd_recent_month, max_drop_days_week):
    """
    根据技术指标生成基础行动提示，移除 max_drop_days_week == 1 的干扰条件。
    注意：rsi_val 传入的是 RSI(14) 的值
    """
    
    # 优先筛选：一个月回撤 >= 10% (HIGH_ELASTICITY_MIN_DRAWDOWN)
    if mdd_recent_month >= HIGH_ELASTICITY_MIN_DRAWDOWN:
        if pd.isna(rsi_val): return '高回撤观察 (RSI数据缺失)'
        
        # P1 极值超卖
        if rsi_val <= EXTREME_RSI_THRESHOLD_P1:
            return f'🌟 P1-极值超卖 (RSI<={EXTREME_RSI_THRESHOLD_P1:.0f})'
        # P2 强力超卖
        elif rsi_val <= STRONG_RSI_THRESHOLD_P2:
            return f'🔥 P2-强力超卖 (RSI<={STRONG_RSI_THRESHOLD_P2:.0f})'
        else:
            return '观察中 (RSI未超卖)'
    
    # 次要筛选：基础回撤 6% <= 回撤 < 10%
    if mdd_recent_month >= MIN_MONTH_DRAWDOWN:
         return f'关注 (回撤 {mdd_recent_month:.2%})'
    
    return '不适用 (未达基础回撤)'

# --- 单基金分析 (函数配置 8/13) ---
def analyze_single_fund(filepath):
    """
    分析单只基金
    """
    fund_code = os.path.splitext(os.path.basename(filepath))[0]
    df = pd.DataFrame()

    try:
        # 尝试默认 UTF-8 编码加载
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        try:
            # 尝试 GBK 编码（解决中文环境乱码问题）
            df = pd.read_csv(filepath, encoding='gbk')
        except Exception as e:
            logging.error(f"分析基金 {filepath} 时发生编码或加载错误: {e}")
            return None
    except Exception as e:
         logging.error(f"分析基金 {filepath} 时发生加载错误: {e}")
         return None

    try:
        # 检查关键列是否存在，非净值文件将直接跳过
        if 'date' not in df.columns or 'net_value' not in df.columns:
            return None
            
        df['date'] = pd.to_datetime(df['date'])
        
        # 【已修正】强制升序排列 (最早日期在最前面，最新日期在最后面)
        df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
        # 保持原始脚本逻辑：重命名列
        df = df.rename(columns={'net_value': 'value'})
        
        is_valid, msg = validate_fund_data(df, fund_code)
        if not is_valid: 
             logging.warning(f"基金 {fund_code} 数据无效: {msg}")
             return None
        
        # 截取近一个月/一周的数据，因为是升序，所以用 tail()
        df_recent_month = df['value'].tail(30)
        df_recent_week = df['value'].tail(5)
        
        mdd_recent_month = calculate_max_drawdown(df_recent_month)
        max_drop_days_week = calculate_consecutive_drops(df_recent_week)
        
        # calculate_technical_indicators 现在接收升序的df
        tech_indicators = calculate_technical_indicators(df)
        
        action_prompt = get_action_prompt(
            tech_indicators.get('RSI(14)', np.nan), # 使用 RSI(14)
            tech_indicators.get('当日跌幅', 0.0), 
            mdd_recent_month, 
            max_drop_days_week
        )
        
        # 注意：这里的条件现在只检查 MIN_MONTH_DRAWDOWN >= 6%
        if mdd_recent_month >= MIN_MONTH_DRAWDOWN:
            return {
                '基金代码': fund_code,
                '最大回撤': mdd_recent_month,
                '最大连续下跌': calculate_consecutive_drops(df['value'].tail(30)), # 再次使用近一个月数据
                '近一周连跌': max_drop_days_week,
                **tech_indicators,
                '行动提示': action_prompt
            }
        return None
    except Exception as e:
        # 捕获后续处理中的其他错误 (如计算错误)
        logging.error(f"分析基金 {filepath} 时发生数据处理错误: {e}")
        return None

# --- 所有基金分析 (函数配置 9/13) ---
def analyze_all_funds(target_codes=None):
    """分析所有基金数据"""
    try:
        if target_codes:
            # 目标代码模式：从 FUND_DATA_DIR 中查找特定文件
            csv_files = [os.path.join(FUND_DATA_DIR, f'{code}.csv') for code in target_codes if os.path.exists(os.path.join(FUND_DATA_DIR, f'{code}.csv'))]
        else:
            # 明确指定查找 FUND_DATA_DIR 目录下的所有 CSV 文件
            csv_files = glob.glob(os.path.join(FUND_DATA_DIR, '*.csv'))
        
        if not csv_files:
            logging.warning(f"在目录 '{FUND_DATA_DIR}' 中未找到CSV文件")
            # 如果 FUND_DATA_DIR 不存在，则尝试在当前目录查找，兼容之前运行环境
            if FUND_DATA_DIR and not os.path.exists(FUND_DATA_DIR):
                logging.warning(f"目录 '{FUND_DATA_DIR}' 不存在，尝试在当前目录查找...")
                csv_files = glob.glob('*.csv')
        
        if not csv_files:
             return []
            
        logging.info(f"找到 {len(csv_files)} 个基金数据文件，开始分析...")
        qualifying_funds = []
        for filepath in csv_files:
            result = analyze_single_fund(filepath)
            if result is not None:
                qualifying_funds.append(result)
        
        logging.info(f"分析完成，共找到 {len(qualifying_funds)} 只符合基础预警条件的基金")
        return qualifying_funds
    except Exception as e:
        logging.error(f"分析所有基金时发生错误: {e}")
        return []

# --- 技术值格式化 (函数配置 10/13) ---
def format_technical_value(value, format_type='percent'):
    """格式化技术指标值用于显示"""
    if pd.isna(value): return 'NaN'
    
    # report_daily_drop 类型直接显示实际涨跌幅，负号表示下跌。
    if format_type == 'report_daily_drop':
        # 如果是负值（下跌），用红色粗体显示；如果是正值（上涨），用绿色粗体显示。
        if value < 0:
            return f"**{value:.2%}**"
        elif value > 0:
            return f"{value:.2%}" # 原始没有颜色，但习惯上是绿色，这里保持原样
        else:
            return "0.00%"
            
    if format_type == 'percent': return f"{value:.2%}"
    elif format_type == 'decimal2': return f"{value:.2f}"
    elif format_type == 'decimal4': return f"{value:.4f}"
    else: return str(value)

# --- 表格行格式化 (函数配置 11/13) ---
def format_table_row(index, row, table_part=1):
    """
    格式化 Markdown 表格行，包含颜色/符号标记，确保清晰度。
    根据 table_part 输出表的某一部分，以解决滚动条问题。
    """
    latest_value = row.get('最新净值', 1.0)
    # 计算试水价：当前净值 * (1 - 3%的跌幅)
    trial_price = latest_value * (1 - 0.03) 
    trend_display = row['MA50/MA250趋势']
    ma_ratio_display = format_technical_value(row['MA50/MA250'], 'decimal2')
    
    # 趋势风险警告
    if trend_display == '向下' and row['MA50/MA250'] < 0.95:
         trend_display = f"⚠️ **{trend_display}**"
         ma_ratio_display = f"⚠️ **{ma_ratio_display}**"
    elif pd.isna(row['MA50/MA250']) or row['MA50/MA250趋势'] == '数据不足':
        # 数据不足 250 条时，这些字段会是 NaN 或 '数据不足'
        trend_display = "---"
        ma_ratio_display = "---"
    else:
        trend_display = f"**{trend_display}**"
        ma_ratio_display = f"**{ma_ratio_display}**"
        
    # 此处使用修正后的 'report_daily_drop'，会直接显示如 -3.79%
    daily_drop_display = format_technical_value(row['当日跌幅'], 'report_daily_drop')


    if table_part == 1:
        # 表格 1 (7列): 排名, 基金代码, 最大回撤 (1M), 当日涨跌幅, RSI(14), RSI(6), 行动提示 - 新增RSI(6)
        return (
            f"| {index} | `{row['基金代码']}` | **{format_technical_value(row['最大回撤'], 'percent')}** | "
            f"{daily_drop_display} | **{row['RSI(14)']:.2f}** | **{row['RSI(6)']:.2f}** | **{row['行动提示']}** |\n"
        )
    else:
        # 表格 2 (8列): 基金代码, MACD信号, 布林带位置, 净值/MA50, MA50/MA250, 趋势, 净值/MA250, 试水买价 (跌3%)
        return (
            f"| `{row['基金代码']}` | {row['MACD信号']} | {row['布林带位置']} | "
            f"{format_technical_value(row['净值/MA50'], 'decimal2')} | {ma_ratio_display} | {trend_display} | "
            f"{format_technical_value(row['净值/MA250'], 'decimal2') if not pd.isna(row['净值/MA250']) else '---'} | `{trial_price:.4f}` |\n"
        )

# --- 报告生成 (函数配置 12/13) ---
def generate_report(results, timestamp_str):
    """
    生成完整的Markdown格式报告。
    """
    try:
        if not results:
            return (f"# 基金预警报告 ({timestamp_str} UTC+8)\n\n"
                    f"**恭喜，没有发现满足基础预警条件的基金。**")

        df_results = pd.DataFrame(results).sort_values(by='最大回撤', ascending=False).reset_index(drop=True)
        actual_total_count = len(results)

        report_parts = []
        report_parts.extend([
            f"# 基金预警报告 ({timestamp_str} UTC+8)\n\n",
            f"## 分析总结\n\n",
            # LaTeX 符号正确转义
            f"本次分析共发现 **{actual_total_count}** 只基金满足基础预警条件（近 1 个月回撤 $\\ge {MIN_MONTH_DRAWDOWN*100:.0f}\\%$）。\n",
            f"**策略更新：RSI第一优先级阈值 $\\le {EXTREME_RSI_THRESHOLD_P1:.0f}$；第二优先级阈值 $\\le {STRONG_RSI_THRESHOLD_P2:.0f}$。**\n",
            f"\n---\n"
        ])

        # 核心筛选：高弹性基金
        df_base_elastic = df_results[
            (df_results['最大回撤'] >= HIGH_ELASTICITY_MIN_DRAWDOWN)
        ].copy()
        
        # 为了兼容原始脚本的判断逻辑：当日跌幅 >= 3% (即 daily_drop <= -0.03)
        CRITICAL_DROP_INT = MIN_DAILY_DROP_PERCENT
        
        # P1A：即时恐慌买入 (当日跌幅 <= -3%)
        df_p1 = df_base_elastic[df_base_elastic['RSI(14)'] <= EXTREME_RSI_THRESHOLD_P1].copy() # 使用 RSI(14)
        # 判断：当日跌幅 <= -0.03 (即实际跌幅大于等于 3%)
        # 修正后 daily_drop < 0 代表下跌。所以判断大跌是 daily_drop <= -CRITICAL_DROP_INT
        df_p1a = df_p1[df_p1['当日跌幅'] <= -CRITICAL_DROP_INT].copy() 
        # P1B：技术共振建仓 (当日跌幅 > -3%)
        df_p1b = df_p1[df_p1['当日跌幅'] > -CRITICAL_DROP_INT].copy()  
        
        # 定义两个表格的头部和对齐分隔符
        # 表格 1 (7列): 排名, 基金代码, 最大回撤 (1M), 当日涨跌幅, RSI(14), RSI(6), 行动提示 - 更新
        TABLE_1_HEADER = f"| 排名 | 基金代码 | 最大回撤 (1M) | **当日涨跌幅** | RSI(14) | **RSI(6)** | 行动提示 |\n"
        TABLE_1_SEPARATOR = f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n" 
        
        # 表格 2 (8列): 基金代码, MACD信号, 布林带位置, 净值/MA50, MA50/MA250, 趋势, 净值/MA250, 试水买价 (跌3%)
        TABLE_2_HEADER = f"| 基金代码 | MACD信号 | 布林带位置 | 净值/MA50 | **MA50/MA250** | **趋势** | 净值/MA250 | 试水买价 (跌3%) |\n"
        TABLE_2_SEPARATOR = f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n" 
        
        
        # ----------------------------------------------------
        # 1. 🥇 第一优先级：RSI <= 29.0
        # ----------------------------------------------------
        
        # --- 报告 P1A ---
        if not df_p1a.empty:
            # 优先按跌幅从大到小排序 (负值绝对值大)
            df_p1a = df_p1a.sort_values(by=['当日跌幅', 'RSI(14)'], ascending=[True, True]).reset_index(drop=True)
            df_p1a.index = df_p1a.index + 1
            
            report_parts.extend([
                f"\n## **🥇 第一优先级 A：【即时恐慌买入】** ({len(df_p1a)}只)\n\n",
                f"**条件：** 长期超跌 + **RSI极度超卖 ($\\le {EXTREME_RSI_THRESHOLD_P1:.0f}$)** + **当日跌幅 $\\le -{MIN_DAILY_DROP_PERCENT*100:.0f}%**\n",
                r"**纪律：** 市场恐慌时出手，本金充足时应优先配置。**（最高优先级）**" + "\n\n",
                "### 核心指标 (1/2)\n",
                TABLE_1_HEADER, # 使用更新的头部
                TABLE_1_SEPARATOR # 使用更新的分割线
            ])
            for index, row in df_p1a.iterrows():
                report_parts.append(format_table_row(index, row, table_part=1))
            
            report_parts.extend([
                "\n### 趋势与技术细节 (2/2)\n",
                TABLE_2_HEADER,
                TABLE_2_SEPARATOR
            ])
            for index, row in df_p1a.iterrows():
                report_parts.append(format_table_row(index, row, table_part=2))
            
            report_parts.append("\n---\n")

        # --- 报告 P1B ---
        if not df_p1b.empty:
            df_p1b = df_p1b.sort_values(by=['RSI(14)', '最大回撤'], ascending=[True, False]).reset_index(drop=True)
            df_p1b.index = df_p1b.index + 1
            
            report_parts.extend([
                f"\n## **🥇 第一优先级 B：【技术共振建仓】** ({len(df_p1b)}只)\n\n",
                f"**条件：** 长期超跌 + **RSI极度超卖 ($\\le {EXTREME_RSI_THRESHOLD_P1:.0f}$)** + **当日跌幅 $ > -{MIN_DAILY_DROP_PERCENT*100:.0f}%**\n",
                r"**纪律：** 极值超卖，适合在非大跌日进行建仓。**（第二高优先级）**" + "\n\n",
                "### 核心指标 (1/2)\n",
                TABLE_1_HEADER, # 使用更新的头部
                TABLE_1_SEPARATOR # 使用更新的分割线
            ])
            for index, row in df_p1b.iterrows():
                report_parts.append(format_table_row(index, row, table_part=1))
                
            report_parts.extend([
                "\n### 趋势与技术细节 (2/2)\n",
                TABLE_2_HEADER,
                TABLE_2_SEPARATOR
            ])
            for index, row in df_p1b.iterrows():
                report_parts.append(format_table_row(index, row, table_part=2))
                
            report_parts.append("\n---\n")

        # ----------------------------------------------------
        # 2. 🥈 第二优先级：29.0 < RSI <= 35.0
        # ----------------------------------------------------
        df_p2 = df_base_elastic[
            (df_base_elastic['RSI(14)'] > EXTREME_RSI_THRESHOLD_P1) & # 使用 RSI(14)
            (df_base_elastic['RSI(14)'] <= STRONG_RSI_THRESHOLD_P2)   # 使用 RSI(14)
        ].copy()

        if not df_p2.empty:
            df_p2 = df_p2.sort_values(by=['RSI(14)', '最大回撤'], ascending=[True, False]).reset_index(drop=True)
            df_p2.index = df_p2.index + 1
            
            report_parts.extend([
                f"\n## **🥈 第二优先级：【强力超卖观察池】** ({len(df_p2)}只)\n\n",
                f"**条件：** 长期超跌 + **强力超卖 ($>{EXTREME_RSI_THRESHOLD_P1:.0f}$ 且 $\\le {STRONG_RSI_THRESHOLD_P2:.0f}$)**。\n",
                r"**纪律：** 接近极值，是良好的观察目标，但需等待 RSI 进一步下行或趋势确立。**（第三优先级）**" + "\n\n",
                "### 核心指标 (1/2)\n",
                TABLE_1_HEADER, # 使用更新的头部
                TABLE_1_SEPARATOR # 使用更新的分割线
            ])

            for index, row in df_p2.iterrows():
                report_parts.append(format_table_row(index, row, table_part=1))
                
            report_parts.extend([
                "\n### 趋势与技术细节 (2/2)\n",
                TABLE_2_HEADER,
                TABLE_2_SEPARATOR
            ])
            for index, row in df_p2.iterrows():
                report_parts.append(format_table_row(index, row, table_part=2))
                
            report_parts.append("\n---\n")
        else:
            report_parts.extend([
                f"\n## **🥈 第二优先级：【强力超卖观察池】**\n\n",
                f"没有基金满足 **长期超跌** 且 **RSI ($>{EXTREME_RSI_THRESHOLD_P1:.0f}$ 且 $\\le {STRONG_RSI_THRESHOLD_P2:.0f}$)** 的条件。" + "\n\n",
                f"---\n"
            ])


        # 3. 🥉 第三优先级：扩展观察池 (RSI > 35.0)
        df_p3 = df_base_elastic[
            df_base_elastic['RSI(14)'] > STRONG_RSI_THRESHOLD_P2 # 使用 RSI(14)
        ].copy()

        if not df_p3.empty:
            df_p3 = df_p3.sort_values(by='最大回撤', ascending=False).reset_index(drop=True)
            df_p3.index = df_p3.index + 1

            report_parts.extend([
                f"\n## **🥉 第三优先级：【扩展观察池】** ({len(df_p3)}只)\n\n",
                f"**条件：** 长期超跌 + **RSI $>{STRONG_RSI_THRESHOLD_P2:.0f}$ (未达强力超卖)**。\n",
                r"**纪律：** 风险较高，仅作为观察和备选，等待 RSI 进一步进入超卖区。**（最低优先级）**" + "\n\n",
                "### 核心指标 (1/2)\n",
                TABLE_1_HEADER, # 使用更新的头部
                TABLE_1_SEPARATOR # 使用更新的分割线
            ])

            for index, row in df_p3.iterrows():
                report_parts.append(format_table_row(index, row, table_part=1))
                
            report_parts.extend([
                "\n### 趋势与技术细节 (2/2)\n",
                TABLE_2_HEADER,
                TABLE_2_SEPARATOR
            ])
            for index, row in df_p3.iterrows():
                report_parts.append(format_table_row(index, row, table_part=2))

            report_parts.append("\n---\n")
        
        # 策略执行纪律（包含行业风险提示）
        report_parts.extend([
            "\n---\n",
            f"## **⚠️ 强化执行纪律：风控与行业审查**\n\n",
            f"**1. 🛑 趋势健康度（MA50/MA250 决定能否买）：**\n",
            f"    * **MA50/MA250 $\\ge 0.95$ 且 趋势方向为 '向上' 或 '平稳'** 的基金，视为 **趋势健康**，允许试水。\n",
            f"    * **若基金趋势显示 ⚠️ 向下，或 MA50/MA250 $< 0.95$，** 则表明长期处于熊市通道，**必须放弃**，无论短期超跌有多严重。\n",
            f"    * **【新基金提示】**：对于数据不足 250 条的基金，MA50/MA250 相关指标将显示 **'---'**，需结合其他指标和人工审查来判断。\n",
            f"**2. 🔍 人工行业与K线审查（排除接飞刀风险）：**\n",
            r"    * **在买入前，必须查阅基金重仓行业。** 如果基金属于近期（如近 3-6 个月）**涨幅巨大、估值过高**的板块（例如：部分AI、半导体），则即使技术超卖，也应视为**高风险回调**，建议**放弃**或**大幅缩减**试水仓位。\n",
            r"    * **同时复核 K 线图：** 确认当前价格是否距离**近半年历史高点**太近。若是，则风险高。\n",
            f"**3. I 级试水建仓（RSI极值策略）：**\n",
            f"    * 仅当基金满足：**趋势健康** + **净值/MA50 $\\le 1.0$** + **RSI $\\le {EXTREME_RSI_THRESHOLD_P1:.0f}$** 时，才进行 $\\mathbf{{I}}$ 级试水。\n",
            f"**4. 风险控制：**\n",
            f"    * 严格止损线：平均成本价**跌幅达到 8%-10%**，立即清仓止损。\n"
        ])

        return "".join(report_parts)
        
    except Exception as e:
        logging.error(f"生成报告时发生错误: {e}")
        return f"# 报告生成错误\n\n错误信息: {str(e)}"

# --- 主函数 (函数配置 13/13) ---
def main():
    """主函数"""
    try:
        setup_logging()
        try:
            # 使用带时区的当前时间
            tz = pytz.timezone('Asia/Shanghai')
            now = datetime.now(tz)
        except:
            now = datetime.now()
            logging.warning("使用时区失败，使用本地时间")
        
        timestamp_for_report = now.strftime('%Y-%m-%d %H:%M:%S')
        timestamp_for_filename = now.strftime('%Y%m%d_%H%M%S')
        dir_name = now.strftime('%Y%m')

        os.makedirs(dir_name, exist_ok=True)
        report_file = os.path.join(dir_name, f"{REPORT_BASE_NAME}_{timestamp_for_filename}.md")

        logging.info("开始分析基金数据...")
        
        results = analyze_all_funds()
        
        report_content = generate_report(results, timestamp_for_report)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logging.info(f"分析完成，报告已保存到 {report_file}")
        return True
        
    except Exception as e:
        logging.error(f"主程序执行失败: {e}")
        return False

if __name__ == '__main__':
    # 请确保 'fund_data' 目录存在，且其中包含以基金代码命名的 CSV 文件 (date, net_value)
    success = main()
    print("脚本执行完毕。所有配置和函数均已完整保留。")
