import pandas as pd
import glob
import os
import numpy as np
from datetime import datetime
import pytz
import logging
import math

# --- V5.0 策略所需配置参数 ---
FUND_DATA_DIR = 'fund_data'
MIN_MONTH_DRAWDOWN = 0.06 # V5.0 震荡市核心触发 (回撤 >= 5%, 此处使用 6% 近似筛选)
HIGH_ELASTICITY_MIN_DRAWDOWN = 0.15 # 高弹性策略的基础回撤要求 (15%)
MIN_DAILY_DROP_PERCENT = 0.03 # 当日大跌的定义 (3%)
REPORT_BASE_NAME = 'fund_warning_report_v5_final_filter'

# --- 核心阈值调整 ---
EXTREME_RSI_THRESHOLD_P1 = 29.0 # 网格级：RSI(14) 极值超卖
STRONG_RSI_THRESHOLD_P2 = 35.0 # 强力超卖观察池
SHORT_TERM_RSI_EXTREME = 20.0 # RSI(6)的极值超卖阈值
TREND_HEALTH_THRESHOLD = 0.95 # MA50/MA250 健康度阈值
MIN_BUY_SIGNAL_SCORE = 4.0 # 【新增】可试仓组的最低信号分数要求 (只允许 网格级/高吸/防御)

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
    """
    df_asc = df.copy()

    try:
        if 'value' not in df_asc.columns or len(df_asc) < 60:
            return {
                'RSI(14)': np.nan, 'RSI(6)': np.nan, 'MACD信号': '数据不足', 
                '净值/MA50': np.nan, '净值/MA250': np.nan, 'MA50/MA250': np.nan, 
                'MA50/MA250趋势': '数据不足', '布林带位置': '数据不足', 
                '最新净值': df_asc['value'].iloc[-1] if not df_asc.empty else np.nan,
                '当日跌幅': np.nan
            }

        delta = df_asc['value'].diff()

        # 1. RSI (14) & (6)
        for window in [14, 6]:
            gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan) 
            df_asc[f'RSI_{window}'] = 100 - (100 / (1 + rs))

        rsi_14_latest = df_asc['RSI_14'].iloc[-1]
        rsi_6_latest = df_asc['RSI_6'].iloc[-1]
        
        # 2. MACD
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
            is_golden_cross = macd_latest > signal_latest and macd_prev < signal_prev
            
            if is_golden_cross:
                if macd_latest > 0: macd_signal = '强势金叉'
                elif macd_latest < 0: macd_signal = '弱势金叉'
                else: macd_signal = '金叉'
        
        # 3. 移动平均线和趋势分析
        df_asc['MA50'] = df_asc['value'].rolling(window=50, min_periods=1).mean()
        df_asc['MA250'] = df_asc['value'].rolling(window=250, min_periods=1).mean() 
        
        ma50_latest = df_asc['MA50'].iloc[-1]
        ma250_latest = df_asc['MA250'].iloc[-1]
        value_latest = df_asc['value'].iloc[-1]
        
        net_to_ma50 = value_latest / ma50_latest if ma50_latest and ma50_latest != 0 else np.nan
        
        if len(df_asc) < 250:
            net_to_ma250 = np.nan
            ma50_to_ma250 = np.nan
            trend_direction = '数据不足'
        else:
            net_to_ma250 = value_latest / ma250_latest if ma250_latest and ma250_latest != 0 else np.nan
            ma50_to_ma250 = ma50_latest / ma250_latest if ma250_latest and ma250_latest != 0 else np.nan
            
            trend_direction = '数据不足'
            recent_ratio = (df_asc['MA50'] / df_asc['MA250']).tail(20).dropna()
            if len(recent_ratio) >= 5:
                # 拟合MA50/MA250比值的斜率
                slope = np.polyfit(np.arange(len(recent_ratio)), recent_ratio.values, 1)[0]
                if slope > 0.001: trend_direction = '向上'
                elif slope < -0.001: trend_direction = '向下'
                else: trend_direction = '平稳'
        
        # 4. 当日涨跌幅
        daily_drop = 0.0
        if len(df_asc) >= 2:
            value_t_minus_1 = df_asc['value'].iloc[-2]
            if value_t_minus_1 > 0:
                daily_drop = (value_latest - value_t_minus_1) / value_t_minus_1
                
        # 5. 布林带位置
        bollinger_position = calculate_bollinger_bands(df_asc['value'])

        return {
            'RSI(14)': round(rsi_14_latest, 2) if not math.isnan(rsi_14_latest) else np.nan, 
            'RSI(6)': round(rsi_6_latest, 2) if not math.isnan(rsi_6_latest) else np.nan,     
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
            'RSI(14)': np.nan, 'RSI(6)': np.nan, 'MACD信号': '计算错误', 
            '净值/MA50': np.nan, '净值/MA250': np.nan, 'MA50/MA250': np.nan, 
            'MA50/MA250趋势': '计算错误', '布林带位置': '计算错误',
            '最新净值': np.nan, '当日跌幅': np.nan
        }

# --- 连续下跌计算 (函数配置 5/13) ---
def calculate_consecutive_drops(series):
    try:
        if series.empty or len(series) < 2: return 0
        drops = (series.diff() < 0).values
        max_drop_days = 0
        current_drop_days = 0
        
        for is_dropped in drops:
            if is_dropped: 
                current_drop_days += 1
                max_drop_days = max(max_drop_days, current_drop_days)
            else: 
                current_drop_days = 0
        
        return max_drop_days
    except Exception as e:
        logging.error(f"计算连续下跌天数时发生错误: {e}")
        return 0

# --- 最大回撤计算 (函数配置 6/13) ---
def calculate_max_drawdown(series):
    try:
        if series.empty: return 0.0
        rolling_max = series.cummax()
        drawdown = (rolling_max - series) / rolling_max
        return drawdown.max()
    except Exception as e:
        logging.error(f"计算最大回撤时发生错误: {e}")
        return 0.0

# --- V5.0 行动信号生成 (函数配置 7/13) ---
def generate_v5_action_signal(row):
    """
    根据 V5.0 策略的技术要求，生成试仓信号。
    """
    rsi_14_val = row.get('RSI(14)', np.nan)
    rsi_6_val = row.get('RSI(6)', np.nan)
    macd_signal = row.get('MACD信号', '')
    bollinger_position = row.get('布林带位置', '')
    mdd_recent_month = row.get('最大回撤', 0.0)
    daily_drop_val = row.get('当日跌幅', 0.0)
    
    signals = []

    # --- V5.0 网格级 / 极值超卖信号 (最高优先级，独立于姿态) ---
    if not pd.isna(rsi_14_val) and rsi_14_val <= EXTREME_RSI_THRESHOLD_P1:
        rsi_display = f"RSI14:{rsi_14_val:.1f}"
        if rsi_6_val <= SHORT_TERM_RSI_EXTREME:
            signals.append(f"💥【网格级】RSI极值共振({rsi_display})")
        elif daily_drop_val <= -MIN_DAILY_DROP_PERCENT:
            signals.append(f"💥【网格级】RSI极值+恐慌({rsi_display})")
        else:
            signals.append(f"🌟【网格级】RSI极值({rsi_display})")

    # --- V5.0 游击姿态 (震荡市) 信号 ---
    if mdd_recent_month >= MIN_MONTH_DRAWDOWN:
        if bollinger_position in ["**下轨下方**", "下轨附近"]:
            signals.append("🎯【震荡-高吸】触及BOLL下轨")
        elif mdd_recent_month >= HIGH_ELASTICITY_MIN_DRAWDOWN:
            signals.append("🔥【震荡-预警】高弹性回撤达标")
        elif not signals:
            signals.append("【震荡-关注】基础回撤达标")

    # --- V5.0 防御姿态 (熊市) 信号 ---
    if macd_signal == '弱势金叉':
        signals.append("🛡️【防御-反弹】MACD弱金叉")
        
    # --- V5.0 进攻姿态 (牛市) 过滤器检查 ---
    if not pd.isna(rsi_14_val) and rsi_14_val > 70.0:
        signals.append("🚫【牛市过滤器】RSI(14)>70")
        
    if not signals:
        return '等待信号 (未达基础回撤)'
        
    return ' | '.join(signals)

# --- 遍历并分析所有基金 (函数配置 8/13) ---
def analyze_all_funds():
    """遍历 FUND_DATA_DIR 下所有 CSV 文件并分析"""
    fund_files = glob.glob(os.path.join(FUND_DATA_DIR, '*.csv'))
    results = []
    
    if not fund_files:
        logging.warning(f"在目录 '{FUND_DATA_DIR}' 中未找到任何基金数据文件。")
        return results

    for filepath in fund_files:
        fund_result = analyze_single_fund(filepath)
        if fund_result:
            results.append(fund_result)
            
    logging.info(f"所有基金分析完成，共 {len(results)} 个基金符合报告条件。")
    return results

# --- 单基金分析 (函数配置 9/13) ---
def analyze_single_fund(filepath):
    fund_code = os.path.splitext(os.path.basename(filepath))[0]
    df = pd.DataFrame()

    try:
        # 尝试使用 UTF-8, 失败后尝试 GBK
        try:
            df = pd.read_csv(filepath)
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='gbk')
        
        if 'date' not in df.columns or 'net_value' not in df.columns:
            if 'Date' in df.columns and 'NetValue' in df.columns:
                 df = df.rename(columns={'Date': 'date', 'NetValue': 'net_value'})
            else:
                return None
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
        df = df.rename(columns={'net_value': 'value'})
        
        is_valid, msg = validate_fund_data(df, fund_code)
        if not is_valid: 
            return None
        
        df_recent_month = df['value'].tail(30)
        
        mdd_recent_month = calculate_max_drawdown(df_recent_month)
        
        tech_indicators = calculate_technical_indicators(df)
        
        row_data = {**tech_indicators, '最大回撤': mdd_recent_month, '当日跌幅': tech_indicators['当日跌幅']}
        
        action_prompt = generate_v5_action_signal(row_data)
        
        if not pd.isna(tech_indicators['最新净值']):
             return {
                '基金代码': fund_code,
                '最大回撤': mdd_recent_month,
                '最大连续下跌': calculate_consecutive_drops(df['value'].tail(30)),
                '近一周连跌': calculate_consecutive_drops(df['value'].tail(5)),
                **tech_indicators,
                '行动提示': action_prompt
            }
        return None
    except Exception as e:
        logging.error(f"分析基金 {filepath} 时发生数据处理错误: {e}")
        return None

# --- 技术值格式化 (函数配置 10/13) ---
def format_technical_value(value, format_type='percent'):
    if pd.isna(value): return '---'
    
    if format_type == 'report_daily_drop':
        if value < 0:
            return f"**{value:.2%}**"
        elif value > 0:
            return f"{value:.2%}" 
        else:
            return "0.00%"
            
    if format_type == 'percent': return f"{value:.2%}"
    elif format_type == 'decimal2': return f"{value:.2f}"
    elif format_type == 'decimal4': return f"{value:.4f}"
    else: return str(value)

# --- 表格行格式化 (函数配置 11/13) ---
def format_table_row(index, row, table_part=1):
    latest_value = row.get('最新净值', 1.0)
    trial_price = latest_value * (1 - 0.03) # 预估跌3%时的试水价格
    
    trend_display = row['MA50/MA250趋势']
    ma_ratio = row.get('MA50/MA250')
    ma_ratio_display = format_technical_value(ma_ratio, 'decimal2')
    
    # 趋势风险警告
    is_data_insufficient = pd.isna(ma_ratio) or trend_display == '数据不足'
    
    if is_data_insufficient:
        trend_status = "---"
    elif trend_display == '向下' or (not pd.isna(ma_ratio) and ma_ratio < TREND_HEALTH_THRESHOLD): 
        trend_status = f"⚠️ **{trend_display}** ({ma_ratio_display})"
    else:
        trend_status = f"**{trend_display}** ({ma_ratio_display})"
        
    daily_drop_display = format_technical_value(row['当日跌幅'], 'report_daily_drop')


    if table_part == 1:
        # 表格 1 (7列): 排名, 基金代码, 最大回撤 (1M), 当日涨跌幅, RSI(14), RSI(6), 行动提示
        # RSI值如果极值超卖，字体加粗
        rsi14_display = f"**{row['RSI(14)']:.2f}**" if not pd.isna(row['RSI(14)']) and row['RSI(14)'] <= STRONG_RSI_THRESHOLD_P2 else f"{row['RSI(14)']:.2f}"
        rsi6_display = f"**{row['RSI(6)']:.2f}**" if not pd.isna(row['RSI(6)']) and row['RSI(6)'] <= SHORT_TERM_RSI_EXTREME else f"{row['RSI(6)']:.2f}"
        
        return (
            f"| {index} | `{row['基金代码']}` | **{format_technical_value(row['最大回撤'], 'percent')}** | "
            f"{daily_drop_display} | {rsi14_display} | {rsi6_display} | **{row['行动提示']}** |\n"
        )
    else:
        # 表格 2 (8列): 基金代码, MACD信号, 布林带位置, 净值/MA50, MA50/MA250趋势健康度, 净值/MA250, 试水买价
        return (
            f"| `{row['基金代码']}` | {row['MACD信号']} | {row['布林带位置']} | "
            f"{format_technical_value(row['净值/MA50'], 'decimal2')} | {trend_status} | "
            f"{format_technical_value(row['净值/MA250'], 'decimal2') if not pd.isna(row['净值/MA250']) else '---'} | `{trial_price:.4f}` |\n"
        )

# --- 报告生成 (函数配置 12/13) ---
def generate_report(results, timestamp_str):
    """
    生成完整的Markdown格式报告，使用综合评分实现 V5.0 最终决策排序，并增加信号下限过滤。
    """
    try:
        if not results:
            return (f"# 基金预警报告 ({timestamp_str} UTC+8)\n\n"
                    f"**恭喜，没有发现任何有效的基金数据。**")

        df_results = pd.DataFrame(results)
        
        # 1. 过滤：只保留回撤达到 MIN_MONTH_DRAWDOWN 的基金
        df_filtered = df_results[df_results['最大回撤'] >= MIN_MONTH_DRAWDOWN].copy()
        
        if df_filtered.empty:
            return (f"# 基金 V5.0 策略选股报告 ({timestamp_str} UTC+8)\n\n"
                    f"**恭喜，没有发现满足基础预警条件（近 1 个月回撤 $\\ge {MIN_MONTH_DRAWDOWN*100:.0f}\\%$）的基金。**")


        # 2. V5.0 信号分数 (Signal Score)
        df_filtered['signal_score'] = 0
        df_filtered.loc[df_filtered['行动提示'].str.contains('💥【网格级】'), 'signal_score'] = 5
        df_filtered.loc[df_filtered['行动提示'].str.contains('🌟【网格级】'), 'signal_score'] = 4.5
        df_filtered.loc[df_filtered['行动提示'].str.contains('🎯【震荡-高吸】'), 'signal_score'] = 4
        df_filtered.loc[df_filtered['行动提示'].str.contains('🛡️【防御-反弹】'), 'signal_score'] = 3
        df_filtered.loc[df_filtered['行动提示'].str.contains('🔥【震荡-预警】'), 'signal_score'] = 2
        df_filtered.loc[df_filtered['行动提示'].str.contains('【震荡-关注】'), 'signal_score'] = 1
        
        # 3. 趋势过滤器 (Trend Filter)
        def get_trend_score(row):
            trend = row['MA50/MA250趋势']
            ratio = row['MA50/MA250']
            
            if pd.isna(ratio) or trend == '数据不足':
                return 50 # 数据不足/需人工审核
                
            if trend == '向下' or ratio < TREND_HEALTH_THRESHOLD:
                return 0 # 拒绝买入
            
            return 100 # 允许买入

        df_filtered['trend_score'] = df_filtered.apply(get_trend_score, axis=1)

        # 4. V5.0 综合评分 (Final Score)
        df_filtered['final_score'] = df_filtered['signal_score'] * (df_filtered['trend_score'] / 100) * 1000 + (df_filtered['最大回撤'] * 100)
        
        # 5. 分组
        # A. 可试仓/最佳共振：趋势健康 AND 信号分数 >= MIN_BUY_SIGNAL_SCORE (4.0)
        df_buy = df_filtered[(df_filtered['trend_score'] == 100) & (df_filtered['signal_score'] >= MIN_BUY_SIGNAL_SCORE)].copy()
        
        # B. 弱信号/等待确认：趋势健康 AND 信号分数 < MIN_BUY_SIGNAL_SCORE (如 关注/预警)
        df_weak_signal = df_filtered[(df_filtered['trend_score'] == 100) & (df_filtered['signal_score'] < MIN_BUY_SIGNAL_SCORE)].copy()
        
        # C. 趋势不明确：趋势分数 = 50
        df_need_check = df_filtered[df_filtered['trend_score'] == 50].copy()
        
        # D. 趋势不健康：趋势分数 = 0
        df_reject_trend = df_filtered[df_filtered['trend_score'] == 0].copy()
        
        
        # 6. 报告排序 (各自组内排序)
        # 可试仓组：按信号分数和回撤排序 (最高优先级)
        df_buy_sorted = df_buy.sort_values(by=['signal_score', '最大回撤'], ascending=[False, False])
        # 弱信号组：按信号分数和回撤排序 (中等优先级)
        df_weak_signal_sorted = df_weak_signal.sort_values(by=['signal_score', '最大回撤'], ascending=[False, False])
        # 拒绝/审核组：按回撤排序 (次要优先级)
        df_need_check_sorted = df_need_check.sort_values(by='最大回撤', ascending=False)
        df_reject_trend_sorted = df_reject_trend.sort_values(by='最大回撤', ascending=False)
        
        
        report_parts = []
        report_parts.extend([
            f"# 基金 V5.0 策略选股报告 ({timestamp_str} UTC+8)\n\n",
            f"## 分析总结\n\n",
            f"本次分析共发现 **{len(df_filtered)}** 只基金满足基础回撤条件（$\\ge {MIN_MONTH_DRAWDOWN*100:.0f}\\%$）。\n",
            f"其中，**{len(df_buy)}** 只基金同时满足 **趋势健康度** 和 **最低信号强度** ($\ge {MIN_BUY_SIGNAL_SCORE:.1f}$)，被列为**最佳试仓目标**。\n",
            f"**决策重点：** **V5.0 策略启动必须先进行宏观环境判断！** 请优先从 I 组选择标的。\n",
            f"\n---\n"
        ])
        
        
        # A. 【可试仓/最高优先级】 (通过趋势健康度审核 & 强信号)
        if not df_buy_sorted.empty:
            report_parts.extend([
                f"\n## 🏆 I. 【V5.0 可试仓/最佳共振目标】 ({len(df_buy_sorted)}只)\n\n",
                f"**纪律：** 这些基金 **趋势健康** 且具有 **强信号**（网格级/高吸/防御），是**优先选择**的试仓标的。\n\n"
            ])
            report_parts.extend(generate_group_tables(df_buy_sorted))

        
        # B. 【弱信号/等待确认】 (趋势健康 & 弱信号)
        if not df_weak_signal_sorted.empty:
            report_parts.extend([
                f"\n## 💡 II. 【弱信号/等待确认】 ({len(df_weak_signal_sorted)}只)\n\n",
                f"**纪律：** 这些基金 **趋势健康** 但信号较弱（如【关注】/【预警】）。**需等待信号增强，或在极特殊情况下少量试仓。**\n\n"
            ])
            report_parts.extend(generate_group_tables(df_weak_signal_sorted))
        
        
        # C. 【趋势不明确/需人工审核】 (数据不足)
        if not df_need_check_sorted.empty:
            report_parts.extend([
                f"\n## 🔍 III. 【趋势不明确/需人工审核】 ({len(df_need_check_sorted)}只)\n\n",
                f"**纪律：** 这些基金**数据不足 250 天** 或 **技术指标计算有误**。回撤已达标，但需**手动核查** MA50/MA250 健康度。\n\n"
            ])
            report_parts.extend(generate_group_tables(df_need_check_sorted))
            

        # D. 【趋势不健康/必须放弃】 (未通过趋势健康度审核)
        if not df_reject_trend_sorted.empty:
            report_parts.extend([
                f"\n## ❌ IV. 【趋势不健康/必须放弃】 ({len(df_reject_trend_sorted)}只)\n\n",
                f"**纪律：** 这些基金**未通过趋势健康度审核**。**风险过高，请放弃试仓。**\n\n"
            ])
            report_parts.extend(generate_group_tables(df_reject_trend_sorted))


        # 策略执行纪律（最后再次强调 V5.0 的宏观判断）
        report_parts.extend([
            "\n---\n",
            f"## **⚠️ V5.0 宏观环境与趋势健康度审核总结**\n\n",
            f"**1. 🛑 趋势健康度（MA50/MA250 决定能否买）：**\n",
            f"    * **趋势健康**：MA50/MA250 $\\ge {TREND_HEALTH_THRESHOLD}$ 且 趋势方向为 '向上' 或 '平稳'，允许试仓。\n",
            f"    * **趋势不健康**：若基金显示 **⚠️ 向下**，或 MA50/MA250 $< {TREND_HEALTH_THRESHOLD}$，**必须放弃试仓**。\n",
            f"**2. 🌐 V1.0 试仓姿态确认（宏观环境决定仓位）：**\n",
            f"    * **在执行试仓前，必须手动判断宏观环境（牛市/震荡市/熊市），并根据 V5.0 手册确定仓位（5%, 10%, 20%）和活性区间**。\n"
        ])

        return "".join(report_parts)
        
    except Exception as e:
        logging.error(f"生成报告时发生错误: {e}")
        return f"# 报告生成错误\n\n错误信息: {str(e)}"

# --- 辅助函数：生成表格 (函数配置 13/13) ---
def generate_group_tables(df_group):
    
    TABLE_1_HEADER = f"| 排名 | 基金代码 | 最大回撤 (1M) | **当日涨跌幅** | RSI(14) | **RSI(6)** | V5.0 信号 |\n"
    TABLE_1_SEPARATOR = f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n" 
    TABLE_2_HEADER = f"| 基金代码 | MACD信号 | 布林带位置 | 净值/MA50 | **MA50/MA250健康度** | 净值/MA250 | 试水买价 (跌3%) |\n"
    TABLE_2_SEPARATOR = f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n" 
    
    parts = []
    
    parts.extend([
        "### 核心指标 (1/2)\n",
        TABLE_1_HEADER,
        TABLE_1_SEPARATOR
    ])

    current_index = 0
    for _, row in df_group.iterrows():
        current_index += 1
        parts.append(format_table_row(current_index, row, table_part=1))
        
    parts.extend([
        "\n### 趋势与技术细节 (2/2)\n",
        TABLE_2_HEADER,
        TABLE_2_SEPARATOR
    ])

    current_index = 0 
    for _, row in df_group.iterrows():
        current_index += 1
        parts.append(format_table_row(current_index, row, table_part=2))
        
    parts.append("\n---\n")
    return parts

# --- 主函数 ---
def main():
    """主函数"""
    try:
        setup_logging()
        try:
            tz = pytz.timezone('Asia/Shanghai')
            now = datetime.now(tz)
        except Exception:
            now = datetime.now()
            logging.warning("使用时区失败，使用本地时间")
            
        timestamp_for_report = now.strftime('%Y-%m-%d %H:%M:%S')
        timestamp_for_filename = now.strftime('%Y%m%d_%H%M%S')
        dir_name = now.strftime('%Y%m')

        os.makedirs(dir_name, exist_ok=True)
        report_file = os.path.join(dir_name, f"{REPORT_BASE_NAME}_{timestamp_for_filename}.md")

        logging.info("开始分析基金数据...")
        
        if not os.path.isdir(FUND_DATA_DIR):
            logging.error(f"基金数据目录 '{FUND_DATA_DIR}' 不存在，请创建该目录并放入 CSV 文件。")
            return False

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
    success = main()
    if success:
        print(f"脚本执行完毕。V5.0 策略报告已更新为'最终决策模式'，并已设置可试仓信号下限 (Score >= {MIN_BUY_SIGNAL_SCORE:.1f})。")
    else:
        print("脚本执行失败，请检查 fund_analysis.log 文件以获取详细错误信息。")