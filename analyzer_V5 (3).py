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
MIN_MONTH_DRAWDOWN = 0.06 # V5.0 震荡市核心触发 (回撤 >= 6%)
HIGH_ELASTICITY_MIN_DRAWDOWN = 0.15 # 高弹性策略的基础回撤要求 (15%)
MIN_DAILY_DROP_PERCENT = 0.03 # 当日大跌的定义 (3%)
REPORT_BASE_NAME = 'fund_warning_report_v5_merged_table'

# --- 核心阈值调整 ---
EXTREME_RSI_THRESHOLD_P1 = 29.0 # 网格级：RSI(14) 极值超卖
STRONG_RSI_THRESHOLD_P2 = 35.0 # 强力超卖观察池
SHORT_TERM_RSI_EXTREME = 20.0 # RSI(6)的极值超卖阈值
TREND_HEALTH_THRESHOLD = 0.9 # MA50/MA250 健康度阈值 (0.9)
MIN_BUY_SIGNAL_SCORE = 3.7 # 最低信号分数 (根据讨论，强信号最低分设为3.7)
TREND_SLOPE_THRESHOLD = 0.005 # 趋势拟合斜率阈值

# --- 设置日志 (函数配置 1/15) ---
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

# --- 数据预处理和验证 (函数配置 2/15) ---
def load_and_preprocess_data(filepath, fund_code):
    """
    加载、预处理和验证基金数据。
    """
    try:
        try:
            df = pd.read_csv(filepath)
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='gbk')
        
        # 统一列名
        if 'date' not in df.columns or 'net_value' not in df.columns:
            if 'Date' in df.columns and 'NetValue' in df.columns:
                 df = df.rename(columns={'Date': 'date', 'NetValue': 'net_value'})
            else:
                logging.warning(f"基金 {fund_code} 缺少 'date' 或 'net_value' 列。")
                return None, "缺少关键列"
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
        df = df.rename(columns={'net_value': 'value'})
        
        if df.empty: return None, "数据为空"
        if 'value' not in df.columns: return None, "缺少净值列"
        if len(df) < 60: return None, f"数据不足60条，当前只有{len(df)}条"
        if (df['value'] <= 0).any(): return None, "存在无效净值(<=0)"
        
        return df, "数据有效"
        
    except Exception as e:
        logging.error(f"加载基金 {fund_code} 数据时发生错误: {e}")
        return None, f"加载错误: {e}"

# --- 布林带计算 (函数配置 3/15) ---
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
        
        if range_band <= 1e-6:
            return "轨道中间" 
            
        position = (latest_value - latest_lower) / range_band
        if position < 0.2:
            return "下轨附近"
        elif position > 0.8:
            return "上轨附近"
        else:
            return "轨道中间"

# --- 技术指标计算 (函数配置 4/15) ---
def calculate_technical_indicators(df):
    """
    计算基金净值的完整技术指标
    RSI 修正：使用 EMA 平滑 Gain/Loss
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

        # 1. RSI (14) & (6) - 修正为使用 EMA 平滑
        for window in [14, 6]:
            # 分离涨跌
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # 使用 EMA 平滑 Gain/Loss
            avg_gain = gain.ewm(span=window, adjust=False, min_periods=1).mean()
            avg_loss = loss.ewm(span=window, adjust=False, min_periods=1).mean()
            
            # 避免 RSI 除零错误
            rs = avg_gain / avg_loss.replace(0, 1e-10) 
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
            is_golden_cross = macd_latest > signal_latest and macd_prev <= signal_prev
            is_dead_cross = macd_latest < signal_latest and macd_prev >= signal_prev 
            
            if is_golden_cross:
                if macd_latest > 0: macd_signal = '强势金叉'
                elif macd_latest < 0: macd_signal = '弱势金叉'
                else: macd_signal = '金叉'
            elif is_dead_cross:
                macd_signal = '死叉' 
        
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
            recent_ratio = (df_asc['MA50'] / df_asc['MA250']).tail(50).dropna() 
            
            if len(recent_ratio) >= 5:
                # 拟合MA50/MA250比值的斜率
                slope = np.polyfit(np.arange(len(recent_ratio)), recent_ratio.values, 1)[0]
                if slope > TREND_SLOPE_THRESHOLD: trend_direction = '向上'
                elif slope < -TREND_SLOPE_THRESHOLD: trend_direction = '向下'
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

# --- 连续下跌计算 (函数配置 5/15) ---
def calculate_consecutive_drops(series):
    """计算连续下跌天数"""
    try:
        if series.empty or len(series) < 2: return 0
        # 检查最新一天是否下跌（与前一天相比），以便计算当前的连跌天数
        drops = (series.diff() < 0).values
        current_drop_days = 0
        
        # 从最后一天向前计算当前的连续下跌天数 (不包括第一天，因为它是 diff 的 NaN)
        for is_dropped in reversed(drops[1:]):
            if is_dropped:
                current_drop_days += 1
            else:
                break # 遇到上涨或平盘即停止
        
        return current_drop_days
    except Exception as e:
        logging.error(f"计算连续下跌天数时发生错误: {e}")
        return 0

# --- 最大回撤计算 (函数配置 6/15) ---
def calculate_max_drawdown(series):
    """计算最大回撤"""
    try:
        if series.empty: return 0.0
        rolling_max = series.cummax()
        drawdown = (rolling_max - series) / rolling_max
        return drawdown.max()
    except Exception as e:
        logging.error(f"计算最大回撤时发生错误: {e}")
        return 0.0

# --- 卖出/止损信号生成 (函数配置 7/15) ---
def generate_exit_signal(row):
    """根据 V5.0 止盈止损策略，生成退出/止损提示"""
    rsi_14_val = row.get('RSI(14)', np.nan)
    macd_signal = row.get('MACD信号', '')
    mdd_recent_month = row.get('最大回撤', 0.0)
    
    exit_signals = []
    
    # 1. 止盈信号：RSI 过买
    if not pd.isna(rsi_14_val) and rsi_14_val > 70.0:
        exit_signals.append("🚫 止盈：RSI(14) 过买")
        
    # 2. 止盈/止损信号：MACD 死叉
    if macd_signal == '死叉': 
        exit_signals.append("🚫 止盈/止损：MACD死叉")
        
    # 3. 止损信号：近一月回撤超限
    if mdd_recent_month > 0.10: 
        exit_signals.append(f"🛑 止损：回撤超 10% ({mdd_recent_month:.2%})")
        
    if not exit_signals:
        return "持有"
        
    return ' | '.join(exit_signals)

# --- V5.0 行动信号生成 (函数配置 8/15) ---
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
    consecutive_drop_recent = row.get('近10日连跌', 0) 
    
    signals = []

    # --- V5.0 网格级 / 极值超卖信号 ---
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
        if consecutive_drop_recent >= 5:
             if not any('网格级' in s for s in signals):
                signals.append("✨【震荡-连跌】连跌5日+高回撤") 
                
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

# --- 遍历并分析所有基金 (函数配置 9/15) ---
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

# --- 单基金分析 (函数配置 10/15) ---
def analyze_single_fund(filepath):
    """
    单基金分析，使用抽象后的数据加载函数。
    """
    fund_code = os.path.splitext(os.path.basename(filepath))[0]
    
    # 使用抽象函数加载数据
    df, msg = load_and_preprocess_data(filepath, fund_code)
    if df is None: 
        logging.warning(f"基金 {fund_code} 分析跳过: {msg}")
        return None
        
    try:
        # 动态日期窗口计算回撤
        latest_date = df['date'].iloc[-1]
        one_month_ago = latest_date - pd.DateOffset(months=1)
        df_recent_month = df[df['date'] >= one_month_ago]['value']
        
        if len(df_recent_month) < 2:
            mdd_recent_month = 0.0
        else:
            mdd_recent_month = calculate_max_drawdown(df_recent_month)
        
        tech_indicators = calculate_technical_indicators(df)
        
        # 注意：这里计算的 consecutive_drop_recent 已经是当前的连跌天数
        consecutive_drop_recent = calculate_consecutive_drops(df['value'].tail(10)) 

        row_data = {
            **tech_indicators, 
            '最大回撤': mdd_recent_month, 
            '当日跌幅': tech_indicators['当日跌幅'],
            '近10日连跌': consecutive_drop_recent
        }
        
        action_prompt = generate_v5_action_signal(row_data)
        exit_prompt = generate_exit_signal(row_data)
        
        if not pd.isna(tech_indicators['最新净值']):
             return {
                 '基金代码': fund_code,
                 '最大回撤': mdd_recent_month,
                 '最大连续下跌': calculate_consecutive_drops(df['value']),
                 '近10日连跌': consecutive_drop_recent,
                 **tech_indicators,
                 '行动提示': action_prompt,
                 '退出提示': exit_prompt
             }
        return None
    except Exception as e:
        logging.error(f"分析基金 {filepath} 时发生数据处理错误: {e}")
        return None

# --- 技术值格式化 (函数配置 11/15) ---
def format_technical_value(value, format_type='percent'):
    """技术值格式化"""
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

# --- 表格行格式化 (函数配置 12/15) ---
def format_table_row(index, row):
    """
    表格行格式化 (精简版 + 冲突处理)
    """
    latest_value = row.get('最新净值', 1.0)
    # 试水买价 (跌3%) 计算保持不变
    trial_price = latest_value * (1 - 0.03) 
    
    trend_display = row['MA50/MA250趋势']
    ma_ratio = row.get('MA50/MA250')
    ma_ratio_display = format_technical_value(ma_ratio, 'decimal2')
    
    is_data_insufficient = pd.isna(ma_ratio) or trend_display == '数据不足'
    
    # 趋势风险警告
    if is_data_insufficient:
        trend_status = "---"
    elif trend_display == '向下' or (not pd.isna(ma_ratio) and ma_ratio < TREND_HEALTH_THRESHOLD): 
        trend_status = f"⚠️ **{trend_display}** ({ma_ratio_display})"
    else:
        trend_status = f"**{trend_display}** ({ma_ratio_display})"
        
    daily_drop_display = format_technical_value(row['当日跌幅'], 'report_daily_drop')
    
    # RSI(14) 使用加粗显示
    rsi14_display = f"**{row['RSI(14)']:.2f}**" if not pd.isna(row['RSI(14)']) and row['RSI(14)'] <= STRONG_RSI_THRESHOLD_P2 else f"{row['RSI(14)']:.2f}"
    
    # *** 核心冲突处理逻辑 ***
    v5_signal_content = row['行动提示']
    exit_prompt = row['退出提示']
    
    if "🛑 止损：" in exit_prompt:
        # 如果触发了止损，则在 V5.0 信号前加上否决提示
        v5_signal_display = f"🚫 **止损否决** | {v5_signal_content}"
    else:
        v5_signal_display = f"**{v5_signal_content}**"


    # *** 对应精简后的表头输出 ***
    return (
        f"| {index} | `{row['基金代码']}` | **{format_technical_value(row['最大回撤'], 'percent')}** | "
        f"{daily_drop_display} | {rsi14_display} | {v5_signal_display} | "
        f"**{exit_prompt}** | "
        f"{trend_status} | `{trial_price:.4f}` |\n"
    )

# --- 报告生成 (函数配置 13/15) ---
def generate_report(results, timestamp_str):
    """生成完整的Markdown格式报告"""
    try:
        if not results:
            return (f"# 基金预警报告 ({timestamp_str} UTC+8)\n\n"
                      f"**恭喜，没有发现任何有效的基金数据。**")

        df_results = pd.DataFrame(results)
        
        # 过滤出符合基础回撤条件的基金
        df_filtered = df_results[df_results['最大回撤'] >= MIN_MONTH_DRAWDOWN].copy()
        
        if df_filtered.empty:
            return (f"# 基金 V5.0 策略选股报告 ({timestamp_str} UTC+8)\n\n"
                      f"**恭喜，没有发现满足基础预警条件（近 1 个月回撤 $\\ge {MIN_MONTH_DRAWDOWN*100:.0f}\\%$）的基金。**")


        # 1. V5.0 信号分数 
        df_filtered['signal_score'] = 0
        df_filtered.loc[df_filtered['行动提示'].str.contains('💥【网格级】RSI极值共振'), 'signal_score'] = 5.0
        df_filtered.loc[df_filtered['行动提示'].str.contains('💥【网格级】RSI极值'), 'signal_score'] = 4.5
        df_filtered.loc[df_filtered['行动提示'].str.contains('🌟【网格级】RSI极值'), 'signal_score'] = 4.5
        df_filtered.loc[df_filtered['行动提示'].str.contains('🎯【震荡-高吸】'), 'signal_score'] = 4.0
        df_filtered.loc[df_filtered['行动提示'].str.contains('✨【震荡-连跌】'), 'signal_score'] = 3.5 
        df_filtered.loc[df_filtered['行动提示'].str.contains('🛡️【防御-反弹】'), 'signal_score'] = 3.0
        df_filtered.loc[df_filtered['行动提示'].str.contains('🔥【震荡-预警】'), 'signal_score'] = 2.0
        df_filtered.loc[df_filtered['行动提示'].str.contains('【震荡-关注】'), 'signal_score'] = 1.0
        
        # 2. 趋势过滤器 
        def get_trend_score(row):
            trend = row['MA50/MA250趋势']
            ratio = row['MA50/MA250']
            
            if pd.isna(ratio) or trend == '数据不足':
                return 50 
                
            if trend == '向下' or ratio < TREND_HEALTH_THRESHOLD: 
                return 0 # 拒绝买入
            
            return 100 

        df_filtered['trend_score'] = df_filtered.apply(get_trend_score, axis=1)

        # 3. V5.0 综合评分 
        df_filtered['final_score'] = df_filtered['signal_score'] * (df_filtered['trend_score'] / 100) * 1000 + (df_filtered['最大回撤'] * 100)
        
        # *** 新增止损否决标志 ***
        # 0 = 未触发止损 (可买入)；1 = 触发止损 (否决买入)
        df_filtered['is_stop_loss'] = np.where(df_filtered['最大回撤'] > 0.10, 1, 0)
        # ------------------------------------
        
        # 4. 分组
        # 仅保留通过趋势健康度且信号强度达标的基金
        df_buy = df_filtered[(df_filtered['trend_score'] == 100) & (df_filtered['signal_score'] >= MIN_BUY_SIGNAL_SCORE)].copy()
        df_reject_trend = df_filtered[df_filtered['trend_score'] == 0].copy()
        
        
        # 5. 报告排序 (核心修改: 优先未止损，然后按信号分和回撤排序)
        df_buy_sorted = df_buy.sort_values(
            by=['is_stop_loss', 'signal_score', '最大回撤'], 
            ascending=[True, False, False] # True: 0排在前面（未止损）；False: 高分高回撤排在前面
        )
        
        # FIX: 对趋势不健康的基金进行排序
        df_reject_trend_sorted = df_reject_trend.sort_values(
            by=['最大回撤', 'signal_score'],
            ascending=[False, False]
        )
        
        
        # 6. 重新分组到 I.1 (可买) 和 I.2 (止损否决) 组
        df_i_buyable = df_buy_sorted[df_buy_sorted['is_stop_loss'] == 0] # 真正可买入的目标
        df_ii_rejected_stoploss = df_buy_sorted[df_buy_sorted['is_stop_loss'] == 1] # 趋势健康但被止损否决的目标
        
        
        report_parts = []
        report_parts.extend([
            f"# 基金 V5.0 策略选股报告 ({timestamp_str} UTC+8)\n\n",
            f"## 分析总结\n\n",
            f"本次分析共发现 **{len(df_filtered)}** 只基金满足基础回撤条件（$\\ge {MIN_MONTH_DRAWDOWN*100:.0f}\\%$）。\n",
            f"其中，**{len(df_i_buyable)}** 只基金同时满足 **趋势健康、最低信号强度** 和 **未触发止损**，被列为**最高优先级试仓目标**。\n",
            f"**决策重点：** **请优先从 🥇 I.1 组选择标的。**\n",
            f"\n---\n"
        ])
        
        
        # A. 【最高优先级可试仓】 -> I.1
        if not df_i_buyable.empty:
            report_parts.extend([
                f"\n## 🥇 I.1 【最高优先级/可试仓目标】 ({len(df_i_buyable)}只)\n\n",
                f"**纪律：** 趋势健康且具有强信号，**未触发止损纪律**。这是**唯一允许试仓**的标的池。\n\n"
            ])
            report_parts.append(generate_merged_table(df_i_buyable))

        
        # B. 【趋势健康但止损否决】 -> I.2
        if not df_ii_rejected_stoploss.empty:
            report_parts.extend([
                f"\n## 🚫 I.2 【趋势健康但止损否决】 ({len(df_ii_rejected_stoploss)}只)\n\n",
                f"**纪律：** 趋势健康且出现买入信号，但**已触发止损纪律（回撤 $> 10\%$）**。不应再投入资金。\n\n"
            ])
            report_parts.append(generate_merged_table(df_ii_rejected_stoploss))
        
        # C. 【趋势不健康/必须放弃】 -> IV. 
        if not df_reject_trend_sorted.empty:
            report_parts.extend([
                f"\n## ❌ IV. 【趋势不健康/必须放弃】 ({len(df_reject_trend_sorted)}只)\n\n",
                f"**纪律：** 这些基金**未通过趋势健康度审核**（MA50/MA250 $< {TREND_HEALTH_THRESHOLD:.1f}$ 或 趋势向下）。**风险过高，请放弃试仓。**\n\n"
            ])
            report_parts.append(generate_merged_table(df_reject_trend_sorted))


        # 策略执行纪律 (精简版)
        report_parts.extend([
            "\n---\n",
            f"## **✅ 核心决策纪律总结**\n\n",
            f"**1. 🏆 优先级：** 优先从 🥇 I.1 组选取目标，**退出提示**具有最高决策优先级。\n",
            f"**2. 🛑 趋势健康度：** 若 MA50/MA250 $< {TREND_HEALTH_THRESHOLD:.1f}$ 或趋势向下，**必须放弃试仓**。\n",
            f"**3. 💰 仓位纪律：** 请手动判断宏观环境（牛市/震荡市/熊市），并据此确定本次试仓仓位（5%, 10%, 20%）。\n"
        ])

        return "".join(report_parts)
        
    except Exception as e:
        logging.error(f"生成报告时发生错误: {e}")
        return f"# 报告生成错误\n\n错误信息: {str(e)}"

# --- 辅助函数：生成合并后的表格 (函数配置 14/15) ---
def generate_merged_table(df_group):
    """生成报告中的Markdown表格 (精简版)"""
    
    # *** 简化后的新表头 (9列) ***
    FULL_HEADER = (
        f"| 排名 | 基金代码 | **最大回撤 (1M)** | **当日跌幅** | RSI(14) | **V5.0 信号** | "
        f"**退出提示** | MA50/MA250健康度 | 试水买价 (跌3%) |\n"
    )
    FULL_SEPARATOR = f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n" 
    
    parts = []
    
    parts.extend([
        "### 综合技术分析表\n",
        FULL_HEADER,
        FULL_SEPARATOR
    ])

    current_index = 0
    for _, row in df_group.iterrows():
        current_index += 1
        parts.append(format_table_row(current_index, row)) 
        
    parts.append("\n---\n")
    return "".join(parts)

# --- 主函数 (函数配置 15/15) ---
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
            # 即使目录不存在，也生成一个空的报告
            with open(report_file, 'w', encoding='utf-8') as f:
                 f.write(f"# 基金预警报告 ({timestamp_for_report} UTC+8)\n\n**错误：** 基金数据目录 `fund_data` 不存在或为空，请检查文件路径。")
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
    # 确保 fund_data 目录存在
    if not os.path.isdir('fund_data'):
        os.makedirs('fund_data', exist_ok=True)
        
    success = main()
    if success:
        print(f"脚本执行完毕。V5.0 策略报告已更新，您的 **可买入目标** 现在会排在报告最前面（🥇 I.1 组）。")
    else:
        print("脚本执行失败，请检查 fund_analysis.log 文件以获取详细错误信息。")
