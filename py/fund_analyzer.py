import pandas as pd
import glob
import os
import numpy as np
from datetime import datetime
import pytz
import logging
import math

# --- 配置参数 (完整保留) ---
FUND_DATA_DIR = 'fund_data'
MIN_CONSECUTIVE_DROP_DAYS = 3
MIN_MONTH_DRAWDOWN = 0.06
HIGH_ELASTICITY_MIN_DRAWDOWN = 0.10  # 高弹性策略的基础回撤要求 (10%)
MIN_DAILY_DROP_PERCENT = 0.03  # 当日大跌的定义 (3%)
REPORT_BASE_NAME = 'fund_warning_report'

# --- 核心阈值调整 (完整保留) ---
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
    if df.empty: 
        return False, "数据为空"
    # 注意：根据您的CSV文件，净值列名为 'net_value'
    if 'net_value' not in df.columns: 
        return False, "缺少净值列 'net_value'"
    if len(df) < 250: 
        return False, f"数据点不足 (当前: {len(df)})"
    
    # 检查是否有缺失值 (只检查关键列)
    if df['net_value'].isnull().any():
         return False, "关键列 'net_value' 存在缺失值"
         
    return True, "数据有效"

# --- 数据加载和预处理 (函数配置 3/13) ---
def load_and_prepare_data(file_path):
    """加载数据，确保格式正确，并计算回报率"""
    try:
        df = pd.read_csv(file_path)
        # 统一列名
        df.columns = [col.lower() for col in df.columns]
        
        # 确保日期是升序排列，这是计算时间序列指标的基础
        df.sort_values(by='date', inplace=True)
        
        # 计算每日回报率（百分比形式，例如 0.0379 -> 3.79）
        df['daily_return'] = df['net_value'].pct_change() * 100
        
        # 移除任何可能因 pct_change 产生的 NaN（通常是第一行）
        df.dropna(subset=['net_value', 'daily_return'], inplace=True)
        
        return df
    except Exception as e:
        logging.error(f"加载或预处理数据 {file_path} 时发生错误: {e}")
        return pd.DataFrame()

# --- RSI 计算 (函数配置 4/13) ---
def calculate_rsi(df, period=14):
    """计算 RSI (相对强弱指数)"""
    df['up'] = df['daily_return'].apply(lambda x: x if x > 0 else 0)
    df['down'] = df['daily_return'].apply(lambda x: -x if x < 0 else 0)

    # 使用 ewm (指数加权移动平均)
    df['avg_up'] = df['up'].ewm(span=period, adjust=False).mean()
    df['avg_down'] = df['down'].ewm(span=period, adjust=False).mean()

    # 计算 RS (相对强度)
    # 避免除以零，如果 avg_down 为零，则 rs 设为无穷大
    df['rs'] = df['avg_up'] / df['avg_down'].replace(0, np.inf)

    # 计算 RSI
    df['rsi'] = 100 - (100 / (1 + df['rs']))
    
    # 返回最新的 RSI 值
    return df['rsi'].iloc[-1]

# --- 最大回撤计算 (函数配置 5/13) ---
def calculate_max_drawdown(df, period_days):
    """计算指定周期内的最大回撤"""
    
    if len(df) < period_days:
        return 0.0
    
    # 选取最近 period_days 的数据
    period_df = df.iloc[-period_days:].copy() # 使用 copy 避免 SettingWithCopyWarning
    
    # 1. 计算累计最高净值
    period_df['cumulative_max'] = period_df['net_value'].cummax()
    
    # 2. 计算回撤 (Drawdown)
    period_df['drawdown'] = (period_df['cumulative_max'] - period_df['net_value']) / period_df['cumulative_max']
    
    # 3. 找到最大回撤
    max_drawdown = period_df['drawdown'].max()
    
    return max_drawdown

# --- 连跌天数计算 (函数配置 6/13) ---
def calculate_consecutive_drop_days(df):
    """计算最新的连续下跌天数"""
    df['is_drop'] = df['daily_return'] < 0
    
    # 反转 is_drop 列，然后计算连续 True 的天数
    consecutive_drop = 0
    for is_drop in reversed(df['is_drop'].iloc[:-1]): # 不计算最新一天，因为最新一天可能上涨（已在每日回报率中体现）
        if is_drop:
            consecutive_drop += 1
        else:
            break
            
    return consecutive_drop

# --- 策略判断 (函数配置 7/13) ---
def determine_strategy_tip(rsi, max_drawdown_1m, max_drawdown_1y, latest_daily_return):
    """根据指标确定行动提示 (Strategy Tip)"""
    action_tip = ""

    # P1: 极值超卖 (RSI 极低)
    if rsi <= EXTREME_RSI_THRESHOLD_P1:
        action_tip += f"🌟 P1-极值超卖 (RSI<={EXTREME_RSI_THRESHOLD_P1})"

    # P2: 强力超卖 (RSI 低)
    elif rsi <= STRONG_RSI_THRESHOLD_P2:
        action_tip += f"💫 P2-强力超卖 (RSI<={STRONG_RSI_THRESHOLD_P2})"

    # 其它策略条件... (例如高弹性、连跌等，此处仅展示与RSI相关的)

    # 补充信息：最大回撤过大
    if max_drawdown_1m > HIGH_ELASTICITY_MIN_DRAWDOWN:
        if action_tip:
             action_tip += " | "
        action_tip += "⚠️ 1M回撤过大"
        
    # 如果没有任何提示，提供默认信息
    if not action_tip:
        action_tip = "👀 持续观察"

    return action_tip

# --- 单基金分析 (函数配置 8/13) ---
def analyze_single_fund(file_path):
    """分析单个基金数据并返回结果字典"""
    fund_code = os.path.basename(file_path).split('.')[0]
    df = load_and_prepare_data(file_path)
    
    is_valid, reason = validate_fund_data(df, fund_code)
    if not is_valid:
        logging.warning(f"基金 {fund_code} 数据无效: {reason}")
        return None

    try:
        # 1. 计算 RSI
        rsi = calculate_rsi(df, period=14)
        
        # 2. 计算最大回撤
        max_drawdown_1m = calculate_max_drawdown(df, period_days=20) # 假设 1M 约为 20 个交易日
        max_drawdown_1y = calculate_max_drawdown(df, period_days=250) # 假设 1Y 约为 250 个交易日
        
        # 3. 获取最新回报率 (百分比)
        latest_daily_return = df['daily_return'].iloc[-1]
        
        # 4. 获取当日净值 (用于后续判断和报告)
        latest_net_value = df['net_value'].iloc[-1]
        
        # 5. 确定行动提示
        action_tip = determine_strategy_tip(rsi, max_drawdown_1m, max_drawdown_1y, latest_daily_return)

        result = {
            'fund_code': fund_code,
            'rsi': rsi,
            'max_drawdown_1m': max_drawdown_1m,
            'max_drawdown_1y': max_drawdown_1y,
            'latest_daily_return': latest_daily_return,
            'latest_net_value': latest_net_value,
            'action_tip': action_tip
        }
        return result

    except Exception as e:
        logging.error(f"分析基金 {fund_code} 时发生错误: {e}")
        return None

# --- 所有基金分析 (函数配置 9/13) ---
def analyze_all_funds():
    """遍历所有基金数据文件进行分析"""
    # glob.glob 用于查找当前目录下的所有 .csv 文件，模拟 FUND_DATA_DIR 的行为
    file_list = glob.glob('*.csv') # 假设 .csv 文件就在当前目录
    
    results = []
    for file_path in file_list:
        result = analyze_single_fund(file_path)
        if result:
            results.append(result)
            
    return results

# --- 排序键 (函数配置 10/13) ---
def sort_key_for_report(result):
    """报告排序逻辑: 主要按 RSI 升序 (RSI越低越靠前)"""
    return result['rsi']

# --- 报告生成 (函数配置 11/13) ---
def generate_report(results, timestamp):
    """生成 Markdown 格式的报告"""
    try:
        report_parts = [
            f"# 基金超卖和高回撤警示报告\n",
            f"\n> **报告生成时间：** {timestamp}\n",
            f"\n## 🔴 P1/P2 策略触发基金列表\n",
            f"\n| 排名 | 基金代码 | 最大回撤 (1M) | 当日跌幅 | RSI(14) | 行动提示 |\n",
            f"|:---:|:---:|:---:|:---:|:---:|:---|\n"
        ]

        # 按照排序键进行排序
        sorted_results = sorted(results, key=sort_key_for_report, reverse=False)

        report_table_rows = []
        
        for rank, result in enumerate(sorted_results, 1):
            
            action_tip = result.get('action_tip', 'N/A')
            
            # 1. 提取原始回报率 (例如: 3.79)
            latest_daily_return = result.get('latest_daily_return', 0.0) 
            
            # 2. *** 核心修正逻辑：仅在下跌时显示负百分比 ***
            if latest_daily_return < 0:
                # 实际下跌时，显示负百分比
                display_percent = latest_daily_return
            else:
                # 实际上涨或持平时，显示 0.00%
                display_percent = 0.00 
            # **********************************************

            # 格式化输出到表格
            report_table_rows.append(
                f"|{rank}|{result['fund_code']}|{result['max_drawdown_1m']:.2%}|{display_percent:.2%}|{result['rsi']:.2f}|{action_tip}|"
            )
            
        report_parts.extend(report_table_rows)

        # 报告总结和操作建议 (保持不变)
        report_parts.extend([
            f"\n## 🛠️ 策略说明与操作建议\n",
            f"\n**1. 指标定义：**\n",
            f"    * **RSI(14)：** 基于 14 天收盘价的相对强弱指数，低于 {EXTREME_RSI_THRESHOLD_P1} 为极值超卖 (P1)。\n",
            f"    * **最大回撤 (1M)：** 最近 20 个交易日内，基金净值从最高点下跌的百分比最大值。\n",
            f"\n**2. 行动提示等级：**\n",
            f"    * 🌟 P1-极值超卖：市场情绪极度恐慌，达到强烈观察或底仓建仓条件。\n",
            f"    * 💫 P2-强力超卖：处于底部区域，可进行少量关注和分批试探。\n",
            f"\n**3. 投资建议：** 建议只在 **P1/P2 提示** 出现时，根据个人风险偏好，考虑**小仓位**或**I 级试水**。\n",
            f"    * **注意：** 本报告仅为技术分析参考，不构成投资建议。请结合基本面和市场环境综合判断。\n",
            f"\n**4. 风险控制：**\n",
            f"    * 严格止损线：平均成本价**跌幅达到 8%-10%**，立即清仓止损。\n"
        ])

        return "".join(report_parts)
        
    except Exception as e:
        logging.error(f"生成报告时发生错误: {e}")
        return f"# 报告生成错误\n\n错误信息: {str(e)}"

# --- 主函数 (函数配置 12/13) ---
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
            
        logging.info(f"分析完成。报告已保存至 {report_file}")

    except Exception as e:
        logging.error(f"主程序运行失败: {e}")

if __name__ == '__main__':
    main()
