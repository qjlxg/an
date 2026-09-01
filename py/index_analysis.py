# index_analysis.py - 独立跟踪标的量化分析脚本
import akshare as ak
import pandas as pd
import numpy as np
import talib
import re
import time
import random
import sys  # 引入sys用于控制标准错误流输出实时日志
# 导入 requests 异常
from requests.exceptions import ConnectionError, Timeout, HTTPError, ChunkedEncodingError, TooManyRedirects
# 导入底层 http 客户端异常，解决 RemoteDisconnected 错误
import http.client

# --- 配置 ---
# 补充后的指数名称到 AkShare 代码的映射
# 已根据最新验证结果修正部分代码，以消除歧义并确保准确性。
INDEX_MAP = {
    '沪深300指数': '000300',
    '中证500指数': '000905',
    '中证800指数': '000906',
    '创业板指数': '399006',
    '上证指数': '000001',
    '恒生指数': 'HSI',
    '科创板50成份指数': '000688',
    '中证智能汽车主题指数': '399976',
    '中证电子指数': '000807',
    '中证军工指数': '399967',
    '中证新能源汽车指数': '399808',
    '中证医药卫生指数': '000933',
    '中证光伏产业指数': '931151',  # 修正
    '中证人工智能主题指数': '930713',  # 修正
    '中证传媒指数': '399971',
    '中证计算机主题指数': '930652',  # 修正
    '创业板50指数': '399673',
    '深圳科技创新主题指数': '399668',
    '中证1000指数': '000852',
    '中证科创创业50指数': '931448',
    '上证科创板50成份指数': '000688',
    '中证全指信息技术指数': '000993',
    '中证500信息技术指数': '000858',  # 修正
    '中证全指半导体产品与设备指数': 'H30184',
    '中证科技100指数': '931201',
    '中证5G通信主题指数': '931079',
    '中证芯片产业指数': '930851',  # 修正（主流版）
    '中证云计算与大数据主题指数': '930651',  # 修正
    '国证半导体芯片指数': '980017',
    '中证海外中国互联网50人民币指数': 'H30566',
    '中证消费电子主题指数': '931098'
}

# MACD 参数
SHORT_PERIOD = 12
LONG_PERIOD = 26
SIGNAL_PERIOD = 9

# 最大重试次数和超时设置
MAX_RETRIES = 10  # 增加到10次，提高成功率
REQUEST_TIMEOUT = 40  # 延长超时时间

# --- 配置结束 ---

def fetch_index_data(index_code, start_date):
    """
    使用 AkShare 获取指数的日K线收盘价数据，并加入增强的重试机制。
    所有的警告和错误日志将输出到 sys.stderr，实现实时监控。
    """
    for attempt in range(MAX_RETRIES):
        try:
            df = pd.DataFrame()
            if index_code == 'HSI':
                # 恒生指数
                sys.stderr.write(f" INFO: 尝试获取恒生指数 (HSI) 数据...\n")
                sys.stderr.flush()
                df = ak.index_global_hist(symbol="恒生指数", period="daily", start_date=start_date)
            elif index_code.startswith(('H', '93', '98')):  # 针对 H/93/98 开头的特殊指数
                sys.stderr.write(f" INFO: 尝试获取特殊指数 ({index_code}) 数据...\n")
                sys.stderr.flush()
                df = ak.index_zh_a_hist(symbol=index_code, period="daily", start_date=start_date)
            else:
                # 沪深 A 股通用指数 (如 000905, 399006)
                sys.stderr.write(f" INFO: 尝试获取 A 股通用指数 ({index_code}) 数据...\n")
                sys.stderr.flush()
                df = ak.index_zh_a_hist(symbol=index_code, period="daily", start_date=start_date)
           
            # 成功获取数据，跳出循环
            if not df.empty:
                df.rename(columns={'日期': 'date', '收盘': 'close'}, inplace=True)
                return df[['date', 'close']].set_index('date')
            else:
                # AkShare 接口返回空数据，通常意味着代码错误或数据源暂不支持
                raise ValueError("获取数据为空或 AkShare 接口不支持此代码")
       
        # 捕获所有可能的网络、连接和数据错误
        except (ConnectionError, Timeout, http.client.RemoteDisconnected, ValueError, HTTPError, ChunkedEncodingError, TooManyRedirects) as e:
            error_type = e.__class__.__name__
           
            # 实时日志输出到 stderr
            sys.stderr.write(f" 警告: 尝试 {attempt + 1}/{MAX_RETRIES} - 无法获取 {index_code} 数据: {error_type} - {e}\n")
            sys.stderr.flush()
           
            if attempt < MAX_RETRIES - 1:
                # 随机指数退避延迟，防止被数据源限流
                base_delay = 10  # 增加基础延迟
                # 增加随机性和指数增长
                sleep_time = random.uniform(base_delay * (attempt + 1), base_delay * (attempt + 2))
                sys.stderr.write(f" 等待 {sleep_time:.2f} 秒后重试...\n")
                sys.stderr.flush()
                time.sleep(sleep_time)
            else:
                sys.stderr.write(f" 错误: 达到最大重试次数 ({MAX_RETRIES} 次)，放弃获取 {index_code} 数据。\n")
                sys.stderr.flush()
                return pd.DataFrame()
       
        except Exception as e:
            sys.stderr.write(f" 致命错误: 发生未知错误，无法获取 {index_code} 数据: {e.__class__.__name__} - {e}\n")
            sys.stderr.flush()
            return pd.DataFrame()
   
    return pd.DataFrame()

def analyze_and_suggest(df_data, index_name, fund_name):
    """
    对单一指数应用 MACD 指标，并输出买卖信号。
    """
    if len(df_data) < LONG_PERIOD * 2:
        return f" [ {index_name} ] 数据不足（{len(df_data)}条），跳过技术分析。"
   
    # 计算 MACD 指标
    df_nav = df_data.copy()
    # 确保输入是 float 类型，并处理 NaN
    close_prices = df_nav['close'].fillna(method='ffill').values.astype(float)
   
    df_nav['MACD'], df_nav['MACD_Signal'], df_nav['MACD_Hist'] = \
        talib.MACD(close_prices,
                   fastperiod=SHORT_PERIOD,
                   slowperiod=LONG_PERIOD,
                   signalperiod=SIGNAL_PERIOD)
   
    df_nav['Signal'] = np.where(df_nav['MACD'] > df_nav['MACD_Signal'], 1, 0)
    df_nav['Position'] = df_nav['Signal'].diff()
   
    # 提取最近的交易信号
    # 确保信号日期在当前日期之前
    recent_signals = df_nav[df_nav['Position'].abs() == 1].tail(3)
   
    report_output = [f"\n--- 📈 {index_name} ({fund_name} 的跟踪标的) 最新信号 ---"]
   
    if recent_signals.empty:
        report_output.append(" 未检测到有效信号。")
    else:
        for index, row in recent_signals.iterrows():
            action = "买入/加仓 (金叉)" if row['Position'] == 1 else "卖出/减仓 (死叉)"
            # 日期格式化，去除时间部分
            date_str = pd.to_datetime(index).strftime('%Y-%m-%d')
            report_output.append(f" 日期: {date_str}, 信号: {action}, 指数收盘价: {row['close']:.2f}")
    # 判断最新状态
    current_signal = df_nav['Signal'].iloc[-1]
    current_date_str = df_nav.index[-1]
    current_position = "多头 (建议持有或加仓)" if current_signal == 1 else "空头 (建议观望或减仓)"
    report_output.append(f" 当前状态 ({current_date_str}): {current_position}")
   
    return "\n".join(report_output)

def main_analysis():
    # 1. 读取 fund_basic_data_c_class.csv
    try:
        # 使用 utf-8-sig 应对可能存在的 BOM
        df_funds = pd.read_csv('fund_basic_data_c_class.csv', encoding='utf_8_sig')
    except FileNotFoundError:
        error_msg = "错误：未找到 fund_basic_data_c_class.csv 文件。请确保您的数据抓取工作流已运行。"
        print(error_msg, file=sys.stderr)
        return error_msg
    except Exception as e:
        error_msg = f"读取 CSV 文件出错: {e}"
        print(error_msg, file=sys.stderr)
        return error_msg
   
    # 设置分析数据的起始日期为一年前
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=1)).strftime('%Y%m%d')
    # 报告累加器，内容将最终输出到文件
    full_report = [f"【基金跟踪标的量化分析报告】\n生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} (UTC)\n--------------------------------------------------"]
   
    total_funds = len(df_funds)
   
    # 2. 遍历每只基金进行分析
    for idx, (index, row) in enumerate(df_funds.iterrows()):
        fund_code = row['基金代码']
        fund_name = row['基金简称']
        # 确保 tracking_index_str 是字符串类型
        tracking_index_str = str(row['跟踪标的'])
       
        # 实时进度信息输出到 stderr
        progress_msg = f"[{idx + 1}/{total_funds}] 正在处理基金: {fund_name} ({fund_code}) - 跟踪标的: {tracking_index_str}..."
        sys.stderr.write(progress_msg + '\n')
        sys.stderr.flush()
       
        # 3. 明确跳过 '该基金无跟踪标的' 或为空的记录
        if tracking_index_str.strip() == 'nan' or tracking_index_str.strip() == '该基金无跟踪标的' or not tracking_index_str.strip():
            full_report.append(f" **跳过:** 基金 {fund_name} 无跟踪标的。")
            continue
       
        # 报告文件内容（输出到 stdout）
        header = f"\n==================================================\n🔬 正在分析指数基金: {fund_name} ({fund_code})\n 跟踪标的: {tracking_index_str}\n=================================================="
        full_report.append(header)
       
        # 4. 尝试从跟踪标的字符串中匹配指数名称 (优化：忽略大小写、括号、特殊字符)
        matched_index_name = None
        # 移除括号、空格、连字符并转小写
        cleaned_tracking_str = re.sub(r'[\(\（\)\）\s-]', '', tracking_index_str).strip().lower()
        for name in INDEX_MAP.keys():
            cleaned_name = re.sub(r'[\(\（\)\）\s-]', '', name).strip().lower()
            # 使用包含关系进行宽松匹配
            if cleaned_name in cleaned_tracking_str or cleaned_tracking_str in cleaned_name:
                matched_index_name = name
                break
       
        if not matched_index_name:
            full_report.append(f" **跳过:** 跟踪标的 '{tracking_index_str}' 未在映射表中或无法匹配。")
            continue
       
        index_code = INDEX_MAP[matched_index_name]
        full_report.append(f"\n-> 开始分析跟踪标的: {matched_index_name} (代码: {index_code})")
       
        # 5. 抓取数据并分析 (包含重试逻辑)
        df_data = fetch_index_data(index_code, start_date)
       
        if not df_data.empty:
            analysis_result = analyze_and_suggest(df_data, matched_index_name, fund_name)
            full_report.append(analysis_result)
        else:
            full_report.append(f" **错误:** 无法获取 {matched_index_name} ({index_code}) 的历史数据，请检查网络或指数代码。")
       
        full_report.append("--------------------------------------------------")
        
        # 全局延迟：每处理一个基金，随机等待5-15秒，减少API调用频率
        global_sleep = random.uniform(5, 15)
        sys.stderr.write(f" 全局延迟: 等待 {global_sleep:.2f} 秒以避免限流...\n")
        sys.stderr.flush()
        time.sleep(global_sleep)
   
    return "\n".join(full_report)

if __name__ == '__main__':
    # 必要的库检查
    try:
        import akshare
        # 检查 talib 是否可用，如果不可用则退出
        try:
            import talib
        except ImportError:
            print("致命错误：talib 库未安装或安装失败。请先安装 TA-Lib 并在 Python 中安装 talib 库。", file=sys.stderr)
            exit(1)
       
        import pandas as pd
        import requests
        import http.client
    except ImportError as e:
        # 致命错误输出到 stderr
        print(f"致命错误：请确保已安装 akshare, talib, pandas, requests 库。缺少: {e}", file=sys.stderr)
        exit(1)
   
    report_content = main_analysis()
   
    # 最终将报告内容输出到标准输出 (会被重定向到文件)
    print(report_content)
