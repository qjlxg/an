import pandas as pd
import numpy as np
import re
import os
import logging
from datetime import datetime, timedelta, time
import random
from io import StringIO
import requests
import tenacity
import concurrent.futures
import time as time_module

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    # 修复: 日志文件从 'market_monitor_c.log' 改为 'market_monitor_z.log'
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_monitor_z.log', encoding='utf-8'), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义本地数据存储目录
DATA_DIR = 'fund_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class MarketMonitor:
    # 修复: 默认报告文件改为 'result_z.txt'
    # 修复: 默认输出文件改为 'market_monitor_report_z.md'
    def __init__(self, report_file='result_z.txt', output_file='market_monitor_report_z.md', filter_mode='all', rsi_threshold=None, holdings=None):
        self.report_file = report_file # 使用新的默认值
        self.output_file = output_file
        self.filter_mode = filter_mode  # 'all', 'strong_buy', 'low_rsi_buy'
        self.rsi_threshold = rsi_threshold  # e.g., 40, only for low_rsi_buy
        self.holdings = holdings or []  # List of held fund codes, for prioritization
        self.fund_codes = []
        self.fund_data = {}
        self.index_data = pd.DataFrame()  # 大盘数据
        self.index_indicators = None  # 大盘指标
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
        }

    def _load_index_data(self):
        """加载大盘数据"""
        index_file = os.path.join('index_data', '000300.csv')
        if os.path.exists(index_file):
            try:
                self.index_data = pd.read_csv(index_file, parse_dates=['date'])
                self.index_data = self.index_data.sort_values(by='date', ascending=True).reset_index(drop=True)
                logger.info("大盘数据加载成功，共 %d 行，最新日期: %s", len(self.index_data), self.index_data['date'].max().date())
                # 计算大盘指标
                self.index_indicators = self._calculate_indicators(self.index_data)
                if self.index_indicators is not None:
                    logger.info("大盘指标计算完成")
                else:
                    logger.warning("大盘数据不足，无法计算指标")
            except Exception as e:
                logger.error("加载大盘数据失败: %s", e)
                self.index_data = pd.DataFrame()
        else:
            logger.warning("大盘数据文件不存在: %s", index_file)
            self.index_data = pd.DataFrame()

    def _get_index_market_trend(self):
        """获取大盘趋势信号"""
        if self.index_indicators is None or self.index_indicators.empty:
            return "中性"
        
        latest_index = self.index_indicators.iloc[-1]
        ma_ratio = latest_index['ma_ratio']
        macd_diff = latest_index['macd'] - latest_index['signal']
        rsi = latest_index['rsi']
        
        if not np.isnan(ma_ratio) and ma_ratio > 1 and not np.isnan(macd_diff) and macd_diff > 0 and not np.isnan(rsi) and rsi < 70:
            return "强势"
        elif not np.isnan(ma_ratio) and ma_ratio < 0.95 or not np.isnan(macd_diff) and macd_diff < 0 or not np.isnan(rsi) and rsi > 70:
            return "弱势"
        else:
            return "中性"

    def _get_expected_latest_date(self):
        """根据当前时间确定期望的最新数据日期"""
        now = datetime.now()
        # 假设净值更新时间为晚上21:00
        update_time = time(21, 0)
        if now.time() < update_time:
            # 如果当前时间早于21:00，则期望最新日期为昨天
            expected_date = now.date() - timedelta(days=1)
        else:
            # 否则，期望最新日期为今天
            expected_date = now.date()
        logger.info("当前时间: %s, 期望最新数据日期: %s", now.strftime('%Y-%m-%d %H:%M:%S'), expected_date)
        return expected_date

    def _parse_report(self):
        """从 self.report_file 提取推荐基金代码 (已更新支持 result_C类.txt 的表格格式)"""
        report_path = self.report_file 
        logger.info("正在解析 %s 获取推荐基金代码...", report_path)
        if not os.path.exists(report_path):
            logger.error("报告文件 %s 不存在", report_path)
            # 修复：当文件不存在时，尝试读取为纯文本文件中的基金代码
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 尝试使用正则表达式从纯文本中提取 6 位数字代码
                extracted_codes = set(re.findall(r'\b\d{6}\b', content))
                
                if extracted_codes:
                    logger.warning(f"报告文件 {report_path} 不存在，尝试作为纯文本文件解析，提取到 {len(extracted_codes)} 个基金代码。")
                    self.fund_codes = sorted(list(extracted_codes))[:1000]
                    return
                else:
                    raise FileNotFoundError(f"{self.report_file} (未找到)")
            except Exception as e:
                logger.error(f"尝试作为纯文本解析 {report_path} 也失败: {e}")
                raise FileNotFoundError(f"{self.report_file} (未找到)")
        
        try:
            # --- START FIXED PARSING LOGIC for tab-separated file ---
            # 使用 pandas 读取 Tab 分隔的文件，使用正则表达式 r'\t+' 将多个连续的 Tab 视作一个分隔符
            # 这能解决文件中有不一致的多个 Tab 导致的列名识别问题
            df = pd.read_csv(report_path, sep=r'\t+', engine='python', dtype=str)
            
            # 尝试从 '编码' 列提取代码，如果不存在则使用第二列的列名（索引1）
            code_column = '编码'
            if code_column not in df.columns:
                 # 尝试使用第二列的列名（索引1）
                 if len(df.columns) >= 2:
                    code_column = df.columns[1]
                    logger.warning(f"列名 '编码' 未找到，使用第二列 '{code_column}' 提取基金代码。")
                 else:
                    # 如果列数不足，尝试作为纯文本文件解析 (兼容 result_z.txt 格式)
                    logger.warning(f"报告文件 {report_path} 列数不足，尝试作为纯文本文件解析。")
                    try:
                        with open(report_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        extracted_codes = set(re.findall(r'\b\d{6}\b', content))
                        
                        if extracted_codes:
                            logger.info(f"成功从纯文本模式提取到 {len(extracted_codes)} 个基金代码。")
                            self.fund_codes = sorted(list(extracted_codes))[:1000]
                            return
                        else:
                            raise ValueError(f"报告文件 {report_path} 格式错误，未找到 '{code_column}' 列或列数不足，且纯文本模式未提取到代码。")
                    except Exception as e:
                         raise ValueError(f"报告文件 {report_path} 格式错误，未找到 '{code_column}' 列或列数不足，尝试纯文本解析失败: {e}")

            # 提取、清理并验证 6 位数字代码
            extracted_codes = set()
            # 遍历选中的列，使用正则表达式确保只提取 6 位数字的代码
            for value in df[code_column].dropna().astype(str):
                # 查找 6 位数字模式
                match = re.search(r'\d{6}', value.strip())
                if match and len(match.group(0)) == 6:
                    extracted_codes.add(match.group(0))

            # --- END FIXED PARSING LOGIC ---
            
            sorted_codes = sorted(list(extracted_codes))
            self.fund_codes = sorted_codes[:1000]
            
            if not self.fund_codes:
                logger.warning("未提取到任何有效基金代码，请检查 %s", self.report_file)
            else:
                logger.info("成功提取到 %d 个基金（测试限制前1000个）: %s", len(self.fund_codes), self.fund_codes)
            
        except Exception as e:
            logger.error("解析报告文件失败: %s", e)
            raise

    def _read_local_data(self, fund_code):
        """读取本地文件，如果存在则返回DataFrame"""
        file_path = os.path.join(DATA_DIR, f"{fund_code}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, parse_dates=['date'])
                if not df.empty and 'date' in df.columns and 'net_value' in df.columns:
                    df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
                    logger.info("本地已存在基金 %s 数据，共 %d 行，最新日期为: %s", fund_code, len(df), df['date'].max().date())
                    return df
            except Exception as e:
                logger.warning("读取本地文件 %s 失败: %s", file_path, e)
        return pd.DataFrame()

    def _save_to_local_file(self, fund_code, df):
        """将DataFrame保存到本地文件，覆盖旧文件"""
        file_path = os.path.join(DATA_DIR, f"{fund_code}.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info("基金 %s 数据已成功保存到本地文件: %s", fund_code, file_path)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_fixed(10),
        retry=tenacity.retry_if_exception_type((requests.exceptions.RequestException, ValueError)),
        before_sleep=lambda retry_state: logger.info(f"重试基金 {retry_state.args[0]}，第 {retry_state.attempt_number} 次")
    )
    def _fetch_fund_data(self, fund_code, latest_local_date=None):
        """
        从网络获取基金数据，实现真正的增量更新。
        如果 latest_local_date 不为空，则只获取其之后的数据。
        """
        all_new_data = []
        page_index = 1
        has_new_data = False
        
        while True:
            url = f"http://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page={page_index}&per=20"
            logger.info("正在获取基金 %s 的第 %d 页数据...", fund_code, page_index)
            
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                content_match = re.search(r'content:"(.*?)"', response.text, re.S)
                pages_match = re.search(r'pages:(\d+)', response.text)
                
                if not content_match or not pages_match:
                    logger.error("基金 %s API返回内容格式不正确，可能已无数据或接口变更", fund_code)
                    break

                raw_content_html = content_match.group(1).replace('\\"', '"')
                total_pages = int(pages_match.group(1))
                
                tables = pd.read_html(StringIO(raw_content_html))
                
                if not tables:
                    logger.warning("基金 %s 在第 %d 页未找到数据表格，爬取结束", fund_code, page_index)
                    break
                
                df_page = tables[0]
                df_page.columns = ['date', 'net_value', 'cumulative_net_value', 'daily_growth_rate', 'purchase_status', 'redemption_status', 'dividend']
                df_page = df_page[['date', 'net_value']].copy()
                df_page['date'] = pd.to_datetime(df_page['date'], errors='coerce')
                df_page['net_value'] = pd.to_numeric(df_page['net_value'], errors='coerce')
                df_page = df_page.dropna(subset=['date', 'net_value'])
                
                # 如果是增量更新模式，检查是否已获取到本地最新数据之前的数据
                if latest_local_date:
                    new_df_page = df_page[df_page['date'].dt.date > latest_local_date]
                    if new_df_page.empty:
                        # 如果当前页没有新数据，且之前已经发现过新数据，则停止爬取
                        if has_new_data:
                            logger.info("基金 %s 已获取所有新数据，爬取结束。", fund_code)
                            break
                        # 如果当前页没有新数据，且是第一页，则说明没有新数据
                        elif page_index == 1:
                            logger.info("基金 %s 无新数据，爬取结束。", fund_code)
                            break
                    else:
                        has_new_data = True
                        all_new_data.append(new_df_page)
                        logger.info("第 %d 页: 发现 %d 行新数据", page_index, len(new_df_page))
                else:
                    # 如果是首次下载，则获取所有数据
                    all_new_data.append(df_page)

                logger.info("基金 %s 总页数: %d, 当前页: %d, 当前页行数: %d", fund_code, total_pages, page_index, len(df_page))
                
                # 如果是增量更新模式，且当前页数据比最新数据日期早，则结束循环
                if latest_local_date and (df_page['date'].dt.date <= latest_local_date).any():
                    logger.info("基金 %s 已追溯到本地数据，增量爬取结束。", fund_code)
                    break

                if page_index >= total_pages:
                    logger.info("基金 %s 已获取所有历史数据，共 %d 页，爬取结束", fund_code, total_pages)
                    break
                
                page_index += 1
                time_module.sleep(random.uniform(1, 2))  # 延长sleep到1-2秒，减少限速风险
                
            except requests.exceptions.RequestException as e:
                logger.error("基金 %s API请求失败: %s", fund_code, str(e))
                raise
            except Exception as e:
                logger.error("基金 %s API数据解析失败: %s", fund_code, str(e))
                raise

        # 合并新数据并返回
        if all_new_data:
            new_combined_df = pd.concat(all_new_data, ignore_index=True)
            return new_combined_df[['date', 'net_value']]
        else:
            return pd.DataFrame()

    def _calculate_indicators(self, df):
        """计算技术指标并生成结果字典"""
        if df is None or df.empty or len(df) < 26:
            return None

        df = df.sort_values(by='date', ascending=True)
        
        # MACD
        exp12 = df['net_value'].ewm(span=12, adjust=False).mean()
        exp26 = df['net_value'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # 布林带
        window = 20
        df['bb_mid'] = df['net_value'].rolling(window=window, min_periods=1).mean()
        df['bb_std'] = df['net_value'].rolling(window=window, min_periods=1).std()
        df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
        
        # RSI
        delta = df['net_value'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MA50
        df['ma50'] = df['net_value'].rolling(window=min(50, len(df)), min_periods=1).mean()
        df['ma_ratio'] = df['net_value'] / df['ma50']

        return df

    def _get_latest_signals(self, fund_code, df):
        """根据最新数据计算信号，结合大盘趋势调整"""
        try:
            processed_df = self._calculate_indicators(df)
            if processed_df is None:
                logger.warning("基金 %s 数据不足，跳过计算", fund_code)
                return {
                    'fund_code': fund_code, 'latest_net_value': "数据获取失败", 'rsi': np.nan, 'ma_ratio': np.nan,
                    'macd_diff': np.nan, 'bb_upper': np.nan, 'bb_lower': np.nan, 'advice': "观察", 'action_signal': 'N/A'
                }
            
            latest_data = processed_df.iloc[-1]
            latest_net_value = latest_data['net_value']
            latest_rsi = latest_data['rsi']
            latest_ma50_ratio = latest_data['ma_ratio']
            latest_macd_diff = latest_data['macd'] - latest_data['signal']
            latest_bb_upper = latest_data['bb_upper']
            latest_bb_lower = latest_data['bb_lower']

            # 获取大盘趋势
            market_trend = self._get_index_market_trend()

            advice = "观察"
            if (not np.isnan(latest_rsi) and latest_rsi > 70) or \
               (not np.isnan(latest_bb_upper) and latest_net_value > latest_bb_upper) or \
               (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio > 1.2):
                advice = "等待回调"
                # 如果大盘弱势，进一步确认卖出
                if market_trend == "弱势":
                    advice = "强烈等待回调"
            elif (not np.isnan(latest_rsi) and latest_rsi < 30) or \
                 (not np.isnan(latest_bb_lower) and latest_net_value < latest_bb_lower) or \
                 (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio < 0.8):
                advice = "可分批买入"
                # 如果大盘强势，加强买入
                if market_trend == "强势":
                    advice = "强烈分批买入"
            elif (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio > 1) and \
                 (not np.isnan(latest_macd_diff) and latest_macd_diff > 0):
                advice = "可分批买入"
                if market_trend == "强势":
                    advice = "强烈分批买入"
            elif (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio < 1) and \
                 (not np.isnan(latest_macd_diff) and latest_macd_diff < 0):
                advice = "等待回调"
                if market_trend == "弱势":
                    advice = "强烈等待回调"

            action_signal = "持有/观察"
            if not np.isnan(latest_ma50_ratio) and latest_ma50_ratio < 0.95:
                action_signal = "强卖出/规避"
                if market_trend == "弱势":
                    action_signal = "强烈强卖出/规避"
            elif (not np.isnan(latest_rsi) and latest_rsi > 70) and \
                 (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio > 1.2) and \
                 (not np.isnan(latest_macd_diff) and latest_macd_diff < 0):
                action_signal = "强卖出/规避"
                if market_trend == "弱势":
                    action_signal = "强烈强卖出/规避"
            elif (not np.isnan(latest_rsi) and latest_rsi > 65) or \
                 (not np.isnan(latest_bb_upper) and latest_net_value > latest_bb_upper) or \
                 (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio > 1.2):
                action_signal = "弱卖出/规避"
                if market_trend == "弱势":
                    action_signal = "强卖出/规避"
            elif (not np.isnan(latest_rsi) and latest_rsi < 35) and \
                 (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio < 0.9) and \
                 (not np.isnan(latest_macd_diff) and latest_macd_diff > 0):
                action_signal = "强买入"
                if market_trend == "强势":
                    action_signal = "强烈强买入"
            elif (not np.isnan(latest_rsi) and latest_rsi < 45) or \
                 (not np.isnan(latest_bb_lower) and latest_net_value < latest_bb_lower) or \
                 (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio < 1):
                action_signal = "弱买入"
                if market_trend == "强势":
                    action_signal = "强买入"
            
            # 在结果中添加大盘趋势
            return {
                'fund_code': fund_code,
                'latest_net_value': latest_net_value,
                'rsi': latest_rsi,
                'ma_ratio': latest_ma50_ratio,
                'macd_diff': latest_macd_diff,
                'bb_upper': latest_bb_upper,
                'bb_lower': latest_bb_lower,
                'advice': advice,
                'action_signal': action_signal,
                'market_trend': market_trend
            }
        except Exception as e:
            logger.error("处理基金 %s 时发生异常: %s", fund_code, str(e))
            return {
                'fund_code': fund_code,
                'latest_net_value': "数据获取失败",
                'rsi': np.nan,
                'ma_ratio': np.nan,
                'macd_diff': np.nan,
                'bb_upper': np.nan,
                'bb_lower': np.nan,
                'advice': "观察",
                'action_signal': 'N/A',
                'market_trend': self._get_index_market_trend()
            }

    def get_fund_data(self):
        """主控函数：优先从本地加载，仅在数据非最新或不完整时下载"""
        # 加载大盘数据
        self._load_index_data()
        
        # 步骤1: 解析推荐基金代码
        self._parse_report()
        if not self.fund_codes:
            logger.error("没有提取到任何基金代码，无法继续处理")
            return

        # 步骤2: 预加载本地数据并检查是否需要下载
        logger.info("开始预加载本地缓存数据...")
        fund_codes_to_fetch = []
        expected_latest_date = self._get_expected_latest_date()
        min_data_points = 26  # 确保有足够数据计算技术指标

        for fund_code in self.fund_codes:
            local_df = self._read_local_data(fund_code)
            
            if not local_df.empty:
                latest_local_date = local_df['date'].max().date()
                data_points = len(local_df)
                
                # 检查数据是否最新且完整
                if latest_local_date >= expected_latest_date and data_points >= min_data_points:
                    logger.info("基金 %s 的本地数据已是最新 (%s, 期望: %s) 且数据量足够 (%d 行)，直接加载。",
                                 fund_code, latest_local_date, expected_latest_date, data_points)
                    self.fund_data[fund_code] = self._get_latest_signals(fund_code, local_df.tail(100))
                    continue
                else:
                    if latest_local_date < expected_latest_date:
                        logger.info("基金 %s 本地数据已过时（最新日期为 %s，期望 %s），需要从网络获取新数据。",
                                     fund_code, latest_local_date, expected_latest_date)
                    if data_points < min_data_points:
                        logger.info("基金 %s 本地数据量不足（仅 %d 行，需至少 %d 行），需要从网络获取。",
                                     fund_code, data_points, min_data_points)
            else:
                logger.info("基金 %s 本地数据不存在，需要从网络获取。", fund_code)
            
            fund_codes_to_fetch.append(fund_code)

        # 步骤3: 多线程网络下载和处理
        if fund_codes_to_fetch:
            logger.info("开始使用多线程获取 %d 个基金的新数据...", len(fund_codes_to_fetch))
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_code = {executor.submit(self._process_single_fund, code): code for code in fund_codes_to_fetch}
                for future in concurrent.futures.as_completed(future_to_code):
                    fund_code = future_to_code[future]
                    try:
                        result = future.result()
                        if result:
                            self.fund_data[fund_code] = result
                    except Exception as e:
                        logger.error("处理基金 %s 数据时出错: %s", fund_code, str(e))
                        self.fund_data[fund_code] = {
                            'fund_code': fund_code, 'latest_net_value': "数据获取失败", 'rsi': np.nan,
                            'ma_ratio': np.nan, 'macd_diff': np.nan, 'bb_upper': np.nan, 'bb_lower': np.nan, 'advice': "观察", 'action_signal': 'N/A',
                            'market_trend': self._get_index_market_trend()
                        }
        else:
            logger.info("所有基金数据均来自本地缓存，无需网络下载。")
        
        if len(self.fund_data) > 0:
            logger.info("所有基金数据处理完成。")
        else:
            logger.error("所有基金数据均获取失败。")

    def _process_single_fund(self, fund_code):
        """处理单个基金数据：读取本地，下载增量，合并，保存，并计算信号"""
        local_df = self._read_local_data(fund_code)
        latest_local_date = local_df['date'].max().date() if not local_df.empty else None

        new_df = self._fetch_fund_data(fund_code, latest_local_date)
        
        if not new_df.empty:
            df_final = pd.concat([local_df, new_df]).drop_duplicates(subset=['date'], keep='last').sort_values(by='date', ascending=True)
            self._save_to_local_file(fund_code, df_final)
            return self._get_latest_signals(fund_code, df_final.tail(100))
        elif not local_df.empty:
            # 如果没有新数据，且本地有数据，则使用本地数据计算信号
            logger.info("基金 %s 无新数据，使用本地历史数据进行分析", fund_code)
            return self._get_latest_signals(fund_code, local_df.tail(100))
        else:
            # 如果既没有新数据，本地又没有数据，则返回失败
            logger.error("基金 %s 未获取到任何有效数据，且本地无缓存", fund_code)
            return None

    def generate_report(self):
        """生成市场情绪与技术指标监控报告"""
        logger.info("正在生成市场监控报告...")
        
        # **开始修改：计算北京时间 (CST/UTC+8)**
        now_utc = datetime.utcnow()
        EIGHT_HOURS = timedelta(hours=8)
        now_cst = now_utc + EIGHT_HOURS
        REPORT_DATE_CST_STR = now_cst.strftime('%Y-%m-%d %H:%M:%S')
        # **修改结束**

        report_df_list = []
        market_trend = self._get_index_market_trend()
        for fund_code in self.fund_codes:
            data = self.fund_data.get(fund_code)
            if data is not None:
                latest_net_value_str = f"{data['latest_net_value']:.4f}" if isinstance(data['latest_net_value'], (float, int)) else str(data['latest_net_value'])
                rsi_str = f"{data['rsi']:.2f}" if isinstance(data['rsi'], (float, int)) and not np.isnan(data['rsi']) else "N/A"
                ma_ratio_str = f"{data['ma_ratio']:.2f}" if isinstance(data['ma_ratio'], (float, int)) and not np.isnan(data['ma_ratio']) else "N/A"
                
                macd_signal = "N/A"
                if isinstance(data['macd_diff'], (float, int)) and not np.isnan(data['macd_diff']):
                    macd_signal = "金叉" if data['macd_diff'] > 0 else "死叉"
                
                bollinger_pos = "中轨"  # 默认中轨
                if isinstance(data['latest_net_value'], (float, int)):
                    if isinstance(data['bb_upper'], (float, int)) and not np.isnan(data['bb_upper']) and data['latest_net_value'] > data['bb_upper']:
                        bollinger_pos = "上轨上方"
                    elif isinstance(data['bb_lower'], (float, int)) and not np.isnan(data['bb_lower']) and data['latest_net_value'] < data['bb_lower']:
                        bollinger_pos = "下轨下方"
                else:
                    bollinger_pos = "N/A"
                
                report_df_list.append({
                    "基金代码": fund_code,
                    "最新净值": latest_net_value_str,
                    "RSI": rsi_str,
                    "净值/MA50": ma_ratio_str,
                    "MACD信号": macd_signal,
                    "布林带位置": bollinger_pos,
                    "投资建议": data['advice'],
                    "行动信号": data['action_signal']
                })
            else:
                # 只有当 fund_codes 不为空时才添加 '数据获取失败' 的行
                # 如果 fund_codes 为空，则 report_df_list 保持为空
                if self.fund_codes:
                     report_df_list.append({
                        "基金代码": fund_code,
                        "最新净值": "数据获取失败",
                        "RSI": "N/A",
                        "净值/MA50": "N/A",
                        "MACD信号": "N/A",
                        "布林带位置": "N/A",
                        "投资建议": "观察",
                        "行动信号": "N/A"
                    })

        report_df = pd.DataFrame(report_df_list)
        
        # 增加对空 DataFrame 的检查，避免 KeyError
        if report_df.empty:
            logger.warning("报告 DataFrame 为空，无法生成详细报告。")
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(f"# 市场情绪与技术指标监控报告\n\n")
                f.write(f"生成日期: {REPORT_DATE_CST_STR}\n\n")
                f.write(f"## 警告\n")
                f.write(f"未能从 {self.report_file} 中提取到任何有效的基金代码，请检查文件格式是否正确。\n")
            return

        # 新增：根据持仓优先排序（持仓基金排前）
        if self.holdings:
            report_df['is_holding'] = report_df['基金代码'].isin(self.holdings).astype(int)
            report_df = report_df.sort_values(by='is_holding', ascending=False).drop(columns=['is_holding'])

        # 新增：根据filter_mode过滤
        filtered_df = report_df.copy()
        if self.filter_mode == 'strong_buy':
            filtered_df = filtered_df[filtered_df['行动信号'].str.contains('强买入', na=False)]
        elif self.filter_mode == 'low_rsi_buy' and self.rsi_threshold:
            # 转换为数值
            filtered_df['RSI_num'] = pd.to_numeric(filtered_df['RSI'], errors='coerce')
            buy_signals = filtered_df['行动信号'].str.contains('买入', na=False)
            filtered_df = filtered_df[(buy_signals) & (filtered_df['RSI_num'] < self.rsi_threshold)].drop(columns=['RSI_num'])
        # 'all' 不过滤
        
        # 再次检查过滤后的 DataFrame 是否为空
        if filtered_df.empty:
             logger.info("过滤后没有符合条件的基金，生成空报告。")
             markdown_table = "无符合过滤条件的基金。\n\n"
             final_fund_count = 0
        else:
            # 定义排序优先级
            order_map_action = {
                "强烈强买入": 1,
                "强买入": 1,
                "弱买入": 2,
                "持有/观察": 3,
                "弱卖出/规避": 4,
                "强卖出/规避": 5,
                "强烈强卖出/规避": 5,
                "N/A": 6
            }
            order_map_advice = {
                "强烈分批买入": 1,
                "可分批买入": 1,
                "观察": 2,
                "等待回调": 3,
                "强烈等待回调": 3,
                "N/A": 4
            }
            
            filtered_df['sort_order_action'] = filtered_df['行动信号'].map(order_map_action)
            filtered_df['sort_order_advice'] = filtered_df['投资建议'].map(order_map_advice)
            
            # 将 NaN 替换为 N/A 并对净值等数据类型进行处理
            filtered_df['最新净值'] = pd.to_numeric(filtered_df['最新净值'], errors='coerce')
            filtered_df['RSI'] = pd.to_numeric(filtered_df['RSI'], errors='coerce')
            filtered_df['净值/MA50'] = pd.to_numeric(filtered_df['净值/MA50'], errors='coerce')

            # 按照您的新排序规则进行排序
            filtered_df = filtered_df.sort_values(
                by=['sort_order_action', 'sort_order_advice', 'RSI'],
                ascending=[True, True, True] # 优先按行动信号、其次按投资建议、最后按RSI从低到高排序
            ).drop(columns=['sort_order_action', 'sort_order_advice'])

            # 将浮点数格式化为字符串，方便Markdown输出
            filtered_df['最新净值'] = filtered_df['最新净值'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
            filtered_df['RSI'] = filtered_df['RSI'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
            filtered_df['净值/MA50'] = filtered_df['净值/MA50'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")

            # 将上述排序后的 DataFrame 转换为 Markdown
            markdown_table = filtered_df.to_markdown(index=False)
            final_fund_count = len(filtered_df)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 市场情绪与技术指标监控报告\n\n")
            f.write(f"生成日期: {REPORT_DATE_CST_STR}\n\n")
            f.write(f"## 大盘趋势分析\n")
            f.write(f"大盘（沪深300）当前趋势: **{market_trend}**\n")
            f.write(f"**说明：** 决策已结合大盘趋势调整，例如大盘强势时加强买入信号。\n\n")
            if self.holdings:
                f.write(f"**持仓基金优先显示**：{', '.join(self.holdings)}\n\n")
            if self.filter_mode != 'all':
                f.write(f"**过滤模式**：{self.filter_mode} (RSI阈值: {self.rsi_threshold if self.rsi_threshold else 'N/A'})\n\n")
            f.write(f"## 推荐基金技术指标 (处理基金数: {final_fund_count} / 原始{len(report_df)})\n")
            
            if final_fund_count > 0:
                f.write("此表格已按**行动信号优先级**排序，'强买入'基金将排在最前面。\n")
                f.write("**注意：** 当'行动信号'和'投资建议'冲突时，请以**行动信号**为准，其条件更严格，更适合机械化决策。\n\n")
            
            f.write(markdown_table)
        
        logger.info("报告生成完成: %s (过滤后基金数: %d)", self.output_file, final_fund_count)


if __name__ == "__main__":
    try:
        logger.info("脚本启动")
        # 示例：使用默认值，报告文件读取 'result_z.txt'，输出文件为 'market_monitor_report_z.md'
        monitor = MarketMonitor(report_file='result_z.txt', output_file='market_monitor_report_z.md')
        monitor.get_fund_data()
        monitor.generate_report()
        logger.info("脚本执行完成")
    except Exception as e:
        logger.error("脚本运行失败: %s", e, exc_info=True)
        raise