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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_monitor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义本地数据存储目录
DATA_DIR = 'fund_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class MarketMonitor:
    def __init__(self, report_file='analysis_report.md', output_file='market_monitor_report.md'):
        self.report_file = report_file
        self.output_file = output_file
        self.fund_codes = []
        self.fund_data = {}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
        }

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
        """从 analysis_report.md 提取推荐基金代码"""
        logger.info("正在解析 %s 获取推荐基金代码...", self.report_file)
        if not os.path.exists(self.report_file):
            logger.error("报告文件 %s 不存在", self.report_file)
            raise FileNotFoundError(f"{self.report_file} 不存在")
        
        try:
            with open(self.report_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            pattern = re.compile(r'(?:^\| +(\d{6})|### 基金 (\d{6}))', re.M)
            matches = pattern.findall(content)

            extracted_codes = set()
            for match in matches:
                code = match[0] if match[0] else match[1]
                extracted_codes.add(code)
            
            sorted_codes = sorted(list(extracted_codes))
            self.fund_codes = sorted_codes[:1000]
            
            if not self.fund_codes:
                logger.warning("未提取到任何有效基金代码，请检查 analysis_report.md")
            else:
                logger.info("提取到 %d 个基金（测试限制前10个）: %s", len(self.fund_codes), self.fund_codes)
            
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
    def _fetch_fund_data(self, fund_code):
        """从网络获取基金数据，并支持增量更新"""
        local_df = self._read_local_data(fund_code)
        latest_local_date = local_df['date'].max().date() if not local_df.empty else None
        
        all_new_data = []
        page_index = 1
        
        while True:
            url = f"http://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page={page_index}&per=20"
            logger.info("访问URL: %s", url)
            
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
                
                df = tables[0]
                df.columns = ['date', 'net_value', 'cumulative_net_value', 'daily_growth_rate', 'purchase_status', 'redemption_status', 'dividend']
                df = df[['date', 'net_value']].copy()
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
                df = df.dropna(subset=['date', 'net_value'])
                
                if latest_local_date:
                    # 只获取比本地最新日期更新的数据
                    new_df = df[df['date'].dt.date > latest_local_date]
                    if not new_df.empty:
                        all_new_data.append(new_df)
                        logger.info("第 %d 页: 发现 %d 行新数据", page_index, len(new_df))
                    
                    if new_df.empty and page_index == 1:
                        logger.info("基金 %s 无新数据，爬取结束", fund_code)
                        break
                else:
                    # 如果本地没有数据，则获取所有历史数据
                    all_new_data.append(df)
                    if len(df) < 20:
                        break

                logger.info("基金 %s 总页数: %d, 当前页: %d, 当前页行数: %d", fund_code, total_pages, page_index, len(df))
                
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

        # 合并新数据和旧数据
        if all_new_data:
            new_combined_df = pd.concat(all_new_data, ignore_index=True)
            df_final = pd.concat([local_df, new_combined_df]).drop_duplicates(subset=['date'], keep='last').sort_values(by='date', ascending=True)
            self._save_to_local_file(fund_code, df_final)
            df_final = df_final.tail(100)
            logger.info("成功合并并保存基金 %s 的数据，总行数: %d, 最新日期: %s, 最新净值: %.4f", 
                                 fund_code, len(df_final), df_final['date'].iloc[-1].strftime('%Y-%m-%d'), df_final['net_value'].iloc[-1])
            return df_final[['date', 'net_value']]
        else:
            if not local_df.empty:
                logger.info("基金 %s 无新数据，使用本地历史数据", fund_code)
                return local_df.tail(100)[['date', 'net_value']]
            else:
                raise ValueError("未获取到任何有效数据，且本地无缓存")

    def _calculate_indicators(self, fund_code, df):
        """计算技术指标并生成结果字典"""
        try:
            if df is None or df.empty or len(df) < 26:
                logger.warning("基金 %s 数据获取失败或数据不足，跳过计算 (数据行数: %s)", fund_code, len(df) if df is not None else 0)
                return {
                    'fund_code': fund_code, 'latest_net_value': "数据获取失败", 'rsi': np.nan, 'ma_ratio': np.nan,
                    'macd_diff': np.nan, 'bb_upper': np.nan, 'bb_lower': np.nan, 'advice': "观察", 'action_signal': 'N/A'
                }

            df = df.sort_values(by='date', ascending=True)
            
            exp12 = df['net_value'].ewm(span=12, adjust=False).mean()
            exp26 = df['net_value'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp12 - exp26
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

            window = 20
            df['bb_mid'] = df['net_value'].rolling(window=window, min_periods=1).mean()
            df['bb_std'] = df['net_value'].rolling(window=window, min_periods=1).std()
            df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
            
            delta = df['net_value'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            ma50 = df['net_value'].rolling(window=min(50, len(df)), min_periods=1).mean()
            
            latest_data = df.iloc[-1]
            latest_net_value = latest_data['net_value']
            latest_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else np.nan
            latest_ma50 = ma50.iloc[-1]
            latest_ma50_ratio = latest_net_value / latest_ma50 if not pd.isna(latest_ma50) and latest_ma50 != 0 else np.nan
            
            latest_macd_diff = latest_data['macd'] - latest_data['signal'] if 'macd' in latest_data and 'signal' in latest_data else np.nan
            latest_bb_upper = latest_data['bb_upper'] if 'bb_upper' in latest_data else np.nan
            latest_bb_lower = latest_data['bb_lower'] if 'bb_lower' in latest_data else np.nan

            advice = "观察"
            if (not np.isnan(latest_rsi) and latest_rsi > 70) or \
               (not np.isnan(latest_bb_upper) and latest_net_value > latest_bb_upper) or \
               (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio > 1.2):
                advice = "等待回调"
            elif (not np.isnan(latest_rsi) and latest_rsi < 30) or \
                 (not np.isnan(latest_bb_lower) and latest_net_value < latest_bb_lower) or \
                 (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio < 0.8):
                advice = "可分批买入"
            elif (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio > 1) and \
                 (not np.isnan(latest_macd_diff) and latest_macd_diff > 0):
                advice = "可分批买入"
            elif (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio < 1) and \
                 (not np.isnan(latest_macd_diff) and latest_macd_diff < 0):
                advice = "等待回调"

            # 新增的机械化投资决策逻辑
            action_signal = "持有/观察"

            # 强卖出/规避信号（止损逻辑）: 净值跌破MA50的5%
            if not np.isnan(latest_ma50_ratio) and latest_ma50_ratio < 0.95:
                action_signal = "强卖出/规避"
            # 强卖出/规避信号: 满足多个消极条件
            elif (not np.isnan(latest_rsi) and latest_rsi > 70) and \
               (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio > 1.2) and \
               (not np.isnan(latest_macd_diff) and latest_macd_diff < 0):
                action_signal = "强卖出/规避"
            # 弱卖出/规避信号：满足部分消极条件
            elif (not np.isnan(latest_rsi) and latest_rsi > 65) or \
                 (not np.isnan(latest_bb_upper) and latest_net_value > latest_bb_upper) or \
                 (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio > 1.2):
                action_signal = "弱卖出/规避"
            # 强买入信号：满足多个积极条件
            elif (not np.isnan(latest_rsi) and latest_rsi < 35) and \
               (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio < 0.9) and \
               (not np.isnan(latest_macd_diff) and latest_macd_diff > 0):
                action_signal = "强买入"
            # 弱买入信号：满足部分积极条件
            elif (not np.isnan(latest_rsi) and latest_rsi < 45) or \
                 (not np.isnan(latest_bb_lower) and latest_net_value < latest_bb_lower) or \
                 (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio < 1):
                action_signal = "弱买入"
            
            # 如果是买入信号，则强制覆盖卖出信号
            # 这段逻辑保证买入信号的优先级高于卖出信号，但新的排序逻辑已涵盖此功能
            # if action_signal in ["强买入", "弱买入"] and latest_ma50_ratio > 1.2:
            #     action_signal = action_signal
            # elif action_signal in ["强卖出/规避", "弱卖出/规避"] and latest_ma50_ratio < 1:
            #     action_signal = action_signal


            return {
                'fund_code': fund_code,
                'latest_net_value': latest_net_value,
                'rsi': latest_rsi,
                'ma_ratio': latest_ma50_ratio,
                'macd_diff': latest_macd_diff,
                'bb_upper': latest_bb_upper,
                'bb_lower': latest_bb_lower,
                'advice': advice,
                'action_signal': action_signal
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
                'action_signal': 'N/A'
            }
        finally:
            for handler in logger.handlers:
                handler.flush()

    def get_fund_data(self):
        """主控函数：优先从本地加载，仅在数据非最新或不完整时下载"""
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
                    self.fund_data[fund_code] = self._calculate_indicators(fund_code, local_df.tail(100))
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

        # 步骤3: 多线程网络下载
        if fund_codes_to_fetch:
            logger.info("开始使用多线程获取 %d 个基金的新数据...", len(fund_codes_to_fetch))
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_code = {executor.submit(self._fetch_fund_data, code): code for code in fund_codes_to_fetch}
                for future in concurrent.futures.as_completed(future_to_code):
                    fund_code = future_to_code[future]
                    try:
                        df = future.result()
                        result = self._calculate_indicators(fund_code, df)
                        self.fund_data[fund_code] = result
                    except Exception as e:
                        logger.error("获取和处理基金 %s 数据时出错: %s", fund_code, str(e))
                        self.fund_data[fund_code] = {
                            'fund_code': fund_code, 'latest_net_value': "数据获取失败", 'rsi': np.nan,
                            'ma_ratio': np.nan, 'macd_diff': np.nan, 'bb_upper': np.nan, 'bb_lower': np.nan, 'advice': "观察", 'action_signal': 'N/A'
                        }
        else:
            logger.info("所有基金数据均来自本地缓存，无需网络下载。")
        
        if len(self.fund_data) > 0:
            logger.info("所有基金数据处理完成。")
        else:
            logger.error("所有基金数据均获取失败。")

    def generate_report(self):
        """生成市场情绪与技术指标监控报告"""
        logger.info("正在生成市场监控报告...")
        report_df_list = []
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

        # 定义排序优先级
        order_map_action = {
            "强买入": 1,
            "弱买入": 2,
            "持有/观察": 3,
            "弱卖出/规避": 4,
            "强卖出/规避": 5,
            "N/A": 6
        }
        order_map_advice = {
            "可分批买入": 1,
            "观察": 2,
            "等待回调": 3,
            "N/A": 4
        }
        
        report_df['sort_order_action'] = report_df['行动信号'].map(order_map_action)
        report_df['sort_order_advice'] = report_df['投资建议'].map(order_map_advice)
        
        # 将 NaN 替换为 N/A 并对净值等数据类型进行处理
        report_df['最新净值'] = pd.to_numeric(report_df['最新净值'], errors='coerce')
        report_df['RSI'] = pd.to_numeric(report_df['RSI'], errors='coerce')
        report_df['净值/MA50'] = pd.to_numeric(report_df['净值/MA50'], errors='coerce')

        # 按照您的新排序规则进行排序
        report_df = report_df.sort_values(
            by=['sort_order_action', 'sort_order_advice', 'RSI'],
            ascending=[True, True, True] # 优先按行动信号、其次按投资建议、最后按RSI从低到高排序
        ).drop(columns=['sort_order_action', 'sort_order_advice'])

        # 将浮点数格式化为字符串，方便Markdown输出
        report_df['最新净值'] = report_df['最新净值'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
        report_df['RSI'] = report_df['RSI'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
        report_df['净值/MA50'] = report_df['净值/MA50'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")

        # 将上述排序后的 DataFrame 转换为 Markdown
        markdown_table = report_df.to_markdown(index=False)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 市场情绪与技术指标监控报告\n\n")
            f.write(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## 推荐基金技术指标 (处理基金数: {len(self.fund_codes)})\n")
            f.write("此表格已按**行动信号优先级**排序，'强买入'基金将排在最前面。\n")
            f.write("**注意：** 当'行动信号'和'投资建议'冲突时，请以**行动信号**为准，其条件更严格，更适合机械化决策。\n\n")
            f.write(markdown_table)
        
        logger.info("报告生成完成: %s", self.output_file)

if __name__ == "__main__":
    try:
        logger.info("脚本启动")
        monitor = MarketMonitor()
        monitor.get_fund_data()
        monitor.generate_report()
        logger.info("脚本执行完成")
    except Exception as e:
        logger.error("脚本运行失败: %s", e)
        raise
