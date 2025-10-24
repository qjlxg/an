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
        # 保持原有 __init__ 签名，尽管 report_file 已在 _parse_report 中被固定为 C类.txt
        self.report_file = report_file
        self.output_file = output_file
        self.fund_codes = []
        self.fund_data = {}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
        }

    def _get_expected_latest_date(self):
        """根据当前时间确定期望的最新数据日期 (保留原有逻辑)"""
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
        """
        修改: 从 C类.txt 提取基金代码。
        """
        code_file = 'C类.txt'
        logger.info("正在解析 %s 获取基金代码...", code_file)
        if not os.path.exists(code_file):
            logger.error("代码文件 %s 不存在", code_file)
            raise FileNotFoundError(f"{code_file} 不存在")

        try:
            with open(code_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 匹配连续的6位数字作为基金代码，忽略标题行如 'code'
            pattern = re.compile(r'^\s*(\d{6})\s*$', re.M)
            matches = pattern.findall(content)

            extracted_codes = set(matches)

            sorted_codes = sorted(list(extracted_codes))
            self.fund_codes = sorted_codes[:1000] # 限制数量

            if not self.fund_codes:
                logger.warning("未提取到任何有效基金代码，请检查 %s", code_file)
            else:
                logger.info("提取到 %d 个基金: %s", len(self.fund_codes), self.fund_codes[:min(len(self.fund_codes), 10)])

        except Exception as e:
            logger.error("解析代码文件失败: %s", e)
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
        """
        修改: 实现新的数据更新逻辑。
        1. 获取本地最新日期。
        2. 获取网站最新日期。
        3. 如果日期相同，则跳过爬取。
        4. 如果日期不同，则开始增量爬取。
        """
        local_df = self._read_local_data(fund_code)
        latest_local_date = local_df['date'].max().date() if not local_df.empty else None
        
        # --- 新数据更新逻辑: 对比网站和本地日期 ---
        # 1. 尝试获取网站最新数据日期 (只获取第一条数据)
        url_check = f"http://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page=1&per=1"
        logger.info("基金 %s 正在对比本地日期 (%s) 和网站最新日期...", fund_code, latest_local_date if latest_local_date else '无')
        site_latest_date = None

        response = requests.get(url_check, headers=self.headers, timeout=15)
        response.raise_for_status()
        
        content_match = re.search(r'content:"(.*?)"', response.text, re.S)
        
        if content_match:
            raw_content_html = content_match.group(1).replace('\\"', '"')
            tables = pd.read_html(StringIO(raw_content_html))
            
            if tables and not tables[0].empty and len(tables[0].columns) >= 1:
                # 假设日期在第一列
                site_latest_date_str = tables[0].iloc[0, 0] 
                site_latest_date = pd.to_datetime(site_latest_date_str, errors='coerce').date()
                logger.info("基金 %s 网站最新日期: %s", fund_code, site_latest_date)

        # 2. 对比日期，决定是否跳过更新
        if latest_local_date and site_latest_date and site_latest_date <= latest_local_date:
            logger.info("基金 %s 数据已是最新或更新 (本地: %s, 网站: %s)，跳过网络获取。", 
                        fund_code, latest_local_date, site_latest_date)
            # 返回本地数据
            return local_df.tail(100)[['date', 'net_value']]
        
        if site_latest_date is None:
            logger.warning("基金 %s 无法获取网站最新日期或网站无数据，将尝试增量爬取。", fund_code)

        logger.info("基金 %s 开始网络爬取 (本地数据需要更新或本地无数据)...", fund_code)
        # --- 结束新数据更新逻辑 ---

        # 增量爬取循环开始
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

                if not raw_content_html.strip():
                    logger.warning("基金 %s 在第 %d 页返回内容为空，爬取结束", fund_code, page_index)
                    break
                    
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
                    # 只获取比本地最新日期严格更新的数据
                    new_df = df[df['date'].dt.date > latest_local_date]
                    
                    if not new_df.empty:
                        all_new_data.append(new_df)
                        logger.info("第 %d 页: 发现 %d 行新数据", page_index, len(new_df))
                    
                    # 如果当前页的数据中，最新的日期已经小于或等于本地最新日期，则说明增量更新完成
                    if new_df.empty and (df['date'].dt.date.max() <= latest_local_date):
                         logger.info("基金 %s 已抓取到本地最新数据，增量爬取结束", fund_code)
                         break

                else:
                    # 如果本地没有数据，则获取所有历史数据
                    all_new_data.append(df)
                    if len(df) < 20: # 爬取到最后一页
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
        """计算技术指标并生成结果字典 (保留原有逻辑)"""
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

            # 计算 MA50
            df['MA50'] = df['net_value'].rolling(window=50, min_periods=1).mean()
            ma_ratio = (df['net_value'].iloc[-1] / df['MA50'].iloc[-1]) if not df['MA50'].iloc[-1] == 0 else np.nan

            latest_net_value = df['net_value'].iloc[-1]
            latest_rsi = rsi.iloc[-1]
            macd_diff = df['macd'].iloc[-1] - df['signal'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            
            # --- 投资建议逻辑 --- (保留原有逻辑)
            advice = "观察"
            action_signal = "N/A"
            
            # 动量信号 (MACD / RSI)
            if macd_diff > 0 and latest_rsi < 70:
                advice = "看涨"
            elif macd_diff < 0 and latest_rsi > 30:
                advice = "看跌"
            
            # 交易信号 (布林带 / RSI)
            if latest_net_value < bb_lower and latest_rsi < 30:
                action_signal = "买入"
            elif latest_net_value > bb_upper and latest_rsi > 70:
                action_signal = "卖出"
            elif latest_rsi < 35:
                 action_signal = "关注买入"
            elif latest_rsi > 65:
                 action_signal = "关注卖出"
            
            return {
                'fund_code': fund_code,
                'latest_net_value': latest_net_value,
                'rsi': latest_rsi,
                'ma_ratio': ma_ratio,
                'macd_diff': macd_diff,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'advice': advice,
                'action_signal': action_signal
            }
        except Exception as e:
            logger.error("计算基金 %s 指标失败: %s", fund_code, e)
            return {
                'fund_code': fund_code, 'latest_net_value': "计算失败", 'rsi': np.nan, 'ma_ratio': np.nan,
                'macd_diff': np.nan, 'bb_upper': np.nan, 'bb_lower': np.nan, 'advice': "观察", 'action_signal': 'N/A'
            }

    def run(self):
        """主执行函数 (保留原有逻辑)"""
        start_time = time_module.time()
        
        try:
            # 1. 获取基金代码 (现在从 C类.txt 读取)
            self._parse_report()
            if not self.fund_codes:
                logger.error("未找到任何基金代码，脚本终止。")
                return

            # 2. 并发获取数据和计算指标
            logger.info("开始并发处理 %d 个基金...", len(self.fund_codes))
            results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # 提交数据抓取任务
                future_to_code = {executor.submit(self._fetch_fund_data, code): code for code in self.fund_codes}
                
                # 等待抓取任务完成
                for future in concurrent.futures.as_completed(future_to_code):
                    fund_code = future_to_code[future]
                    try:
                        df = future.result()
                        # 提交指标计算任务
                        results.append(self._calculate_indicators(fund_code, df))
                    except Exception as e:
                        logger.error("基金 %s 数据获取/处理失败: %s", fund_code, e)
                        # 如果失败，添加一个失败的条目到结果中
                        results.append({
                            'fund_code': fund_code, 'latest_net_value': "数据获取失败", 'rsi': np.nan, 'ma_ratio': np.nan,
                            'macd_diff': np.nan, 'bb_upper': np.nan, 'bb_lower': np.nan, 'advice': "观察", 'action_signal': 'N/A'
                        })

            # 3. 生成报告
            self._generate_report(results)
            
        except Exception as e:
            logger.critical("市场监控脚本运行中发生致命错误: %s", e)
        
        end_time = time_module.time()
        logger.info("脚本执行完毕，总耗时: %.2f 秒", end_time - start_time)


    def _generate_report(self, results):
        """生成 Markdown 格式的报告文件 (保留原有逻辑)"""
        logger.info("开始生成市场监控报告...")
        
        # 转换结果列表为 DataFrame
        report_df = pd.DataFrame(results)
        
        if report_df.empty:
            logger.warning("结果集为空，无法生成报告。")
            return

        # 重命名和选择列
        report_df = report_df.rename(columns={
            'fund_code': '基金代码',
            'latest_net_value': '最新净值',
            'rsi': 'RSI',
            'ma_ratio': '净值/MA50',
            'advice': '投资建议',
            'action_signal': '行动信号',
            'macd_diff': 'MACD差值',
            'bb_upper': '布林上轨',
            'bb_lower': '布林下轨'
        })
        
        report_df = report_df[['基金代码', '最新净值', 'RSI', '净值/MA50', '投资建议', '行动信号', 'MACD差值', '布林上轨', '布林下轨']]

        # 添加排序辅助列
        action_order = {'买入': 1, '关注买入': 2, '观察': 3, '关注卖出': 4, '卖出': 5, 'N/A': 6}
        advice_order = {'看涨': 1, '观察': 2, '看跌': 3}
        
        report_df['sort_order_action'] = report_df['行动信号'].map(action_order).fillna(6)
        report_df['sort_order_advice'] = report_df['投资建议'].map(advice_order).fillna(4)

        # 按照您的新排序规则进行排序
        report_df = report_df.sort_values(
            by=['sort_order_action', 'sort_order_advice', 'RSI'],
            ascending=[True, True, True] # 优先按行动信号、其次按投资建议、最后按RSI从低到高排序
        ).drop(columns=['sort_order_action', 'sort_order_advice'])

        # 将浮点数格式化为字符串，方便Markdown输出
        # 使用 apply 结合 lambda 确保对 NaN 值的处理
        report_df['最新净值'] = report_df['最新净值'].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) and not pd.isna(x) else ("数据获取失败" if x == "数据获取失败" else "N/A"))
        report_df['RSI'] = report_df['RSI'].apply(lambda x: f"{x:.2f}" if isinstance(x, (float, np.floating)) and not pd.isna(x) else "N/A")
        report_df['净值/MA50'] = report_df['净值/MA50'].apply(lambda x: f"{x:.2f}" if isinstance(x, (float, np.floating)) and not pd.isna(x) else "N/A")
        report_df['MACD差值'] = report_df['MACD差值'].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) and not pd.isna(x) else "N/A")
        report_df['布林上轨'] = report_df['布林上轨'].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) and not pd.isna(x) else "N/A")
        report_df['布林下轨'] = report_df['布林下轨'].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) and not pd.isna(x) else "N/A")

        # 将上述排序后的 DataFrame 转换为 Markdown
        markdown_table = report_df.to_markdown(index=False)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 市场情绪与技术指标监控报告\n\n")
            f.write(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## 推荐基金技术指标 (处理基金数: {len(self.fund_codes)} / 有效数据数: {len(report_df.dropna(subset=['最新净值']))})\n\n")
            f.write(markdown_table)
            f.write("\n\n---\n\n")
            f.write("### 指标说明\n")
            f.write("* **行动信号**：基于布林带和RSI的即时交易信号。\n")
            f.write("* **投资建议**：基于MACD的趋势判断。\n")
            f.write("* **RSI**：相对强弱指数，衡量超买超卖，<30为超卖，>70为超买。\n")
            f.write("* **净值/MA50**：最新净值与50日均线比值，>1表明短期走势强于中期均值。\n")


if __name__ == '__main__':
    # 请确保 C类.txt 文件和 fund_data 目录在脚本同一级目录下
    monitor = MarketMonitor()
    monitor.run()
