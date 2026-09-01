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
        update_time = time(21, 0)
        if now.time() < update_time:
            expected_date = now.date() - timedelta(days=1)
        else:
            expected_date = now.date()
        logger.info("当前时间: %s, 期望最新数据日期: %s", now.strftime('%Y-%m-%d %H:%M:%S'), expected_date)
        return expected_date

    def _parse_report(self):
        """从 C类.txt 提取基金代码"""
        code_file = 'C类.txt'
        logger.info("正在解析 %s 获取基金代码...", code_file)
        if not os.path.exists(code_file):
            logger.error("代码文件 %s 不存在", code_file)
            raise FileNotFoundError(f"{code_file} 不存在")

        try:
            with open(code_file, 'r', encoding='utf-8') as f:
                content = f.read()

            pattern = re.compile(r'^\s*(\d{6})\s*$', re.M)
            matches = pattern.findall(content)
            extracted_codes = set(matches)
            sorted_codes = sorted(list(extracted_codes))
            self.fund_codes = sorted_codes[:1700]

            if not self.fund_codes:
                logger.warning("未提取到任何有效基金代码，请检查 %s", code_file)
            else:
                logger.info("提取到 %d 个基金: %s", len(self.fund_codes), self.fund_codes[:min(len(self.fund_codes), 10)])
        except Exception as e:
            logger.error("解析代码文件失败: %s", e)
            raise

    def _read_local_data(self, fund_code):
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
        local_df = self._read_local_data(fund_code)
        latest_local_date = local_df['date'].max().date() if not local_df.empty else None
        
        # 对比网站最新日期
        url_check = f"http://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page=1&per=1"
        logger.info("基金 %s 正在对比本地日期 (%s) 和网站最新日期...", fund_code, latest_local_date if latest_local_date else '无')
        site_latest_date = None

        try:
            response = requests.get(url_check, headers=self.headers, timeout=15)
            response.raise_for_status()
            content_match = re.search(r'content:"(.*?)"', response.text, re.S)
            if content_match:
                raw_content_html = content_match.group(1).replace('\\"', '"')
                tables = pd.read_html(StringIO(raw_content_html))
                if tables and not tables[0].empty and len(tables[0].columns) >= 1:
                    site_latest_date_str = tables[0].iloc[0, 0]
                    site_latest_date = pd.to_datetime(site_latest_date_str, errors='coerce').date()
                    logger.info("基金 %s 网站最新日期: %s", fund_code, site_latest_date)
        except Exception as e:
            logger.warning("获取网站最新日期失败: %s", e)

        if latest_local_date and site_latest_date and site_latest_date <= latest_local_date:
            logger.info("基金 %s 数据已是最新，跳过网络获取。", fund_code)
            # --- MODIFICATION 1: Increased data limit for calculation ---
            return local_df.tail(500)[['date', 'net_value']]

        logger.info("基金 %s 开始网络爬取...", fund_code)
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
                    logger.error("基金 %s API返回内容格式不正确", fund_code)
                    break

                raw_content_html = content_match.group(1).replace('\\"', '"')
                total_pages = int(pages_match.group(1))

                if not raw_content_html.strip():
                    logger.warning("基金 %s 在第 %d 页返回内容为空", fund_code, page_index)
                    break
                    
                tables = pd.read_html(StringIO(raw_content_html))
                if not tables:
                    logger.warning("基金 %s 在第 %d 页未找到数据表格", fund_code, page_index)
                    break
                
                df = tables[0]
                df.columns = ['date', 'net_value', 'cumulative_net_value', 'daily_growth_rate', 'purchase_status', 'redemption_status', 'dividend']
                df = df[['date', 'net_value']].copy()
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
                df = df.dropna(subset=['date', 'net_value'])
                
                if latest_local_date:
                    new_df = df[df['date'].dt.date > latest_local_date]
                    if not new_df.empty:
                        all_new_data.append(new_df)
                    if new_df.empty and (df['date'].dt.date.max() <= latest_local_date):
                        logger.info("基金 %s 已抓取到本地最新数据，增量爬取结束", fund_code)
                        break
                else:
                    all_new_data.append(df)
                    if len(df) < 20:
                        break

                logger.info("基金 %s 总页数: %d, 当前页: %d, 当前页行数: %d", fund_code, total_pages, page_index, len(df))
                if page_index >= total_pages:
                    logger.info("基金 %s 已获取所有历史数据，共 %d 页", fund_code, total_pages)
                    break
                
                page_index += 1
                time_module.sleep(random.uniform(1, 2))
                
            except Exception as e:
                logger.error("基金 %s API数据解析失败: %s", fund_code, str(e))
                raise

        if all_new_data:
            new_combined_df = pd.concat(all_new_data, ignore_index=True)
            df_final = pd.concat([local_df, new_combined_df]).drop_duplicates(subset=['date'], keep='last').sort_values(by='date', ascending=True)
            self._save_to_local_file(fund_code, df_final)
            # --- MODIFICATION 2: Increased data limit for calculation ---
            df_final = df_final.tail(500)
            logger.info("成功合并并保存基金 %s 的数据，总行数: %d, 最新日期: %s", fund_code, len(df_final), df_final['date'].iloc[-1].strftime('%Y-%m-%d'))
            return df_final[['date', 'net_value']]
        else:
            if not local_df.empty:
                # --- MODIFICATION 3: Increased data limit for calculation ---
                return local_df.tail(500)[['date', 'net_value']]
            else:
                raise ValueError("未获取到任何有效数据")

    def _calculate_indicators(self, fund_code, df):
        try:
            if df is None or df.empty or len(df) < 26:
                logger.warning("基金 %s 数据不足，跳过计算", fund_code)
                return {
                    'fund_code': fund_code, 'latest_net_value': "数据不足", 'rsi': np.nan, 'ma_ratio': np.nan,
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

            df['MA50'] = df['net_value'].rolling(window=50, min_periods=1).mean()
            ma_ratio = (df['net_value'].iloc[-1] / df['MA50'].iloc[-1]) if not df['MA50'].iloc[-1] == 0 else np.nan

            latest_net_value = df['net_value'].iloc[-1]
            latest_rsi = rsi.iloc[-1]
            macd_diff = df['macd'].iloc[-1] - df['signal'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            
            advice = "观察"
            action_signal = "N/A"
            if macd_diff > 0 and latest_rsi < 70:
                advice = "看涨"
            elif macd_diff < 0 and latest_rsi > 30:
                advice = "看跌"
            
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

    def filter_funds(self, results):
        """四层筛选 + 综合评分"""
        df = pd.DataFrame(results)

        action_priority = {'买入': 5, '关注买入': 4, '观察': 3, '关注卖出': 2, '卖出': 1, 'N/A': 0}
        df['action_score'] = df['action_signal'].map(action_priority)

        advice_priority = {'看涨': 3, '观察': 2, '看跌': 1}
        df['advice_score'] = df['advice'].map(advice_priority)

        df['valid'] = True
        df.loc[df['rsi'] >= 40, 'valid'] = False
        df['near_lower'] = (df['latest_net_value'] <= df['bb_lower'] * 1.05)
        df.loc[~df['near_lower'], 'valid'] = False
        df.loc[~(df['macd_diff'] > 0), 'valid'] = False
        df.loc[df['ma_ratio'] < 0.95, 'valid'] = False

        df['composite_score'] = 0
        df['composite_score'] += df['action_score'] * 15
        df['composite_score'] += df['advice_score'] * 10
        df['rsi_score'] = (40 - df['rsi'].clip(upper=40)) / 40 * 25
        df['composite_score'] += df['rsi_score']
        df['bb_distance'] = (df['bb_lower'] - df['latest_net_value']) / df['bb_lower']
        df['bb_score'] = df['bb_distance'].clip(lower=0) * 100
        df['composite_score'] += df['bb_score'].clip(upper=20)
        df['ma_score'] = (df['ma_ratio'] - 0.95) / 0.15 * 15
        df['composite_score'] += df['ma_score'].clip(lower=0, upper=15)

        filtered = df[df['valid']].copy()
        filtered = filtered.sort_values('composite_score', ascending=False)
        top_picks = filtered.head(30).copy()
        top_picks['rank'] = range(1, len(top_picks) + 1)

        return top_picks[['fund_code', 'latest_net_value', 'rsi', 'ma_ratio',
                          'action_signal', 'advice', 'macd_diff', 'bb_lower',
                          'composite_score', 'rank']]

    def run(self):
        start_time = time_module.time()
        try:
            self._parse_report()
            if not self.fund_codes:
                logger.error("未找到任何基金代码，脚本终止。")
                return

            logger.info("开始并发处理 %d 个基金...", len(self.fund_codes))
            results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_code = {executor.submit(self._fetch_fund_data, code): code for code in self.fund_codes}
                for future in concurrent.futures.as_completed(future_to_code):
                    fund_code = future_to_code[future]
                    try:
                        df = future.result()
                        results.append(self._calculate_indicators(fund_code, df))
                    except Exception as e:
                        logger.error("基金 %s 处理失败: %s", fund_code, e)
                        results.append({
                            'fund_code': fund_code, 'latest_net_value': "失败", 'rsi': np.nan, 'ma_ratio': np.nan,
                            'macd_diff': np.nan, 'bb_upper': np.nan, 'bb_lower': np.nan, 'advice': "观察", 'action_signal': 'N/A'
                        })

            self._generate_report(results)
            
        except Exception as e:
            logger.critical("脚本运行失败: %s", e)
        
        end_time = time_module.time()
        logger.info("脚本执行完毕，总耗时: %.2f 秒", end_time - start_time)

    def _generate_report(self, results):
        logger.info("开始生成报告...")
        df = pd.DataFrame(results)
        top_picks = self.filter_funds(results)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 市场情绪与技术指标监控报告\n\n")
            f.write(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 重点推荐
            if not top_picks.empty:
                f.write(f"## 重点推荐买入基金（共 {len(top_picks)} 只）\n\n")
                pick_table = top_picks.rename(columns={
                    'fund_code': '基金代码', 'latest_net_value': '最新净值', 'rsi': 'RSI',
                    'ma_ratio': '净值/MA50', 'action_signal': '行动信号', 'advice': '投资建议',
                    'bb_lower': '布林下轨', 'composite_score': '综合评分', 'rank': '排名'
                })
                pick_table = pick_table[['排名', '基金代码', '最新净值', 'RSI', '净值/MA50', '行动信号', '投资建议', '布林下轨', '综合评分']]
                for col in ['最新净值', 'RSI', '净值/MA50', '布林下轨', '综合评分']:
                    pick_table[col] = pick_table[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                f.write(pick_table.to_markdown(index=False))
                f.write("\n\n---\n\n")
            else:
                f.write("## 重点推荐买入基金\n\n暂无符合条件的基金。\n\n---\n\n")

            # 全量表格
            report_df = df.rename(columns={
                'fund_code': '基金代码', 'latest_net_value': '最新净值', 'rsi': 'RSI',
                'ma_ratio': '净值/MA50', 'advice': '投资建议', 'action_signal': '行动信号',
                'macd_diff': 'MACD差值', 'bb_upper': '布林上轨', 'bb_lower': '布林下轨'
            })
            report_df = report_df[['基金代码', '最新净值', 'RSI', '净值/MA50', '投资建议', '行动信号', 'MACD差值', '布林上轨', '布林下轨']]

            action_order = {'买入': 1, '关注买入': 2, '观察': 3, '关注卖出': 4, '卖出': 5, 'N/A': 6}
            advice_order = {'看涨': 1, '观察': 2, '看跌': 3}
            report_df['sort_action'] = report_df['行动信号'].map(action_order).fillna(6)
            report_df['sort_advice'] = report_df['投资建议'].map(advice_order).fillna(4)
            report_df = report_df.sort_values(by=['sort_action', 'sort_advice', 'RSI'], ascending=[True, True, True])
            report_df = report_df.drop(columns=['sort_action', 'sort_advice'])

            for col in ['最新净值', 'RSI', '净值/MA50', 'MACD差值', '布林上轨', '布林下轨']:
                report_df[col] = report_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) and not pd.isna(x) else ("失败" if "失败" in str(x) else "N/A"))

            f.write(f"## 全部基金技术指标 (处理: {len(self.fund_codes)} / 有效: {len(report_df.dropna(subset=['最新净值']))})\n\n")
            f.write(report_df.to_markdown(index=False))
            f.write("\n\n---\n\n")
            f.write("### 指标说明\n")
            f.write("* **行动信号**：布林带+RSI即时交易信号\n")
            f.write("* **投资建议**：MACD趋势判断\n")
            f.write("* **综合评分**：0~100，>70 可重仓\n")

if __name__ == '__main__':
    monitor = MarketMonitor()
    monitor.run()
