# -*- coding: UTF-8 -*-

import os
import re
import random
import time as time_module
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tenacity import retry, stop_after_attempt, wait_fixed, after_log
from io import BytesIO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 常量设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FUND_DATA_DIR = os.path.join(BASE_DIR, 'fund_data')
REPORT_FILE = os.path.join(BASE_DIR, 'analysis_report.md')
HOLIDAYS_FILE = os.path.join(BASE_DIR, 'holidays.txt')
HOLIDAYS_URL = "http://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=1803&tab1PAGENO=1&tab1PAGECOUNT=50&tab1CATEGORY=1073&tab1KEYWORD=%E4%BC%91%E5%B8%82&tab1CURPAGE=1&random=0.291775794770026"

# 确保数据目录存在
os.makedirs(FUND_DATA_DIR, exist_ok=True)

class MarketMonitor:
    """
    一个完整的基金市场监控与技术分析系统。
    """
    def __init__(self, filter_mode='strong_buy', top_n=5, holdings=None):
        """
        初始化 MarketMonitor。

        Args:
            filter_mode (str): 过滤模式，可选 'strong_buy', 'low_rsi_buy' 等。
            top_n (int): 在报告中显示的前 N 个基金。
            holdings (list): 用户持仓基金代码列表。
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.fund_codes = self._get_fund_codes_from_report()
        self.sh_index_data = self._get_sh_index_data()
        self.filter_mode = filter_mode
        self.top_n = top_n
        self.holdings = holdings if holdings else []
        self.holidays = self._get_holidays()

    def _get_fund_codes_from_report(self):
        """
        从 analysis_report.md 中提取基金代码列表。
        """
        try:
            with open(REPORT_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            # 使用正则表达式匹配代码块中的基金代码
            matches = re.findall(r"```python\s*funds = \[(.*?)\]\s*```", content, re.DOTALL)
            if matches:
                codes = matches[0].replace("'", "").replace('"', "").replace(" ", "").split(',')
                logging.info(f"从报告中成功提取 {len(codes)} 个基金代码。")
                return [code for code in codes if code]
        except FileNotFoundError:
            logging.error(f"报告文件未找到：{REPORT_FILE}")
        return []

    def _get_holidays(self):
        """
        获取节假日信息，优先从本地文件加载，否则从网络爬取。
        """
        if os.path.exists(HOLIDAYS_FILE):
            with open(HOLIDAYS_FILE, 'r', encoding='utf-8') as f:
                return {line.strip() for line in f}
        try:
            response = self.session.get(HOLIDAYS_URL, timeout=10)
            
            try:
                # 尝试使用 openpyxl 读取，如果失败则抛出 ImportError
                df = pd.read_excel(BytesIO(response.content), engine='openpyxl')
            except ImportError:
                logging.error("依赖库 openpyxl 未安装，无法处理 xlsx 文件。请运行 'pip install openpyxl' 安装。")
                return set()
            
            holidays_col = None
            # 查找包含“节假日”的列
            for col in df.columns:
                if '节假日' in str(col):
                    holidays_col = col
                    break

            if not holidays_col:
                logging.error("无法在节假日文件中找到 '节假日' 列，请检查数据源。")
                return set()

            holidays = set(pd.to_datetime(df[holidays_col]).dt.strftime('%Y-%m-%d'))
            with open(HOLIDAYS_FILE, 'w', encoding='utf-8') as f:
                for h in holidays:
                    f.write(f"{h}\n")
            logging.info(f"成功获取并缓存了 {len(holidays)} 个节假日。")
            return holidays
        except Exception as e:
            logging.error(f"无法获取节假日信息：{e}")
            return set()

    def _get_sh_index_data(self):
        """
        获取沪深300指数数据。
        """
        try:
            url = "http://push2.eastmoney.com/api/qt/stock/kline/get?cb=jQuery112404095400977226164_1625463137537&secid=1.000300&ut=fa5fd1943c7112009228b3f17d721a71&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=1&end=20500101&lmt=120"
            response = self.session.get(url, timeout=10)
            match = re.search(r'\(({.*?})\)', response.text)
            
            if not match:
                raise ValueError("无法解析指数数据。")
            
            parsed_data = eval(match.group(1))
            if not parsed_data['data'] or not parsed_data['data']['klines']:
                logging.warning("大盘数据API返回数据为空。")
                return pd.DataFrame()
            
            data = pd.DataFrame(parsed_data['data']['klines']).iloc[::-1]
            data.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'amplitude', 'change_percent', 'change_amount']
            data['date'] = pd.to_datetime(data['date']).dt.date
            data['change_percent'] = pd.to_numeric(data['change_percent'])
            return data
        except Exception as e:
            logging.error(f"获取大盘数据失败：{e}")
            return pd.DataFrame()

    def _get_market_trend(self):
        """
        根据最近交易日的大盘涨跌情况判断市场趋势。
        """
        if self.sh_index_data.empty:
            return '未知'
        
        today = date.today()
        # 确保数据是按日期排序的，并选择最新的交易日
        latest_trading_date_df = self.sh_index_data[self.sh_index_data['date'] <= today].iloc[-1]
        
        change_percent = latest_trading_date_df['change_percent']
        trend_date = latest_trading_date_df['date'].strftime('%Y-%m-%d')
        
        if change_percent > 1.5:
            trend = "强势"
        elif change_percent > 0:
            trend = "温和"
        elif change_percent < -1.5:
            trend = "弱势"
        else:
            trend = "震荡"
        
        logging.info(f"大盘趋势 ({trend_date}): {trend} ({change_percent:.2f}%)")
        return trend

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), after=after_log(logging.getLogger(__name__), logging.WARNING))
    def _fetch_fund_data(self, fund_code, page_index=1):
        """
        从东财 API 爬取单页基金历史净值数据。
        """
        url = f"http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={fund_code}&page={page_index}&per=20"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # 使用正则表达式提取包含数据的字符串
            match = re.search(r'content:"(.*?)",records', response.text, re.DOTALL)
            if not match:
                logging.warning(f"基金 {fund_code} API返回内容为空或格式不正确")
                return None, 0, 0
            
            html_content = match.group(1).replace('\\"', '"').replace('\\n', '')
            
            # 使用 pandas 解析 HTML 表格
            tables = pd.read_html(html_content)
            if not tables or tables[0].empty:
                return None, 0, 0
                
            df = tables[0]
            
            # 动态处理不同列数的情况
            num_cols = df.shape[1]
            if num_cols == 7:
                # 正常七列（股票/混合基金）
                df.columns = ['date', 'net_value', 'accum_net_value', 'daily_growth', 'purchase_status', 'redemption_status', 'dividend_info']
                df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
                logging.info(f"基金 {fund_code} API返回7列数据")
            elif num_cols == 6:
                # 修正：处理货币基金返回的6列数据
                # 头部是 净值日期, 每万份收益, 7日年化收益率（%）, 申购状态, 赎回状态, 分红送配
                # 这里将 '每万份收益' 作为净值来计算指标
                df.columns = ['date', 'yield_per_10k', 'annualized_yield_7d', 'purchase_status', 'redemption_status', 'dividend_info']
                df['net_value'] = pd.to_numeric(df['yield_per_10k'], errors='coerce')
                df['accum_net_value'] = np.nan # 累积净值不存在，设置为NaN
                df['daily_growth'] = np.nan # 日增长率不存在，设置为NaN
                logging.warning(f"基金 {fund_code} API返回6列数据，已将 '每万份收益' 作为净值处理。")
            else:
                logging.error(f"基金 {fund_code} 返回了未知列数 ({num_cols}) 的数据，跳过。")
                return None, 0, 0
                
            # 提取总记录数和总页数
            records = int(re.search(r'records:(\d+)', response.text).group(1))
            pages = int(re.search(r'pages:(\d+)', response.text).group(1))
            
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            return df, records, pages
        except Exception as e:
            logging.error(f"基金 {fund_code} 在第 {page_index} 页爬取失败: {e}")
            return None, 0, 0

    def _read_local_data(self, fund_code):
        """
        读取本地缓存的基金数据。
        """
        filepath = os.path.join(FUND_DATA_DIR, f"{fund_code}.csv")
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, parse_dates=['date'])
                df['date'] = df['date'].dt.date
                logging.info(f"成功读取基金 {fund_code} 的本地缓存数据。")
                return df
            except Exception as e:
                logging.warning(f"读取基金 {fund_code} 本地缓存文件失败: {e}")
        return pd.DataFrame()

    def _save_to_local_file(self, df, fund_code):
        """
        保存基金数据到本地文件。
        """
        filepath = os.path.join(FUND_DATA_DIR, f"{fund_code}.csv")
        try:
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logging.info(f"成功将基金 {fund_code} 数据保存到本地。")
        except Exception as e:
            logging.error(f"保存基金 {fund_code} 数据到本地失败: {e}")

    def _calculate_indicators(self, df):
        """
        计算基金的技术指标。
        """
        df = df.sort_values(by='date')
        df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
        
        # MACD
        df['ema12'] = df['net_value'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['net_value'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['signal']
        
        # RSI
        delta = df['net_value'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 布林带 (BBands)
        df['ma20'] = df['net_value'].rolling(window=20).mean()
        df['std20'] = df['net_value'].rolling(window=20).std()
        df['upper_band'] = df['ma20'] + (df['std20'] * 2)
        df['lower_band'] = df['ma20'] - (df['std20'] * 2)
        
        # MA50
        df['ma50'] = df['net_value'].rolling(window=50).mean()
        df['ma50_ratio'] = df['net_value'] / df['ma50']
        
        return df

    def _get_latest_signals(self, df, fund_code):
        """
        生成最新的技术信号。
        """
        if df.shape[0] < 50:
            return "数据不足"
        
        latest_data = df.iloc[-1]
        
        # 修正MACD金叉/死叉判断逻辑
        if len(df) >= 2:
            yesterday_diff = df.iloc[-2]['macd_diff']
            today_diff = latest_data['macd_diff']
            macd_signal = "无"
            if today_diff > 0 and yesterday_diff <= 0:
                macd_signal = "金叉(买入)"
            elif today_diff < 0 and yesterday_diff >= 0:
                macd_signal = "死叉(卖出)"
        else:
            macd_signal = "数据不足"
        
        rsi_signal = "中性"
        if latest_data['rsi'] < 30:
            rsi_signal = "超卖(买入)"
        elif latest_data['rsi'] > 70:
            rsi_signal = "超买(卖出)"
            
        bbands_signal = "中性"
        if latest_data['net_value'] < latest_data['lower_band']:
            bbands_signal = "跌破下轨(买入)"
        elif latest_data['net_value'] > latest_data['upper_band']:
            bbands_signal = "突破上轨(卖出)"
            
        ma50_signal = "中性"
        if latest_data['ma50_ratio'] < 0.95:
            ma50_signal = "大幅低于MA50(买入)"
        elif latest_data['ma50_ratio'] > 1.05:
            ma50_signal = "大幅高于MA50(卖出)"
            
        signals = {
            "代码": fund_code,
            "日期": latest_data['date'],
            "净值": round(latest_data['net_value'], 4),
            "净值日期": latest_data['date'],
            "MACD信号": macd_signal,
            "RSI信号": rsi_signal,
            "布林带信号": bbands_signal,
            "MA50信号": ma50_signal,
        }
        
        # 综合信号
        strong_buy = (
            (macd_signal == "金叉(买入)") and
            (rsi_signal == "超卖(买入)")
        )
        low_rsi_buy = (
            (rsi_signal == "超卖(买入)") and
            (bbands_signal == "跌破下轨(买入)")
        )
        
        signals['行动信号'] = '无'
        if strong_buy:
            signals['行动信号'] = '强烈买入'
        elif low_rsi_buy:
            signals['行动信号'] = '买入'
        
        return signals

    def _process_single_fund(self, fund_code):
        """
        处理单个基金的数据：读取本地、增量更新、计算指标并生成信号。
        """
        logging.info(f"--- 正在处理基金 {fund_code} ---")
        
        local_df = self._read_local_data(fund_code)
        
        # 获取最新日期以进行增量更新
        start_date = local_df['date'].max() if not local_df.empty else date(2000, 1, 1)
        latest_data_date = date.today()
        
        # 检查是否已是最新，并考虑节假日
        if not local_df.empty and local_df['date'].max() == latest_data_date and str(latest_data_date) not in self.holidays:
            logging.info(f"基金 {fund_code} 数据已是最新，跳过下载。")
            df = local_df
        else:
            # 增量下载新数据
            new_df = pd.DataFrame()
            page_index = 1
            total_pages = 1
            
            while page_index <= total_pages:
                time_module.sleep(random.uniform(0.5, 1.5))
                logging.info(f"正在获取基金 {fund_code} 的第 {page_index} 页数据...")
                temp_df, total_records, total_pages = self._fetch_fund_data(fund_code, page_index)
                
                if temp_df is None or temp_df.empty:
                    logging.warning(f"获取基金 {fund_code} 数据时 API 未返回内容。")
                    break
                    
                # 合并数据
                if new_df.empty:
                    new_df = temp_df
                else:
                    new_df = pd.concat([new_df, temp_df], ignore_index=True)
                
                # 检查是否已达到本地最新日期，如果已达到则停止下载
                if not local_df.empty and (new_df['date'] <= start_date).any():
                    logging.info(f"已下载至本地最新数据，停止爬取。")
                    break
                
                page_index += 1
            
            # 合并本地和新数据
            if not new_df.empty:
                df = pd.concat([local_df, new_df], ignore_index=True)
                df.drop_duplicates(subset=['date'], keep='last', inplace=True)
                df.sort_values(by='date', inplace=True)
                self._save_to_local_file(df, fund_code)
            else:
                df = local_df

        if df.empty or df.shape[0] < 50:
            logging.warning(f"基金 {fund_code} 数据量不足，无法进行技术分析。")
            return None
        
        processed_df = self._calculate_indicators(df)
        signals = self._get_latest_signals(processed_df, fund_code)
        
        # 额外添加MACD和RSI值到结果中
        latest_data = processed_df.iloc[-1]
        signals['RSI'] = round(latest_data['rsi'], 2)
        signals['MACD'] = round(latest_data['macd'], 4)
        signals['Signal'] = round(latest_data['signal'], 4)
        
        return signals

    def get_fund_data(self):
        """
        多线程获取所有基金数据并生成分析信号。
        """
        fund_signals = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_fund = {executor.submit(self._process_single_fund, code): code for code in self.fund_codes}
            for future in as_completed(future_to_fund):
                fund_code = future_to_fund[future]
                try:
                    result = future.result()
                    if result:
                        fund_signals.append(result)
                except Exception as exc:
                    logging.error(f"处理基金 {fund_code} 时发生异常: {exc}")
        
        return fund_signals

    def generate_report(self):
        """
        生成 Markdown 格式的投资建议报告。
        """
        all_signals = self.get_fund_data()
        
        # 根据过滤模式筛选和排序
        if self.filter_mode == 'strong_buy':
            filtered_signals = [s for s in all_signals if s['行动信号'] == '强烈买入']
        elif self.filter_mode == 'low_rsi_buy':
            filtered_signals = [s for s in all_signals if s['RSI'] < 30]
        else:
            filtered_signals = [s for s in all_signals if s['行动信号'] != '无']

        # 优先显示持仓基金
        holding_signals = [s for s in filtered_signals if s['代码'] in self.holdings]
        other_signals = [s for s in filtered_signals if s['代码'] not in self.holdings]
        
        # 根据RSI或MACD进行排序
        sorted_signals = sorted(holding_signals, key=lambda x: x['RSI']) + sorted(other_signals, key=lambda x: x['RSI'])
        
        # 获取大盘趋势
        market_trend = self._get_market_trend()
        
        # 生成报告内容
        report_content = f"# 基金市场技术分析报告\n\n"
        report_content += f"**生成日期**: {date.today().strftime('%Y-%m-%d')}\n"
        report_content += f"**大盘趋势（沪深300）**: {market_trend}\n\n"
        report_content += f"## 投资建议\n\n"
        report_content += f"以下是根据 `{self.filter_mode}` 模式筛选出的，且结合大盘趋势的建议。\n\n"
        
        if market_trend == "强势":
            report_content += "💡 **市场情绪积极，可适当关注技术买入信号。**\n\n"
        elif market_trend == "弱势":
            report_content += "⚠️ **市场情绪谨慎，技术信号强度降低，建议观望或小额试探。**\n\n"
        
        if sorted_signals:
            report_content += "| 基金代码 | 基金净值 | 净值日期 | 行动信号 | MACD信号 | RSI信号 | 布林带信号 | MA50信号 |\n"
            report_content += "|---|---|---|---|---|---|---|---|\n"
            for s in sorted_signals[:self.top_n]:
                report_content += f"| {s['代码']} | {s['净值']} | {s['净值日期']} | **{s['行动信号']}** | {s['MACD信号']} | {s['RSI']} | {s['布林带信号']} | {s['MA50信号']} |\n"
        else:
            report_content += "目前没有符合条件的基金。\n"
        
        # 写入报告文件
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logging.info(f"报告已生成至 {REPORT_FILE}")
        return report_content

if __name__ == "__main__":
    monitor = MarketMonitor(holdings=['000013', '161725'])
    monitor.generate_report()
