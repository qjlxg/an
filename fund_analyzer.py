import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import numpy as np
import time
import requests
from bs4 import BeautifulSoup
import re
import json
import os
import logging
import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
# 导入并发执行库
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志记录 (使用 utf-8-sig，与 CSV 文件保持一致性)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # 使用 utf-8-sig 编码写入日志文件
        logging.FileHandler('fund_analyzer.log', encoding='utf-8-sig'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FundAnalyzer')

# --- 辅助类：SeleniumFetcher ---
class SeleniumFetcher:
    """
    使用 Selenium 模拟浏览器进行数据抓取。
    """
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 无头模式，不显示浏览器窗口
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        # 从环境变量获取路径
        chrome_options.binary_location = os.getenv('CHROME_BINARY_PATH', '/usr/bin/chromium-browser')
        service = ChromeService(executable_path=os.getenv('CHROMEDRIVER_PATH', '/usr/bin/chromedriver'))
        try:
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        except WebDriverException as e:
            logger.error(f"Selenium WebDriver 初始化失败: {e}")
            self.driver = None

    def get_page_source(self, url, wait_for_element=None, timeout=30):
        if not self.driver:
            return None
        try:
            self.driver.get(url)
            if wait_for_element:
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_element))
                )
            return self.driver.page_source
        except (TimeoutException, WebDriverException) as e:
            logger.error(f"Selenium 抓取失败: {e}")
            return None

    def __del__(self):
        # 避免在 __del__ 中调用 quit()
        pass

# --- 核心分析类：FundAnalyzer ---

class FundAnalyzer:
    """
    一个用于自动化分析中国公募基金的类。
    """
    def __init__(self, risk_free_rate=0.01858, cache_file='fund_cache.json', cache_data=True, max_workers=10):
        self.fund_data = {}
        self.manager_data = {}
        self.holdings_data = {}
        self.market_data = {}
        self.report_data = []
        self.cache_file = cache_file
        self.cache_data = cache_data
        self.cache = self._load_cache()
        self.risk_free_rate = risk_free_rate
        self._selenium_fetcher = None
        # 新增最大工作线程数 (用于并发)
        self.max_workers = max_workers

    @property
    def selenium_fetcher(self):
        if self._selenium_fetcher is None:
            self._selenium_fetcher = SeleniumFetcher()
        return self._selenium_fetcher
        
    def _log(self, message, level='info'):
        """统一的日志记录方法"""
        if level == 'info':
            logger.info(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'error':
            logger.error(message)

    def _load_cache(self):
        """从文件加载缓存数据 (使用 utf-8 读取)"""
        if self.cache_data and os.path.exists(self.cache_file):
            try:
                # 缓存文件通常用 utf-8 保存
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                   self._log("缓存文件 fund_cache.json 损坏，正在重新创建。", level='warning')
                   return {}
        return {}

    def _save_cache(self):
        """将缓存数据保存到文件 (使用 utf-8 保存)"""
        if self.cache_data:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=4)

    # 封装核心抓取逻辑为独立方法，便于多线程调用
    def _fetch_and_calculate_fund_data(self, fund_code: str):
        """
        获取基金的单位净值和累计净值数据，并计算夏普比率和最大回撤。
        """
        if fund_code in self.cache.get('fund', {}):
            return fund_code, self.cache['fund'][fund_code]

        for attempt in range(3):
            try:
                fund_data = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
                fund_data['净值日期'] = pd.to_datetime(fund_data['净值日期'])
                fund_data.set_index('净值日期', inplace=True)
                
                fund_data = fund_data.dropna()
                if len(fund_data) < 252:
                    raise ValueError("数据不足，无法计算可靠的夏普比率和回撤")

                returns = fund_data['单位净值'].pct_change().dropna()
                
                annual_returns = returns.mean() * 252
                annual_volatility = returns.std() * (252**0.5)
                sharpe_ratio = (annual_returns - self.risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
                
                rolling_max = fund_data['单位净值'].cummax()
                daily_drawdown = (fund_data['单位净值'] - rolling_max) / rolling_max
                max_drawdown = daily_drawdown.min() * -1
                
                result = {
                    'latest_nav': float(fund_data['单位净值'].iloc[-1]),
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': float(max_drawdown)
                }
                return fund_code, result
            except Exception as e:
                self._log(f"获取基金 {fund_code} 数据失败 (尝试 {attempt+1}/3): {e}")
                time.sleep(1) # 缩短等待时间

        return fund_code, {'latest_nav': np.nan, 'sharpe_ratio': np.nan, 'max_drawdown': np.nan}


    def _scrape_manager_data_from_web(self, fund_code: str) -> dict:
        """从天天基金网通过网页抓取获取基金经理数据"""
        manager_url = f"http://fundf10.eastmoney.com/jjjl_{fund_code}.html"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(manager_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            title_label = soup.find('label', string='基金经理变动一览')
            if not title_label:
                return None
            
            manager_table = title_label.find_parent().find_next_sibling('table')
            if not manager_table:
                return None
            
            rows = manager_table.find_all('tr')
            if len(rows) < 2:
                return None
            
            latest_manager_row = rows[1]
            cols = latest_manager_row.find_all('td')
            
            if len(cols) < 5:
                return None
            
            manager_name = cols[2].text.strip()
            tenure_str = cols[3].text.strip()
            cumulative_return_str = cols[4].text.strip()
            
            tenure_days = np.nan
            if '年又' in tenure_str:
                tenure_parts = tenure_str.split('年又')
                years_match = re.search(r'\d+', tenure_parts[0])
                days_match = re.search(r'\d+', tenure_parts[1])
                years = float(years_match.group()) if years_match else 0
                days = float(days_match.group()) if days_match else 0
                tenure_days = years * 365 + days
            elif '天' in tenure_str:
                days_match = re.search(r'\d+', tenure_str)
                tenure_days = float(days_match.group()) if days_match else np.nan
            elif '年' in tenure_str:
                years_match = re.search(r'\d+', tenure_str)
                tenure_days = float(years_match.group()) * 365 if years_match else np.nan
            else:
                tenure_days = np.nan
                
            cumulative_return = float(re.search(r'[-+]?\d*\.?\d+', cumulative_return_str).group()) if '%' in cumulative_return_str else np.nan

            return {
                'name': manager_name,
                'tenure_years': float(tenure_days) / 365.0 if pd.notna(tenure_days) else np.nan,
                'cumulative_return': cumulative_return
            }
        except requests.exceptions.RequestException as e:
            self._log(f"网页抓取基金 {fund_code} 经理数据失败: {e}", level='warning')
            return None
        except Exception as e:
            self._log(f"解析网页内容失败: {e}", level='warning')
            return None


    def _fetch_manager_data(self, fund_code: str):
        """
        获取基金经理数据（首先尝试使用 akshare，失败则通过网页抓取）
        """
        if fund_code in self.cache.get('manager', {}):
            return fund_code, self.cache['manager'][fund_code]

        try:
            manager_info = ak.fund_manager_em(symbol=fund_code)
            if not manager_info.empty:
                latest_manager = manager_info.sort_values(by='上任日期', ascending=False).iloc[0]
                name = latest_manager.get('姓名', 'N/A')
                tenure_days = latest_manager.get('任职天数', np.nan)
                cumulative_return = latest_manager.get('任职回报', '0%')
                cumulative_return = float(str(cumulative_return).replace('%', '')) if isinstance(cumulative_return, str) else float(cumulative_return)
                
                result = {
                    'name': name,
                    'tenure_years': float(tenure_days) / 365.0 if pd.notna(tenure_days) else np.nan,
                    'cumulative_return': cumulative_return
                }
                return fund_code, result
        except Exception as e:
            # 如果akshare失败，尝试网页抓取
            scraped_data = self._scrape_manager_data_from_web(fund_code)
            if scraped_data:
                return fund_code, scraped_data

        return fund_code, {'name': 'N/A', 'tenure_years': np.nan, 'cumulative_return': np.nan}

    
    def get_fund_holdings_data(self, fund_code: str):
        """
        抓取基金的股票持仓数据。
        """
        if fund_code in self.cache.get('holdings', {}):
            self.holdings_data[fund_code] = self.cache['holdings'][fund_code]
            self._log(f"使用缓存的基金 {fund_code} 持仓数据")
            return True

        self._log(f"正在获取基金 {fund_code} 的持仓数据...")
        
        # 优先使用 akshare 接口
        try:
            holdings_df = ak.fund_portfolio_hold_em(symbol=fund_code)
            if not holdings_df.empty:
                # 确保只保留需要的字段，并处理NaN
                holdings_df = holdings_df[['股票代码', '股票名称', '占净值比例', '持仓市值(万元)']].rename(
                    columns={'持仓市值(万元)': '持仓市值（万元）'} # 统一列名
                ).fillna(np.nan)

                self.holdings_data[fund_code] = holdings_df.to_dict('records')
                self._log(f"基金 {fund_code} 持仓数据已通过akshare获取。")
                if self.cache_data:
                    self.cache.setdefault('holdings', {})[fund_code] = self.holdings_data[fund_code]
                    self._save_cache()
                return True
        except Exception as e:
            self._log(f"通过akshare获取基金 {fund_code} 持仓数据失败: {e}")

        # 如果 akshare 失败，尝试网页抓取
        holdings_url = f"http://fundf10.eastmoney.com/ccmx_{fund_code}.html"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(holdings_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            holdings_header = soup.find('h4', string=lambda t: t and '股票投资明细' in t)
            if not holdings_header:
                raise ValueError("未找到持仓表格标题。")
            
            holdings_table = holdings_header.find_next('table')
            if not holdings_table:
                raise ValueError("未找到持仓表格。")
            
            rows = holdings_table.find_all('tr')[1:] # 跳过表头
            
            holdings = []
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 7: # 确保列数正确
                    # 注意：这里假设网页抓取的列索引是固定的
                    ratio_text = cols[4].text.strip().replace('%', '')
                    market_value_text = cols[6].text.strip().replace(',', '')
                    
                    # 尝试转换，失败则为 NaN
                    ratio = float(ratio_text) if re.match(r'^-?\d+(\.\d+)?$', ratio_text) else np.nan
                    market_value = float(market_value_text) if re.match(r'^-?\d+(\.\d+)?$', market_value_text) else np.nan

                    holdings.append({
                        '股票代码': cols[1].text.strip(),
                        '股票名称': cols[2].text.strip(),
                        '占净值比例': ratio,
                        '持仓市值（万元）': market_value,
                    })
            self.holdings_data[fund_code] = holdings
            self._log(f"基金 {fund_code} 持仓数据已通过网页抓取获取。")
            if self.cache_data:
                self.cache.setdefault('holdings', {})[fund_code] = self.holdings_data[fund_code]
                self._save_cache()
            return True
        except Exception as e:
            self._log(f"获取基金 {fund_code} 持仓数据失败: {e}")
            self.holdings_data[fund_code] = []
            return False

    def get_market_sentiment(self):
        """获取市场情绪（仅调用一次，基于上证指数）"""
        if self.market_data:
            self._log("使用缓存的市场情绪数据")
            return True
        self._log("正在获取市场情绪数据...")
        try:
            index_data = ak.stock_zh_index_daily_em(symbol="sh000001")
            index_data['date'] = pd.to_datetime(index_data['date'])
            last_week_data = index_data.iloc[-7:]
            
            if last_week_data.empty or len(last_week_data) < 2:
                raise ValueError("指数数据不足")
            
            price_change = last_week_data['close'].iloc[-1] / last_week_data['close'].iloc[0] - 1
            
            # 计算前6天的平均成交量，以确保有足够的历史数据
            if len(last_week_data) >= 7:
                 volume_change = last_week_data['volume'].iloc[-1] / last_week_data['volume'].iloc[:-1].mean() - 1
            else:
                 volume_change = 0 # 无法计算，设为中性
            
            if price_change > 0.01 and volume_change > 0:
                sentiment, trend = 'optimistic', 'bullish'
            elif price_change < -0.01:
                sentiment, trend = 'pessimistic', 'bearish'
            else:
                sentiment, trend = 'neutral', 'neutral'
            
            self.market_data = {'sentiment': sentiment, 'trend': trend}
            self._log(f"市场情绪数据已获取：{self.market_data}")
            return True
        except Exception as e:
            self._log(f"获取市场数据失败: {e}")
            self.market_data = {'sentiment': 'unknown', 'trend': 'unknown'}
            return False

    def _evaluate_fund(self, fund_code, fund_name, fund_type):
        """
        评估单个基金的综合分数。
        """
        self._log(f"--- 正在分析基金 {fund_code} ---")
        
        # 检查并发获取的数据是否有效
        if fund_code not in self.fund_data or pd.isna(self.fund_data[fund_code].get('sharpe_ratio')):
            self._log(f"基金 {fund_code} 基本信息获取失败或数据不足，跳过分析。")
            self.report_data.append({'fund_code': fund_code, 'fund_name': fund_name, 'decision': 'Skip', 'score': np.nan})
            return
            
        # 获取持仓数据 (串行获取，因为它只影响最终评分，且是 IO 密集型)
        self.get_fund_holdings_data(fund_code)

        # 评分体系
        scores = {}
        values = {}
        
        # 1. 夏普比率评分 (最高 10 分)
        sharpe_ratio = self.fund_data[fund_code].get('sharpe_ratio')
        if pd.notna(sharpe_ratio):
            # 夏普率 1.0 满分，每 0.1 分得 1 分
            scores['sharpe_ratio_score'] = min(10, max(0, int(sharpe_ratio * 10))) 
            values['sharpe_ratio_value'] = sharpe_ratio
        else:
            scores['sharpe_ratio_score'] = 0
            values['sharpe_ratio_value'] = np.nan

        # 2. 最大回撤评分 (最高 10 分)
        max_drawdown = self.fund_data[fund_code].get('max_drawdown')
        if pd.notna(max_drawdown):
            # 回撤 10% 以下满分 (10 - 回撤*100)
            scores['max_drawdown_score'] = min(10, max(0, 10 - int(max_drawdown * 100 / 10))) 
            values['max_drawdown_value'] = max_drawdown
        else:
            scores['max_drawdown_score'] = 0
            values['max_drawdown_value'] = np.nan
            
        # 3. 基金经理任职年限评分 (最高 10 分)
        manager_years = self.manager_data[fund_code].get('tenure_years')
        if pd.notna(manager_years):
            scores['manager_years_score'] = min(10, int(manager_years * 2.5)) # 4年满分
        else:
            scores['manager_years_score'] = 0
        values['manager_years_value'] = manager_years
        
        # 4. 基金经理任职回报评分 (最高 10 分)
        manager_return = self.manager_data[fund_code].get('cumulative_return')
        if pd.notna(manager_return):
            # 每 10% 回报得 1 分，最高 10 分
            scores['manager_return_score'] = min(10, max(0, int(manager_return / 10)))
        else:
            scores['manager_return_score'] = 0
        values['manager_return_value'] = manager_return
            
        # 5. 持仓集中度评分 (最高 10 分)
        if self.holdings_data.get(fund_code):
            holdings_df = pd.DataFrame(self.holdings_data[fund_code])
            # 确保 '占净值比例' 字段存在且有效
            if '占净值比例' in holdings_df.columns and not holdings_df['占净值比例'].empty:
                 # 清除 NaN 值后再求和
                top_10_holdings_ratio = holdings_df['占净值比例'].iloc[:10].sum(skipna=True)
            else:
                top_10_holdings_ratio = np.nan

            if pd.notna(top_10_holdings_ratio):
                # 集中度 50% 以下给高分，集中度越低分数越高 (10 - (比例 - 30)/3)
                scores['holding_concentration_score'] = min(10, max(0, int(10 - (top_10_holdings_ratio - 30) / 5)))
                values['holding_concentration_value'] = top_10_holdings_ratio
            else:
                scores['holding_concentration_score'] = 0
                values['holding_concentration_value'] = np.nan
        else:
            scores['holding_concentration_score'] = 0
            values['holding_concentration_value'] = np.nan
        
        # 6. 基金类型/市场情绪调整 (最高 10 分)
        # 假设我们只分析股票/混合型，故基础分高
        scores['fund_type_score'] = 5 if '股票型' in fund_type or '混合型' in fund_type else 0 
        
        # 市场情绪：牛市气氛且基金表现优秀，额外加分 (最高 5 分)
        market_adj = 0
        if self.market_data.get('trend') == 'bullish' and scores.get('sharpe_ratio_score', 0) > 5:
             market_adj = 5
        scores['market_sentiment_adj_score'] = market_adj
        
        total_score = sum(scores.values())
        
        # 决策逻辑
        decision = '推荐' if total_score > 35 else '观望' # 调整推荐阈值
        
        self.report_data.append({
            'fund_code': fund_code,
            'fund_name': fund_name,
            'fund_type': fund_type, # 增加基金类型
            'decision': decision,
            'score': total_score,
            'scores_details': scores,
            'values_details': values
        })
        self._log(f"评分详情: {scores}")
        self._log(f"基金 {fund_code} 评估完成，总分: {total_score:.2f}，决策: {decision}")

    def _save_report_to_markdown(self):
        """将分析报告保存为 Markdown 文件"""
        if not self.report_data:
            return
        
        report_path = "analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("--- 批量基金分析报告 ---\n\n")
            f.write(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"市场情绪: **{self.market_data.get('sentiment', 'N/A').upper()}** ({self.market_data.get('trend', 'N/A')})\n\n")
            f.write("--- 汇总结果 ---\n\n")
            
            results_df = pd.DataFrame(self.report_data)
            valid_results = results_df[results_df['decision'] != 'Skip']
            
            if not valid_results.empty:
                f.write("### 推荐基金 (分数 > 35)\n\n")
                recommended = valid_results[valid_results['decision'] == '推荐'].sort_values(by='score', ascending=False)
                if not recommended.empty:
                    # 格式化输出的 DataFrame
                    recommended_output = recommended[['fund_code', 'fund_name', 'score', 'fund_type']].copy()
                    recommended_output['score'] = recommended_output['score'].round(2)
                    f.write(recommended_output.to_markdown(index=False) + "\n\n")
                else:
                    f.write("无\n\n")
                    
                f.write("### 观望基金\n\n")
                watchlist = valid_results[valid_results['decision'] == '观望'].sort_values(by='score', ascending=False)
                if not watchlist.empty:
                    watchlist_output = watchlist[['fund_code', 'fund_name', 'score', 'fund_type']].copy()
                    watchlist_output['score'] = watchlist_output['score'].round(2)
                    f.write(watchlist_output.to_markdown(index=False) + "\n\n")
                else:
                    f.write("无\n\n")
            
            f.write("--- 详细分析 ---\n\n")
            for item in self.report_data:
                f.write(f"### 基金 {item['fund_code']} - {item.get('fund_name', 'N/A')}\n")
                f.write(f"- 类型: {item.get('fund_type', 'N/A')}\n")
                f.write(f"- 最终决策: **{item['decision']}**\n")
                f.write(f"- 综合分数: **{item['score']:.2f}**\n")
                
                if item['decision'] != 'Skip':
                    f.write("- **关键指标值**:\n")
                    # 使用格式化后的值
                    values_formatted = {
                        '最新净值': f"{item['values_details'].get('latest_nav', np.nan):.4f}" if pd.notna(item['values_details'].get('latest_nav')) else 'N/A',
                        '夏普比率': f"{item['values_details'].get('sharpe_ratio_value', np.nan):.2f}" if pd.notna(item['values_details'].get('sharpe_ratio_value')) else 'N/A',
                        '最大回撤': f"{item['values_details'].get('max_drawdown_value', np.nan) * 100:.2f}%" if pd.notna(item['values_details'].get('max_drawdown_value')) else 'N/A',
                        '经理年限': f"{item['values_details'].get('manager_years_value', np.nan):.1f}年" if pd.notna(item['values_details'].get('manager_years_value')) else 'N/A',
                        '经理回报': f"{item['values_details'].get('cumulative_return_value', np.nan):.2f}%" if pd.notna(item['values_details'].get('cumulative_return_value')) else 'N/A',
                        '前十集中度': f"{item['values_details'].get('holding_concentration_value', np.nan):.2f}%" if pd.notna(item['values_details'].get('holding_concentration_value')) else 'N/A'
                    }
                    for k, v in values_formatted.items():
                         f.write(f"  - {k}: {v}\n")
                         
                    f.write("- **细项得分**:\n")
                    score_mapping = {
                        'sharpe_ratio_score': '夏普比率',
                        'max_drawdown_score': '最大回撤',
                        'manager_years_score': '经理年限',
                        'manager_return_score': '经理回报',
                        'holding_concentration_score': '持仓集中度',
                        'fund_type_score': '基金类型',
                        'market_sentiment_adj_score': '市场情绪调整'
                    }
                    for k, v in item.get('scores_details', {}).items():
                        f.write(f"  - {score_mapping.get(k, k)}: {v}\n")
                        
                f.write("\n---\n\n")


    def run_analysis(self, fund_codes: list, fund_info: dict):
        """
        运行批量基金分析的主函数，引入并发处理。
        """
        self._log("--- 批量基金分析启动 ---")
        
        self.get_market_sentiment()
        
        start_time = time.time()
        
        # 1. 使用并发线程池获取所有基金的基本数据和经理数据
        self._log(f"开始使用 {self.max_workers} 个线程并发获取数据...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交基金基本数据和夏普率计算任务
            fund_futures = {executor.submit(self._fetch_and_calculate_fund_data, code): code for code in fund_codes}
            # 提交基金经理数据获取任务
            manager_futures = {executor.submit(self._fetch_manager_data, code): code for code in fund_codes}

            # 处理基本数据的结果
            for future in as_completed(fund_futures):
                code = fund_futures[future]
                try:
                    code, data = future.result()
                    self.fund_data[code] = data
                    self.cache.setdefault('fund', {})[code] = data # 填充缓存
                except Exception as e:
                    self._log(f"并发获取基金 {code} 基本数据失败: {e}", level='error')
                    self.fund_data[code] = {'latest_nav': np.nan, 'sharpe_ratio': np.nan, 'max_drawdown': np.nan}
            
            # 处理经理数据的结果
            for future in as_completed(manager_futures):
                code = manager_futures[future]
                try:
                    code, data = future.result()
                    self.manager_data[code] = data
                    self.cache.setdefault('manager', {})[code] = data # 填充缓存
                except Exception as e:
                    self._log(f"并发获取基金 {code} 经理数据失败: {e}", level='error')
                    self.manager_data[code] = {'name': 'N/A', 'tenure_years': np.nan, 'cumulative_return': np.nan}
        
        # 统一保存缓存
        self._save_cache()
        self._log(f"并发数据获取完成，耗时: {time.time() - start_time:.2f} 秒")
        
        # 2. 串行评估（评估只涉及本地计算和持仓数据，持仓数据是 IO 密集型，串行可以减轻对 IO 的瞬间压力）
        self._log("开始串行评估和评分...")
        for code in fund_codes:
            # 假设 fund_info 字典包含了名称和类型，如果类型缺失，默认使用 '混合型'
            fund_name = fund_info.get(code, {}).get('name', 'N/A')
            fund_type = fund_info.get(code, {}).get('type', '混合型')
            self._evaluate_fund(code, fund_name, fund_type)
        
        end_time = time.time()
        self._log(f"全部评估完成，总耗时: {end_time - start_time:.2f} 秒")
        
        # 生成并保存最终报告
        results_df = pd.DataFrame(self.report_data)
        if not results_df.empty and 'decision' in results_df.columns:
            self._log("\n--- 全部基金分析结果 ---")
            self._log("\n" + results_df[['decision', 'score', 'fund_code', 'fund_name']].to_markdown(index=False))
        else:
            self._log("\n没有基金获得有效评分。")
        
        self._save_report_to_markdown()
        
        return results_df

# --- 脚本主入口 ---
if __name__ == '__main__':
    # *** 核心修改点：直接从本地文件读取 ***
    funds_list_path = 'recommended_cn_funds.csv'
    
    df_funds = None
    
    # 核心修复点：使用容错读取，解决 BOM 编码问题
    # 尝试多种编码读取本地文件
    possible_encodings = ['utf-8-sig', 'gbk', 'utf-8']
    
    if not os.path.exists(funds_list_path):
        logger.error(f"错误：本地文件 '{funds_list_path}' 不存在。请确保文件与脚本在同一目录下。")
        sys.exit(1)
        
    for encoding in possible_encodings:
        try:
            logger.info(f"正在从本地 CSV 文件 '{funds_list_path}' 导入基金代码列表，尝试编码: {encoding}...")
            # 读取本地 CSV 文件
            df_funds = pd.read_csv(funds_list_path, encoding=encoding)
            logger.info(f"导入成功，使用的编码是: {encoding}")
            break # 成功读取后跳出循环
        except Exception as e:
            # 将错误级别改为 warning，以便继续尝试其他编码
            logger.warning(f"使用 {encoding} 导入失败: {e}")
    
    # 检查是否成功读取以及列名是否正确
    if df_funds is not None and not df_funds.empty:
        # 规范化列名，确保存在 '代码' 和 '名称'
        df_funds.columns = [str(col).strip() for col in df_funds.columns]
        
        if '代码' not in df_funds.columns:
            # 尝试识别其他可能的代码列名
            if 'code' in df_funds.columns:
                df_funds = df_funds.rename(columns={'code': '代码'})
            else:
                 logger.error("CSV 文件中未找到 '代码' 或 'code' 列！请检查文件结构。")
                 sys.exit(1)
                 
        if '名称' not in df_funds.columns:
             # 尝试识别其他可能的名称列名
             if 'name' in df_funds.columns:
                 df_funds = df_funds.rename(columns={'name': '名称'})
             else:
                 logger.warning("CSV 文件中未找到 '名称' 或 'name' 列，将使用代码作为名称。")
                 df_funds['名称'] = df_funds['代码'] # 容错处理

        # 同样容错处理 '类型' 列
        if '类型' not in df_funds.columns and 'type' in df_funds.columns:
            df_funds = df_funds.rename(columns={'type': '类型'})
        elif '类型' not in df_funds.columns:
             df_funds['类型'] = '混合型' # 默认设置为混合型，以便评分逻辑能够运行

        fund_codes_to_analyze = [str(code).zfill(6) for code in df_funds['代码'].unique().tolist()]
        
        # 构建包含名称和类型的字典
        fund_info_dict = {}
        for code, name, f_type in zip(df_funds['代码'], df_funds['名称'], df_funds['类型']):
             fund_info_dict[str(code).zfill(6)] = {'name': name, 'type': f_type}
        
        logger.info(f"导入成功，共 {len(fund_codes_to_analyze)} 个基金代码")
        
        # 使用前 X 个基金进行分析 (可根据运行时间预算调整)
        max_funds = 700 
        test_fund_codes = fund_codes_to_analyze[:max_funds] 
        logger.info(f"分析前 {len(test_fund_codes)} 个基金...")
        
        # 实例化分析器，设置最大并发线程数
        analyzer = FundAnalyzer(max_workers=32)
        analyzer.run_analysis(test_fund_codes, fund_info_dict)
    else:
        logger.error("所有尝试的编码都无法成功导入基金列表，程序结束。")
        sys.exit(1)
