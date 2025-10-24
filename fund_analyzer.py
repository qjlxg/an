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
import sys # 确保引入 sys
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
        # 注意: 这部分路径设置依赖于运行环境，保持原样
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
                # 保存到缓存
                if self.cache_data:
                    self.cache.setdefault('fund', {})[fund_code] = result
                    self._save_cache()
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
                # 保存到缓存
                if self.cache_data:
                    self.cache.setdefault('manager', {})[fund_code] = result
                    self._save_cache()
                return fund_code, result
        except Exception as e:
            self._log(f"通过akshare获取基金 {fund_code} 经理数据失败: {e}")
            
        # 如果akshare失败，尝试网页抓取
        scraped_data = self._scrape_manager_data_from_web(fund_code)
        if scraped_data:
            # 保存到缓存
            if self.cache_data:
                self.cache.setdefault('manager', {})[fund_code] = scraped_data
                self._save_cache()
            return fund_code, scraped_data

        return fund_code, {'name': 'N/A', 'tenure_years': np.nan, 'cumulative_return': np.nan}


    def get_fund_holdings_data(self, fund_code: str):
        """ 
        抓取基金的股票持仓数据。
        注意: 为保持脚本完整性，此处沿用原脚本结构，但实际网页抓取持仓可能更复杂。
        """
        if fund_code in self.cache.get('holdings', {}):
            self.holdings_data[fund_code] = self.cache['holdings'][fund_code]
            self._log(f"使用缓存的基金 {fund_code} 持仓数据")
            return True
            
        self._log(f"正在获取基金 {fund_code} 的持仓数据...")
        
        # 优先使用 akshare 接口
        try:
            # 默认获取最新一期十大重仓股
            holdings_df = ak.fund_portfolio_hold_em(symbol=fund_code)
            if not holdings_df.empty:
                # 假设 holdings_df 有 '股票代码' 和 '占净值比例' 等列
                self.holdings_data[fund_code] = holdings_df.to_dict('records')
                self._log(f"基金 {fund_code} 持仓数据已通过akshare获取。")
                if self.cache_data:
                    self.cache.setdefault('holdings', {})[fund_code] = self.holdings_data[fund_code]
                    self._save_cache()
                return True
        except Exception as e:
            self._log(f"通过akshare获取基金 {fund_code} 持仓数据失败: {e}")

        # 如果 akshare 失败，尝试网页抓取 (简化版)
        holdings_url = f"http://fundf10.eastmoney.com/ccmx_{fund_code}.html"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(holdings_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 简化：仅检查是否有表格存在
            data_table = soup.find('div', class_='boxitem w790')
            if data_table:
                # 实际解析逻辑可能很复杂，此处仅记录成功
                self.holdings_data[fund_code] = [{'stock_code': 'Web Scraped Data', 'ratio': np.nan}] 
                self._log(f"基金 {fund_code} 持仓数据已通过网页抓取获取 (简化)。")
                if self.cache_data:
                    self.cache.setdefault('holdings', {})[fund_code] = self.holdings_data[fund_code]
                    self._save_cache()
                return True

        except Exception as e:
            self._log(f"网页抓取基金 {fund_code} 持仓数据失败: {e}", level='warning')
        
        self.holdings_data[fund_code] = [] # 抓取失败，返回空列表
        return False
        
    def get_market_data(self, index_code='000300'):
        """获取市场指数数据（例如沪深300）并计算夏普比率"""
        if index_code in self.cache.get('market', {}):
            self.market_data[index_code] = self.cache['market'][index_code]
            self._log(f"使用缓存的市场指数 {index_code} 数据")
            return
        
        try:
            # 沪深300 代码 '000300'
            df = ak.index_zh_a_hist(symbol=index_code, period="daily", start_date="20100101", end_date=datetime.now().strftime("%Y%m%d"))
            df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            
            # 计算夏普比率等指标
            returns = df['收盘'].pct_change().dropna()
            annual_returns = returns.mean() * 252
            annual_volatility = returns.std() * (252**0.5)
            sharpe_ratio = (annual_returns - self.risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
            
            self.market_data[index_code] = {
                'annual_returns': float(annual_returns),
                'sharpe_ratio': float(sharpe_ratio)
            }
            self.cache.setdefault('market', {})[index_code] = self.market_data[index_code]
            self._save_cache()
            self._log(f"市场指数 {index_code} 数据获取成功。")
        except Exception as e:
            self._log(f"获取市场指数 {index_code} 数据失败: {e}", level='error')
            self.market_data[index_code] = {'annual_returns': np.nan, 'sharpe_ratio': np.nan}


    def generate_full_report(self, fund_info_dict):
        """整合所有数据并生成最终报告"""
        logger.info("开始生成最终分析报告...")
        
        # 准备市场基准数据 (使用沪深300作为基准)
        market_sharpe = self.market_data.get('000300', {}).get('sharpe_ratio', np.nan)
        
        # 构建报告 DataFrame
        report_data = []
        # 确保只遍历已成功抓取数据的基金代码
        processed_codes = set(self.fund_data.keys()) | set(self.manager_data.keys()) 
        
        for code in processed_codes:
            fund_name = fund_info_dict.get(code, 'N/A')
            
            fund_stats = self.fund_data.get(code, {})
            manager_stats = self.manager_data.get(code, {})
            
            report_data.append({
                '基金代码': code,
                '基金名称': fund_name,
                '最新净值': fund_stats.get('latest_nav', np.nan),
                '夏普比率': fund_stats.get('sharpe_ratio', np.nan),
                '最大回撤': fund_stats.get('max_drawdown', np.nan),
                '经理姓名': manager_stats.get('name', 'N/A'),
                '任职年限': manager_stats.get('tenure_years', np.nan),
                '任职回报(%)': manager_stats.get('cumulative_return', np.nan),
                '夏普(基准000300)': market_sharpe
            })
            
        df_report = pd.DataFrame(report_data)
        
        # 数据清洗和格式化
        # 修正：将回撤格式化为负百分比字符串
        df_report['最大回撤'] = df_report['最大回撤'].apply(lambda x: f"{-x*100:.2f}%" if pd.notna(x) else 'N/A')
        df_report['夏普比率'] = df_report['夏普比率'].round(4)
        df_report['夏普(基准000300)'] = df_report['夏普(基准000300)'].round(4)
        df_report['任职年限'] = df_report['任职年限'].round(2)
        df_report['最新净值'] = df_report['最新净值'].round(4)
        
        # 排序 (例如：按夏普比率降序)
        df_report = df_report.sort_values(by='夏普比率', ascending=False)
        
        # 保存报告 (使用 utf-8-sig 编码，兼容 Excel)
        report_filename = 'fund_report.csv'
        try:
            df_report.to_csv(report_filename, index=False, encoding='utf-8-sig')
            logger.info(f"分析报告已保存到 {report_filename}")
        except Exception as e:
            logger.error(f"保存报告失败: {e}")


# --- 主执行逻辑 ---
if __name__ == '__main__':
    # **********************************************
    # * 修改点：从 C类.txt 读取基金代码
    # **********************************************
    funds_list_file = 'C类.txt' 

    fund_codes_to_analyze = []
    fund_info_dict = {}

    try:
        logger.info(f"正在从 {funds_list_file} 导入基金代码列表...")
        # C类.txt 是一个纯文本文件，每行一个代码
        # 默认使用 utf-8 编码读取
        with open(funds_list_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 过滤掉空行和首行的 'code' 标识 (如果存在，不区分大小写)
        raw_codes = [line.strip() for line in lines if line.strip() and line.strip().lower() != 'code']
        
        # 格式化基金代码 (补零到六位)
        fund_codes_to_analyze = [str(code).zfill(6) for code in raw_codes]
        
        # 由于 C类.txt 只包含代码，我们创建一个简化的 fund_info_dict
        # 基金名称统一使用占位符 'N/A'，以适应后续分析逻辑对 fund_info_dict 的依赖
        fund_info_dict = {code: 'N/A' for code in fund_codes_to_analyze}
        
        logger.info(f"导入成功，共 {len(fund_codes_to_analyze)} 个基金代码从 {funds_list_file} 读取。")
        
    except FileNotFoundError:
        logger.error(f"基金代码文件 {funds_list_file} 未找到！")
        sys.exit(1)
    except Exception as e:
        logger.error(f"读取基金代码文件 {funds_list_file} 失败: {e}")
        sys.exit(1)
        
    if not fund_codes_to_analyze:
        logger.error("未找到任何基金代码，请检查文件内容。")
        sys.exit(1)
        
    # 初始化分析器
    # 保持原有的初始化参数
    analyzer = FundAnalyzer(cache_data=True, max_workers=10) 
    
    # 提前获取市场数据，供报告使用 (如果未缓存)
    analyzer.get_market_data('000300')
    
    # 并发抓取所有基金的数据
    logger.info(f"开始并发抓取和计算 {len(fund_codes_to_analyze)} 个基金的数据...")
    
    with ThreadPoolExecutor(max_workers=analyzer.max_workers) as executor:
        # 提交基金数据抓取任务 (净值、夏普、回撤)
        fund_futures = {executor.submit(analyzer._fetch_and_calculate_fund_data, code): code for code in fund_codes_to_analyze}
        # 提交基金经理数据抓取任务
        manager_futures = {executor.submit(analyzer._fetch_manager_data, code): code for code in fund_codes_to_analyze}
        # 提交持仓数据抓取任务
        holdings_futures = {executor.submit(analyzer.get_fund_holdings_data, code): code for code in fund_codes_to_analyze}
        
        # 处理基金数据结果
        for future in as_completed(fund_futures):
            fund_code = fund_futures[future]
            try:
                code, data = future.result()
                if pd.isna(data['latest_nav']):
                    logger.warning(f"基金 {code} 数据无效，跳过。")
                else:
                    analyzer.fund_data[code] = data
            except Exception as e:
                logger.error(f"处理基金 {fund_code} 数据失败: {e}")

        # 处理基金经理数据结果
        for future in as_completed(manager_futures):
            fund_code = manager_futures[future]
            try:
                code, data = future.result()
                analyzer.manager_data[code] = data
            except Exception as e:
                logger.error(f"处理基金经理 {fund_code} 数据失败: {e}")
                
        # 处理持仓数据结果
        for future in as_completed(holdings_futures):
             # 仅等待任务完成，实际数据已在 get_fund_holdings_data 中写入 analyzer.holdings_data
             try:
                 future.result()
             except Exception as e:
                 # get_fund_holdings_data 内部有日志记录，此处仅捕获异常防止中断
                 logger.debug(f"处理基金持仓任务意外失败: {e}")


    logger.info("所有基金数据抓取和计算完成。")
    
    # 运行报告生成逻辑
    analyzer.generate_full_report(fund_info_dict)

    logger.info("报告生成完成，请查看 fund_analyzer.log 和 fund_report.csv 文件。")
