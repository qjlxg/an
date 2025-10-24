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
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fund_analyzer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FundAnalyzer')

# 定义历史数据存储目录
HISTORY_DATA_DIR = 'fund_history_data' 
# 确保历史数据目录存在
os.makedirs(HISTORY_DATA_DIR, exist_ok=True)

class SeleniumFetcher:
    """
    使用 Selenium 模拟浏览器进行数据抓取。（保留原有功能，但不再用于持仓抓取）
    """
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 无头模式，不显示浏览器窗口
        
        # --- 增强稳定性的参数 START ---
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions") # 禁用扩展
        chrome_options.add_argument("--log-level=3") # 仅输出致命错误
        chrome_options.add_argument("--window-size=1920,1080") # 设置窗口大小，避免某些页面布局问题
        chrome_options.add_argument("--remote-debugging-port=9222") # 有时有助于稳定连接
        # --- 增强稳定性的参数 END ---

        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        # 指定 Chromium 二进制路径和 ChromeDriver 路径
        chrome_options.binary_location = os.getenv('CHROME_BINARY_PATH', '/usr/bin/chromium-browser')
        service = ChromeService(executable_path=os.getenv('CHROMEDRIVER_PATH', '/usr/bin/chromedriver'))
        
        try:
            # 尝试使用环境变量中的 ChromeDriver
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
            # 记录更详细的错误信息
            logger.error(f"Selenium 抓取 {url} 失败: {e}", exc_info=True)
            return None

    def __del__(self):
        if self.driver:
            self.driver.quit()

class FundAnalyzer:
    """
    一个用于自动化分析中国公募基金的类。
    """
    def __init__(self, risk_free_rate=0.01858, cache_file='fund_cache.json', cache_data=True):
        self.fund_data = {}
        self.manager_data = {}
        self.holdings_data = {}
        self.market_data = {}
        self.report_data = []
        self.cache_file = cache_file
        self.cache_data = cache_data
        self.cache = self._load_cache()
        # 直接使用用户提供的无风险利率，不再进行抓取
        self.risk_free_rate = risk_free_rate
        self.selenium_fetcher = SeleniumFetcher()
        self.session = requests.Session() # NEW: 新增 requests Session 以提高抓取效率
        
        # 新增属性：用于存储持仓基金代码
        self.holding_codes = []
        # 新增属性：用于存储相对排名数据
        self.ranking_data = {}
        
        self._load_holdings_from_config() # 在初始化时加载持仓配置

    def _load_holdings_from_config(self, config_file='holdings_config.json'):
        """从 holdings_config.json 文件加载持仓基金代码"""
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 提取配置中的所有键作为持仓代码
                # 确保代码是6位字符串格式
                config_codes = []
                for code_str in config_data.keys():
                    # 尝试将键转换为6位基金代码，跳过非数字或长度不符的键
                    if str(code_str).isdigit() and len(str(code_str)) <= 6:
                        config_codes.append(str(code_str).zfill(6))
                
                self.holding_codes = sorted(list(set(config_codes)))
                self._log(f"从 {config_file} 加载了 {len(self.holding_codes)} 个持仓基金代码: {self.holding_codes}", level='info')
            except Exception as e:
                self._log(f"加载持仓配置文件 {config_file} 失败: {e}", level='error')
                self.holding_codes = []
        else:
            self._log(f"持仓配置文件 {config_file} 不存在。", level='warning')
            self.holding_codes = []

    def _log(self, message, level='info'):
        """统一的日志记录方法"""
        if level == 'info':
            logger.info(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'error':
            logger.error(message)

    def _load_cache(self):
        """从文件加载缓存数据 (层次二：分析结果缓存)"""
        if self.cache_data and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self._log(f"加载缓存文件 {self.cache_file} 失败: {e}", level='error')
                return {}
        return {}

    def _save_cache(self):
        """将缓存数据保存到文件 (层次二：分析结果缓存)"""
        if self.cache_data:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    # 将 NumPy 的 NaN 转换为 None/null，以便 JSON 序列化
                    def default_serializer(obj):
                        if pd.isna(obj):
                            return None
                        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

                    json.dump(self.cache, f, ensure_ascii=False, indent=4, default=default_serializer)
            except Exception as e:
                self._log(f"保存缓存文件 {self.cache_file} 失败: {e}", level='error')

    def _get_history_file_path(self, fund_code: str) -> str:
        """获取历史净值数据文件的完整路径"""
        return os.path.join(HISTORY_DATA_DIR, f"{fund_code}_history.csv")

    def _load_fund_history(self, fund_code: str) -> pd.DataFrame or None:
        """
        加载基金的历史净值数据 (层次一：历史净值数据缓存)
        """
        file_path = self._get_history_file_path(fund_code)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, parse_dates=['净值日期'])
                df.set_index('净值日期', inplace=True)
                self._log(f"基金 {fund_code} 历史数据从缓存文件 {file_path} 加载成功。")
                return df
            except Exception as e:
                self._log(f"加载基金 {fund_code} 历史数据失败: {e}", level='error')
                return None
        return None

    def _save_fund_history(self, fund_code: str, fund_data: pd.DataFrame):
        """
        保存基金的历史净值数据 (层次一：历史净值数据缓存)
        """
        file_path = self._get_history_file_path(fund_code)
        try:
            # 确保索引（净值日期）作为列保存
            fund_data.to_csv(file_path, encoding='utf-8')
            self._log(f"基金 {fund_code} 历史数据已保存到 {file_path}。")
        except Exception as e:
            self._log(f"保存基金 {fund_code} 历史数据失败: {e}", level='error')

    def _update_and_calculate(self, fund_code: str, history_df: pd.DataFrame or None) -> pd.DataFrame or None:
        """
        执行增量更新和指标计算。
        核心修复：由于 akshare 接口不再支持 start_date，改为获取全量数据，然后与本地数据合并去重。
        """
        
        last_date = None
        if history_df is not None and not history_df.empty:
            last_date = history_df.index.max()
            # 如果本地数据最新日期是今天，则跳过网络请求
            if last_date.date() == datetime.now().date():
                self._log(f"基金 {fund_code} 历史数据已是最新，无需增量更新。")
                return history_df
            
            self._log(f"基金 {fund_code} 正在请求全量数据以进行增量合并...")

        # 尝试使用 akshare 获取全量数据
        new_data_df = None
        for attempt in range(3):
            try:
                # 抓取全量数据，不使用 start_date, end_date 参数
                new_data = ak.fund_open_fund_info_em(
                    symbol=fund_code, 
                    indicator="单位净值走势"
                )
                if not new_data.empty:
                    new_data['净值日期'] = pd.to_datetime(new_data['净值日期'])
                    new_data.set_index('净值日期', inplace=True)
                    new_data_df = new_data.copy()
                    break
            except Exception as e:
                self._log(f"akshare 获取基金 {fund_code} 全量数据失败 (尝试 {attempt+1}/3): {e}", level='error')
                time.sleep(2)
        
        if new_data_df is None:
            self._log(f"基金 {fund_code} 数据获取失败，将使用本地缓存数据进行计算（可能不是最新的）。", level='warning')
            if history_df is not None:
                 return history_df
            return None # 无法获取任何数据

        # 合并数据并去重 (实现增量更新效果)
        if history_df is None or history_df.empty:
            fund_data = new_data_df
        else:
            # 合并新旧数据，通过索引（日期）去重，保留新数据（新抓取的数据可能更准确或更全）
            # 注意：这里我们信任新数据，但由于新数据是全量的，合并后需要去重。
            fund_data = pd.concat([history_df, new_data_df])
            # 根据索引去重，保留最新的/新抓取的值 (keep='last')
            fund_data = fund_data[~fund_data.index.duplicated(keep='last')]
            
        # 重新按日期排序
        fund_data.sort_index(inplace=True) 

        # 保存更新后的完整历史数据
        if not fund_data.empty:
            self._save_fund_history(fund_code, fund_data)
            
        # 检查是否成功更新
        if last_date is not None and fund_data.index.max() > last_date:
            self._log(f"基金 {fund_code} 增量合并成功，最新日期：{fund_data.index.max().strftime('%Y-%m-%d')}")
        elif last_date is None:
             self._log(f"基金 {fund_code} 首次全量数据抓取成功，最新日期：{fund_data.index.max().strftime('%Y-%m-%d')}")
        
        return fund_data

    def _calculate_metrics(self, fund_data: pd.DataFrame) -> dict:
        """从基金历史净值数据计算夏普比率和最大回撤"""
        
        # 数据清洗：去除异常值和缺失值
        fund_data = fund_data.dropna(subset=['单位净值'])
        
        # 至少一年数据 (252个交易日)
        if len(fund_data) < 252:
            self._log(f"基金数据不足 ({len(fund_data)} 条)，无法计算可靠指标。")
            return {'latest_nav': np.nan, 'sharpe_ratio': np.nan, 'max_drawdown': np.nan}
        
        # 仅使用最近三年的数据进行指标计算（更具时效性）
        three_years_ago = datetime.now() - timedelta(days=3 * 365)
        recent_data = fund_data[fund_data.index >= three_years_ago]
        if recent_data.empty:
             recent_data = fund_data # 如果不足三年，使用所有数据
        
        returns = recent_data['单位净值'].pct_change().dropna()
        
        # 换算到年化
        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * (252**0.5)
        
        # 计算夏普比率
        sharpe_ratio = (annual_returns - self.risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
        
        # 计算最大回撤 (使用所有数据，因为回撤是历史风险的体现)
        rolling_max = fund_data['单位净值'].cummax()
        daily_drawdown = (fund_data['单位净值'] - rolling_max) / rolling_max
        max_drawdown = daily_drawdown.min() * -1
        
        return {
            'latest_nav': float(fund_data['单位净值'].iloc[-1]),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'last_updated': datetime.now().strftime('%Y-%m-%d') # 记录指标更新日期
        }

    def _get_fund_data(self, fund_code: str):
        """
        获取基金的单位净值和累计净值数据，用于计算夏普比率和最大回撤。
        实现：优先使用缓存 -> 检查历史文件 -> 增量更新/全量抓取 -> 计算 -> 保存。
        """
        # 1. 层次二：检查分析结果缓存 (fund_cache.json)
        cached_metrics = self.cache.get('fund', {}).get(fund_code)
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        if cached_metrics and cached_metrics.get('last_updated') == today_str:
            # 如果指标是今天更新的，直接使用
            self.fund_data[fund_code] = cached_metrics
            self._log(f"基金 {fund_code} 数据已使用当日缓存：{self.fund_data[fund_code]}")
            return True

        self._log(f"正在获取/更新基金 {fund_code} 的历史数据...")
        
        # 2. 层次一：加载历史净值数据 (fund_history_data/{code}_history.csv)
        history_df = self._load_fund_history(fund_code)
        
        # 3. 增量更新或全量抓取
        fund_data_df = self._update_and_calculate(fund_code, history_df)
        
        if fund_data_df is None or fund_data_df.empty:
            self.fund_data[fund_code] = {'latest_nav': np.nan, 'sharpe_ratio': np.nan, 'max_drawdown': np.nan}
            self._log(f"基金 {fund_code} 无法获取有效历史数据。", level='error')
            return False

        # 4. 计算指标
        metrics = self._calculate_metrics(fund_data_df)
        
        # 5. 保存并更新缓存
        self.fund_data[fund_code] = metrics
        if self.cache_data:
            self.cache.setdefault('fund', {})[fund_code] = metrics
            self._save_cache()
            
        if pd.isna(metrics['sharpe_ratio']):
             self._log(f"基金 {fund_code} 数据不足，指标计算结果为 NaN。")
        else:
             self._log(f"基金 {fund_code} 数据已计算：{self.fund_data[fund_code]}")
             
        return True


    def _scrape_manager_data_from_web(self, fund_code: str) -> dict:
        """
        从天天基金网通过网页抓取获取基金经理数据
        """
        self._log(f"尝试通过网页抓取获取基金 {fund_code} 的基金经理数据...")
        manager_url = f"http://fundf10.eastmoney.com/jjjl_{fund_code}.html"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(manager_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # 找到包含“基金经理变动一览”文本的标签
            title_label = soup.find('label', string='基金经理变动一览')
            if not title_label:
                return None
            
            # 从父容器中找到表格
            manager_table = title_label.find_parent().find_next_sibling('table')
            if not manager_table:
                return None
            
            rows = manager_table.find_all('tr')
            if len(rows) < 2:
                return None
            
            # 找到第一行数据，即最新任职的经理
            latest_manager_row = rows[1]
            cols = latest_manager_row.find_all('td')
            
            if len(cols) < 5:
                return None
            
            manager_name = cols[2].text.strip()
            tenure_str = cols[3].text.strip()
            cumulative_return_str = cols[4].text.strip()
            
            # 解析任职天数和累计回报
            tenure_days = np.nan
            if '年又' in tenure_str:
                tenure_parts = tenure_str.split('年又')
                years = float(re.search(r'\d+', tenure_parts[0]).group())
                days = float(re.search(r'\d+', tenure_parts[1]).group())
                tenure_days = years * 365 + days
            elif '天' in tenure_str:
                tenure_days = float(re.search(r'\d+', tenure_str).group())
            elif '年' in tenure_str:
                tenure_days = float(re.search(r'\d+', tenure_str).group()) * 365
            else:
                tenure_days = np.nan
                
            cumulative_return = float(re.search(r'[-+]?\d*\.?\d+', cumulative_return_str).group()) if '%' in cumulative_return_str else np.nan

            return {
                'name': manager_name,
                'tenure_years': float(tenure_days) / 365.0 if pd.notna(tenure_days) else np.nan,
                'cumulative_return': cumulative_return
            }
        except requests.exceptions.RequestException as e:
            self._log(f"网页抓取基金 {fund_code} 经理数据失败: {e}", level='error')
            return None
        except Exception as e:
            self._log(f"解析网页内容失败: {e}", level='error')
            return None

    def get_fund_manager_data(self, fund_code: str):
        """
        获取基金经理数据（首先尝试使用 akshare，失败则通过网页抓取）
        """
        if fund_code in self.cache.get('manager', {}):
            self.manager_data[fund_code] = self.cache['manager'][fund_code]
            self._log(f"使用缓存的基金 {fund_code} 经理数据")
            return True

        self._log(f"正在获取基金 {fund_code} 的基金经理数据...")
        try:
            # 修复：akshare接口已变更为 fund_manager_em(code) 或 fund_manager_em(symbol=code)
            # 兼容性起见，使用 symbol=code
            manager_info = ak.fund_manager_em(symbol=fund_code)
            if not manager_info.empty:
                latest_manager = manager_info.sort_values(by='上任日期', ascending=False).iloc[0]
                name = latest_manager.get('姓名', 'N/A')
                tenure_days = latest_manager.get('任职天数', np.nan)
                cumulative_return = latest_manager.get('任职回报', '0%')
                cumulative_return = float(str(cumulative_return).replace('%', '')) if isinstance(cumulative_return, str) else float(cumulative_return)
                
                self.manager_data[fund_code] = {
                    'name': name,
                    'tenure_years': float(tenure_days) / 365.0 if pd.notna(tenure_days) else np.nan,
                    'cumulative_return': cumulative_return
                }
                if self.cache_data:
                    self.cache.setdefault('manager', {})[fund_code] = self.manager_data[fund_code]
                    self._save_cache()
                self._log(f"基金 {fund_code} 经理数据已通过akshare获取：{self.manager_data[fund_code]}")
                return True
        except Exception as e:
            self._log(f"使用akshare获取基金 {fund_code} 经理数据失败: {e}", level='error')

        # 如果akshare失败，尝试网页抓取
        scraped_data = self._scrape_manager_data_from_web(fund_code)
        if scraped_data:
            self.manager_data[fund_code] = scraped_data
            self._log(f"基金 {fund_code} 经理数据已通过网页抓取获取：{self.manager_data[fund_code]}")
            if self.cache_data:
                self.cache.setdefault('manager', {})[fund_code] = self.manager_data[fund_code]
                self._save_cache()
            return True
        else:
            self.manager_data[fund_code] = {'name': 'N/A', 'tenure_years': np.nan, 'cumulative_return': np.nan}
            return False

    def get_market_sentiment(self):
        """获取市场情绪（仅调用一次，基于上证指数）"""
        # 市场情绪数据通常不每天更新，可考虑加入时间戳判断
        if self.market_data and self.market_data.get('last_updated') == datetime.now().strftime('%Y-%m-%d'):
            self._log("使用缓存的市场情绪数据")
            return True

        self._log("正在获取市场情绪数据...")
        try:
            index_data = ak.stock_zh_index_daily_em(symbol="sh000001")
            index_data['date'] = pd.to_datetime(index_data['date'])
            # 使用最近20个交易日数据进行计算，更稳定
            last_20_data = index_data.iloc[-20:]
            
            # 价格变化 (20日)
            price_change = last_20_data['close'].iloc[-1] / last_20_data['close'].iloc[0] - 1
            # 成交量变化 (最近5日均值对比前15日均值)
            volume_change = last_20_data['volume'].iloc[-5:].mean() / last_20_data['volume'].iloc[:-5].mean() - 1
            
            if price_change > 0.03 and volume_change > 0.1:
                sentiment, trend = 'optimistic', 'bullish'
            elif price_change < -0.03 and volume_change > 0.1: # 暴跌放量，恐慌情绪
                sentiment, trend = 'fear', 'bearish'
            elif price_change < -0.03:
                sentiment, trend = 'pessimistic', 'bearish'
            else:
                sentiment, trend = 'neutral', 'neutral'
            
            self.market_data = {
                'sentiment': sentiment, 
                'trend': trend,
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }
            # 同时更新到总缓存
            self.cache['market'] = self.market_data
            self._save_cache()

            self._log(f"市场情绪数据已获取：{self.market_data}")
            return True
        except Exception as e:
            self._log(f"获取市场数据失败: {e}", level='error')
            self.market_data = {'sentiment': 'unknown', 'trend': 'unknown', 'last_updated': datetime.now().strftime('%Y-%m-%d')}
            return False
            
    # --- HOLDINGS FETCHING LOGIC START (Using requests + read_html) ---

    def _clean_holdings_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗持仓数据，统一列名和数据类型"""
        if df.empty:
            return df
            
        # 兼容处理多级索引（pd.read_html常见问题）
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        
        # 移除空行
        df = df.dropna(how='all')
            
        # 标准化列名，根据东方财富表格结构进行假设
        df.columns = [c.strip() for c in df.columns]
        
        # 查找关键列
        code_col = next((c for c in df.columns if '代码' in c and '基金代码' not in c), None)
        name_col = next((c for c in df.columns if '名称' in c and '基金名称' not in c), None)
        ratio_col = next((c for c in df.columns if '占净值比例' in c), None)
        
        if not all([code_col, name_col, ratio_col]):
             self._log(f"持仓数据清洗失败: 未找到关键列 (代码/名称/比例). Columns: {df.columns.tolist()}", level='warning')
             return pd.DataFrame()
             
        # 重命名并只保留核心列
        df.rename(columns={
            code_col: '股票代码', 
            name_col: '股票名称', 
            ratio_col: '占净值比例'
        }, inplace=True)
        
        df = df[['股票代码', '股票名称', '占净值比例']].copy()
        
        # 转换数值列
        if '占净值比例' in df.columns:
            # 移除可能的逗号或百分号，并转换为数字
            df['占净值比例'] = df['占净值比例'].astype(str).str.replace(',', '').str.replace('%', '')
            df['占净值比例'] = pd.to_numeric(df['占净值比例'], errors='coerce')
        
        # 丢弃无效行 (股票代码为空或比例为NaN)
        df = df.dropna(subset=['股票代码', '占净值比例'])
        
        return df

    def _fetch_holdings_by_year(self, fund_code: str, year: int) -> pd.DataFrame or None:
        """
        从东方财富网获取特定年份的最新基金持仓信息（包含所有季度，只取最新的一个表格）
        """
        base_url = "http://fundf10.eastmoney.com"
        # 使用旧版数据接口，返回静态HTML表格
        url = f"{base_url}/FundArchivesDatas.aspx?type=jjcc&code={fund_code}&topline=10&year={year}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = self.session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # 使用 StringIO 包装字符串，避免FutureWarning，并让 read_html 解析
            tables = pd.read_html(StringIO(response.text), encoding='utf-8')
            
            if not tables:
                self._log(f"⚠️ 基金 {fund_code} 在 {year} 年没有表格数据", level='warning')
                return None

            # 东方财富的这个接口返回的表格顺序是从新到旧，我们只需要最新的那个 (tables[0])
            latest_table = tables[0]
            
            # 清洗数据
            cleaned_df = self._clean_holdings_data(latest_table)
            
            if not cleaned_df.empty:
                # 尝试从响应文本中提取最新的季度信息
                # 季度信息位于第一个表格前的 <td> 标签中
                quarter_match = re.search(r'<td>(\d{4}年\d季度)</td>', response.text)
                quarter_info = quarter_match.group(1) if quarter_match else f"{year}年最新季度"
                
                cleaned_df['报告期'] = quarter_info
                self._log(f"✅ 成功获取基金 {fund_code} 在 {year} 年的持仓数据 ({quarter_info})")
                return cleaned_df
            else:
                self._log(f"⚠️ 基金 {fund_code} 在 {year} 年没有有效的持仓数据", level='warning')
                return None
                
        except requests.exceptions.RequestException as e:
            self._log(f"❌ 网络请求失败 - 基金 {fund_code}, 年份 {year}: {e}", level='error')
            return None
        except Exception as e:
            self._log(f"❌ 解析HTML表格或处理数据失败 - 基金 {fund_code}, 年份 {year}: {e}", level='error')
            return None

    def get_fund_holdings_data(self, fund_code: str):
        """
        获取基金的股票持仓数据。
        - 优先使用 akshare
        - 失败则使用 requests + pd.read_html 静态抓取（替换 Selenium 抓取逻辑）
        """
        if fund_code in self.cache.get('holdings', {}):
            self.holdings_data[fund_code] = self.cache['holdings'][fund_code]
            self._log(f"使用缓存的基金 {fund_code} 持仓数据")
            return True

        self._log(f"正在获取基金 {fund_code} 的持仓数据...")
        
        latest_holdings_df = None
        
        # 1. 优先使用 akshare 接口 
        try:
            holdings_df_ak = ak.fund_portfolio_hold_em(symbol=fund_code)
            if not holdings_df_ak.empty:
                # 列名标准化
                holdings_df_ak.columns = ['截止日期', '股票代码', '股票名称', '持仓市值（万元）', '占净值比例']
                holdings_df_ak['占净值比例'] = pd.to_numeric(holdings_df_ak['占净值比例'], errors='coerce')
                latest_holdings_df = holdings_df_ak
                self._log(f"基金 {fund_code} 持仓数据已通过akshare获取。")
        except Exception as e:
            self._log(f"通过akshare获取基金 {fund_code} 持仓数据失败: {e}", level='error')

        # 2. 如果 akshare 失败，使用 requests + pd.read_html 静态抓取
        if latest_holdings_df is None or latest_holdings_df.empty:
            self._log(f"使用静态 HTML 抓取基金 {fund_code} 持仓数据...")
            
            # 尝试抓取当前年份和前一年的数据，以确保获取到最新的报告期
            current_year = datetime.now().year
            
            for year in [current_year, current_year - 1]:
                df = self._fetch_holdings_by_year(fund_code, year)
                if df is not None and not df.empty:
                    latest_holdings_df = df
                    break

        # 3. 最终处理和缓存
        if latest_holdings_df is not None and not latest_holdings_df.empty:
            # 转换为字典列表格式，用于缓存和后续分析
            holdings_for_cache = latest_holdings_df[['股票代码', '股票名称', '占净值比例']].fillna(np.nan).to_dict('records')
            
            # 补充持仓市值（万元）列，保持数据结构一致
            for item in holdings_for_cache:
                if '持仓市值（万元）' not in item:
                    item['持仓市值（万元）'] = np.nan
                    
            self.holdings_data[fund_code] = holdings_for_cache
            
            if self.cache_data:
                self.cache.setdefault('holdings', {})[fund_code] = self.holdings_data[fund_code]
                self._save_cache()
            return True
        else:
            self._log(f"获取基金 {fund_code} 持仓数据最终失败。", level='error')
            self.holdings_data[fund_code] = []
            return False
            
    # --- HOLDINGS FETCHING LOGIC END ---
            
    def _calculate_ranks(self, all_funds_metrics):
        """
        计算所有基金的夏普比率和最大回撤的百分位排名。
        - 夏普比率：越高越好 (百分位越高)
        - 最大回撤：越小越好 (百分位越高)
        """
        if not all_funds_metrics:
            self.ranking_data = {}
            return
            
        df = pd.DataFrame(all_funds_metrics)
        
        # 排除夏普比率和最大回撤为 NaN 的基金，只对有数据的基金进行排名
        valid_df = df.dropna(subset=['sharpe_ratio', 'max_drawdown']).copy()
        
        if valid_df.empty:
            self.ranking_data = {}
            return
            
        N = len(valid_df)
        
        # 1. 夏普比率百分位：越高越好
        # rank(pct=True, ascending=True) 自动计算百分位排名，最大值接近 1.0 (Top 1% 接近 1.0)
        valid_df['sharpe_percentile'] = valid_df['sharpe_ratio'].rank(pct=True, ascending=True) * 100
        
        # 2. 最大回撤百分位：越小越好
        # rank(pct=True, ascending=True) 赋予最小值最低的排名（接近0），
        # 使用 (1 - rank) * 100 反转，使得最小值（最好）获得最高百分位（接近100）
        drawdown_pct_rank = valid_df['max_drawdown'].rank(pct=True, ascending=True)
        valid_df['drawdown_percentile'] = (1 - drawdown_pct_rank) * 100
        
        # 将结果存储到 self.ranking_data 字典中
        self.ranking_data = valid_df[['fund_code', 'sharpe_percentile', 'drawdown_percentile']].set_index('fund_code').T.to_dict()
        self._log(f"已计算 {len(self.ranking_data)} 个基金的相对排名数据。")
            
    def _evaluate_fund(self, fund_code, fund_name, fund_type):
        """
        评估单个基金的综合分数，夏普比率和最大回撤的评分逻辑已改为基于相对排名。
        """
        self._log(f"--- 正在分析基金 {fund_code} ---")
        
        # 尝试获取基本信息，如果失败则跳过整个分析
        if not self._get_fund_data(fund_code):
            self._log(f"基金 {fund_code} 基本信息获取失败，跳过分析。")
            self.report_data.append({'fund_code': fund_code, 'fund_name': fund_name, 'decision': 'Skip', 'score': np.nan})
            return
            
        # 获取基金经理数据
        self.get_fund_manager_data(fund_code)
        
        # 获取持仓数据
        self.get_fund_holdings_data(fund_code)

        # 评分体系 (示例)
        scores = {}
        values = {}
        
        # 确保获取到原始值
        sharpe_ratio = self.fund_data[fund_code].get('sharpe_ratio')
        max_drawdown = self.fund_data[fund_code].get('max_drawdown')
        
        # 获取相对排名数据
        rank_data = self.ranking_data.get(fund_code, {})
        sharpe_percentile = rank_data.get('sharpe_percentile', np.nan)
        drawdown_percentile = rank_data.get('drawdown_percentile', np.nan)

        # 1. 夏普比率评分 (基于相对排名)
        if pd.notna(sharpe_percentile):
            # 排名越高（百分位越接近100）得分越高
            if sharpe_percentile >= 90:
                scores['sharpe_ratio_score'] = 10 # Top 10%
            elif sharpe_percentile >= 80:
                scores['sharpe_ratio_score'] = 8
            elif sharpe_percentile >= 70:
                scores['sharpe_ratio_score'] = 6
            elif sharpe_percentile >= 50:
                scores['sharpe_ratio_score'] = 4
            else:
                scores['sharpe_ratio_score'] = 2
                
            # 显示原始值和百分位排名
            percentile_rank_display = sharpe_percentile
            values['sharpe_ratio_value'] = f"{sharpe_ratio:.4f} (优于 {percentile_rank_display:.1f}% 的基金)"
        else:
            scores['sharpe_ratio_score'] = 0
            values['sharpe_ratio_value'] = np.nan

        # 2. 最大回撤评分 (基于相对排名)
        if pd.notna(drawdown_percentile):
            # 排名越高（百分位越接近100，即回撤越小）得分越高
            if drawdown_percentile >= 90:
                scores['max_drawdown_score'] = 10 # 回撤控制Top 10%
            elif drawdown_percentile >= 80:
                scores['max_drawdown_score'] = 8
            elif drawdown_percentile >= 70:
                scores['max_drawdown_score'] = 6
            elif drawdown_percentile >= 50:
                scores['max_drawdown_score'] = 4
            else:
                scores['max_drawdown_score'] = 2
            
            # 显示原始值和百分位排名
            percentile_rank_display = drawdown_percentile
            values['max_drawdown_value'] = f"{max_drawdown:.4f} (回撤控制优于 {percentile_rank_display:.1f}% 的基金)"
        else:
            scores['max_drawdown_score'] = 0
            values['max_drawdown_value'] = np.nan
            
        # 3. 基金经理任职年限评分 (保持不变)
        manager_years = self.manager_data[fund_code].get('tenure_years')
        if pd.notna(manager_years) and manager_years >= 3:
            scores['manager_years_score'] = 10
        else:
            scores['manager_years_score'] = 0
        values['manager_years_value'] = manager_years
        
        # 4. 基金经理任职回报评分 (保持不变)
        manager_return = self.manager_data[fund_code].get('cumulative_return')
        if pd.notna(manager_return) and manager_return > 0:
            scores['manager_return_score'] = 10
        else:
            scores['manager_return_score'] = 0
        values['manager_return_value'] = manager_return
            
        # 5. 持仓集中度评分 (保持不变)
        if self.holdings_data[fund_code]:
            holdings_df = pd.DataFrame(self.holdings_data[fund_code])
            # 确保 DataFrame 不为空且包含 '占净值比例' 列
            if '占净值比例' in holdings_df.columns and not holdings_df.empty:
                # 确保取前10个，避免索引错误
                top_10_holdings_ratio = holdings_df['占净值比例'].iloc[:10].sum()
                if top_10_holdings_ratio < 60:
                    scores['holding_concentration_score'] = 10
                else:
                    scores['holding_concentration_score'] = 5
                values['holding_concentration_value'] = top_10_holdings_ratio
            else:
                scores['holding_concentration_score'] = 0
                values['holding_concentration_value'] = np.nan
        else:
            scores['holding_concentration_score'] = 0
            values['holding_concentration_value'] = np.nan
        
        # 其他评分项... (保持不变)
        scores['fund_type_score'] = 10 if '股票型' in fund_type or '混合型' in fund_type else 5
        # 市场情绪调整：如果市场看涨且基金夏普比率排名靠前，则加分
        scores['market_sentiment_adj_score'] = 5 if self.market_data.get('trend') == 'bullish' and scores.get('sharpe_ratio_score', 0) > 5 else 0
        
        total_score = sum(scores.values())
        
        # 决策逻辑
        decision = '推荐' if total_score > 30 else '观望'
        
        self.report_data.append({
            'fund_code': fund_code,
            'fund_name': fund_name,
            'decision': decision,
            'score': total_score,
            'scores_details': scores,
            'values_details': values
        })
        self._log(f"评分详情: {scores}")
        self._log(f"基金 {fund_code} 评估完成，总分: {total_score}，决策: {decision}")

    def _save_report_to_markdown(self):
        """将分析报告保存为 Markdown 文件"""
        if not self.report_data:
            return
        
        report_path = "analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("--- 批量基金分析报告 ---\n\n")
            f.write(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("--- 汇总结果 ---\n\n")
            
            results_df = pd.DataFrame(self.report_data)
            valid_results = results_df[results_df['decision'] != 'Skip']
            
            if not valid_results.empty:
                f.write("### 推荐基金\n\n")
                # 使用 fillna 替换 NaN 以便 to_markdown 正常工作
                recommended = valid_results[valid_results['decision'] == '推荐'].sort_values(by='score', ascending=False).fillna('N/A')
                if not recommended.empty:
                    f.write(recommended[['fund_code', 'fund_name', 'score']].to_markdown(index=False) + "\n\n")
                else:
                    f.write("无\n\n")
                    
                f.write("### 观望基金\n\n")
                watchlist = valid_results[valid_results['decision'] == '观望'].sort_values(by='score', ascending=False).fillna('N/A')
                if not watchlist.empty:
                    f.write(watchlist[['fund_code', 'fund_name', 'score']].to_markdown(index=False) + "\n\n")
                else:
                    f.write("无\n\n")
            
            f.write("--- 详细分析 ---\n\n")
            for item in self.report_data:
                f.write(f"### 基金 {item['fund_code']} - {item.get('fund_name', 'N/A')}\n")
                f.write(f"- 最终决策: **{item['decision']}**\n")
                # 安全地获取分数并格式化，处理 NaN
                score_display = f"{item['score']:.2f}" if pd.notna(item['score']) else 'N/A'
                f.write(f"- 综合分数: **{score_display}**\n")
                
                if item['decision'] != 'Skip':
                    # 标记持仓基金
                    if item['fund_code'] in self.holding_codes:
                        f.write("- **持仓状态**: **是**\n")
                        
                    f.write("- **评分细项**:\n")
                    for k, v in item.get('scores_details', {}).items():
                        # 安全地处理 None/NaN 的评分
                        score_v = str(v) if pd.notna(v) else 'N/A'
                        f.write(f"  - {k}: {score_v}\n")
                        
                    f.write("- **数据值**:\n")
                    for k, v in item.get('values_details', {}).items():
                        # 仅显示非 NaN 的数据值
                        if pd.notna(v):
                            # 对于夏普比率和最大回撤，v已经是格式化后的字符串
                            if 'ratio_value' in k or 'drawdown_value' in k:
                                f.write(f"  - {k}: {v}\n")
                            # 对于其他数值，进行格式化
                            elif isinstance(v, (float, int)) and ('return' in k or 'concentration' in k or 'years' in k):
                                f.write(f"  - {k}: {v:.4f}\n")
                            else:
                                f.write(f"  - {k}: {v}\n")
                        else:
                             f.write(f"  - {k}: N/A\n")
                f.write("\n---\n\n")

    def run_analysis(self, fund_codes: list, fund_info: dict, fund_type_info: dict):
        """
        运行批量基金分析的主函数。
        fund_codes: 从 CSV 中导入的基金代码
        fund_info: 从 CSV 中导入的基金名称信息
        fund_type_info: 从 CSV 中导入的基金类型信息
        """
        self._log("--- 批量基金分析启动 ---")
        
        # 1. 合并基金列表：将推荐基金和持仓基金合并并去重
        all_codes_set = set(fund_codes)
        all_codes_set.update(self.holding_codes)
        all_codes_to_analyze = sorted(list(all_codes_set))
        
        # 2. 更新基金名称字典和类型字典：将持仓基金代码（可能缺失名称/类型）加入字典
        for code in self.holding_codes:
            if code not in fund_info:
                fund_info[code] = "持仓基金" # 标记为持仓基金，方便后续识别
            if code not in fund_type_info:
                # 假设持仓基金最可能是混合型
                fund_type_info[code] = "混合型" 
        
        self._log(f"合并后的分析基金列表总数: {len(all_codes_to_analyze)} (推荐基金: {len(fund_codes)}, 持仓基金: {len(self.holding_codes)})")
        
        # 3. 预先获取所有基金的基础指标，用于计算相对排名 (现在包含缓存/增量更新逻辑)
        all_funds_metrics = []
        self._log("正在预先获取所有基金数据以计算相对排名 (含缓存/增量更新)...")
        for code in all_codes_to_analyze:
            # _get_fund_data 会自动处理缓存和增量更新
            if self._get_fund_data(code):
                # 只有成功获取（即使指标为 NaN）才将其加入 FundAnalyzer 的 fund_data 中
                metrics = self.fund_data[code].copy()
                metrics['fund_code'] = code
                all_funds_metrics.append(metrics)
        self._log("基础指标获取完毕。")
        
        # 4. 计算相对排名
        self._calculate_ranks(all_funds_metrics)
        
        # 5. 仅调用一次获取市场情绪 (含缓存逻辑)
        self.get_market_sentiment()
        
        # 6. 逐个评估基金
        for code in all_codes_to_analyze:
            fund_name = fund_info.get(code, 'N/A')
            fund_type = fund_type_info.get(code, 'N/A')
            self._evaluate_fund(code, fund_name, fund_type) 
        
        # 生成并保存最终报告
        results_df = pd.DataFrame(self.report_data)
        if not results_df.empty and 'decision' in results_df.columns:
            self._log("\n--- 全部基金分析结果 ---")
            # 使用 fillna 确保输出到日志的 DataFrame 也能正确显示
            log_df = results_df[['decision', 'score', 'fund_code', 'fund_name']].fillna({'score': 'N/A', 'fund_name': 'N/A'}).copy()
            self._log("\n" + log_df.to_markdown(index=False))
        else:
            self._log("\n没有基金获得有效评分。")
        
        self._save_report_to_markdown()
        
        return results_df

if __name__ == '__main__':
    # 请确保已安装所有库，特别是 Selenium 和 ChromeDriver
    # pip install selenium akshare pandas numpy requests beautifulsoup4 lxml
    # 还需要手动下载与您的 Chrome 版本匹配的 ChromeDriver 并配置环境变量或修改路径
    
    funds_list_url = 'https://raw.githubusercontent.com/qjlxg/rep/main/recommended_cn_funds.csv'
    
    # --- 核心修改部分：修复 CSV 编码问题，使用 UTF-8 ---
    try:
        logger.info("正在从 CSV 导入基金代码列表...")
        # 核心修复：将编码改为最通用的 'utf-8'，这解决了 'gbk' codec can't decode byte 0x80 的错误。
        df_funds = pd.read_csv(funds_list_url, encoding='utf-8', engine='python', usecols=['code', 'name', '类型'])
        
        # 确保 code 是 6 位字符串格式
        df_funds['code'] = df_funds['code'].astype(str).str.zfill(6)
        
        # 从匹配的列名中提取数据
        fund_codes_to_analyze = df_funds['code'].unique().tolist()
        fund_info_dict = dict(zip(df_funds['code'], df_funds['name']))
        fund_type_info_dict = dict(zip(df_funds['code'], df_funds['类型']))
        
        logger.info(f"导入成功，共 {len(fund_codes_to_analyze)} 个推荐基金代码")
    except KeyError as e:
        logger.error(f"导入基金列表失败: CSV 文件中缺少必要的列 ({e})。请检查 CSV 文件的列名是否为 'code', 'name', '类型'。", exc_info=True)
        fund_codes_to_analyze = []
        fund_info_dict = {}
        fund_type_info_dict = {}
    except Exception as e:
        logger.error(f"导入基金列表失败: {e}", exc_info=True)
        fund_codes_to_analyze = []
        fund_info_dict = {}
        fund_type_info_dict = {}
    # --------------------------------------------------
    
    analyzer = FundAnalyzer()
    
    if fund_codes_to_analyze or analyzer.holding_codes:
        logger.info(f"开始分析基金...")
        
        # 传递推荐列表、名称字典和类型字典
        analyzer.run_analysis(fund_codes_to_analyze, fund_info_dict, fund_type_info_dict)
    else:
        logger.info("没有基金列表可供分析，程序结束。")
