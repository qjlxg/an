import pandas as pd
import numpy as np
import re
import os
import logging
from datetime import datetime
import time
import random
from io import StringIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, WebDriverException
import requests
import tenacity
import concurrent.futures

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

    def _parse_report(self):
        """从 analysis_report.md 提取推荐基金代码"""
        logger.info("正在解析 %s 获取推荐基金代码...", self.report_file)
        if not os.path.exists(self.report_file):
            logger.error("报告文件 %s 不存在", self.report_file)
            raise FileNotFoundError(f"{self.report_file} 不存在")
        
        try:
            with open(self.report_file, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info("analysis_report.md 内容（前1000字符）: %s", content[:1000])
            
            pattern = re.compile(r'(?:^\| +(\d{6})|### 基金 (\d{6}))', re.M)
            matches = pattern.findall(content)

            extracted_codes = set()
            for match in matches:
                code = match[0] if match[0] else match[1]
                extracted_codes.add(code)
            
            sorted_codes = sorted(list(extracted_codes))
            self.fund_codes = sorted_codes[:10]
            
            if not self.fund_codes:
                logger.warning("未提取到任何有效基金代码，请检查 analysis_report.md")
            else:
                logger.info("提取到 %d 个基金（测试限制前10个）: %s", len(self.fund_codes), self.fund_codes)
            for handler in logger.handlers:
                handler.flush()
            
        except Exception as e:
            logger.error("解析报告文件失败: %s", e)
            raise

    def _get_latest_local_date(self, fund_code):
        """检查本地是否存在数据，并返回最新日期"""
        file_path = os.path.join(DATA_DIR, f"{fund_code}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if not df.empty and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    latest_date = df['date'].max().date()
                    logger.info("本地已存在基金 %s 数据，最新日期为: %s", fund_code, latest_date)
                    return latest_date, df
            except Exception as e:
                logger.warning("读取本地文件 %s 失败: %s", file_path, e)
        return None, pd.DataFrame()

    def _save_to_local_file(self, fund_code, new_data_df):
        """将新数据追加到本地文件"""
        file_path = os.path.join(DATA_DIR, f"{fund_code}.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if os.path.exists(file_path):
            new_data_df.to_csv(file_path, mode='a', header=False, index=False)
            logger.info("新数据已追加到本地文件: %s", file_path)
        else:
            new_data_df.to_csv(file_path, index=False)
            logger.info("新数据已保存到本地文件: %s", file_path)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_fixed(5),
        retry=tenacity.retry_if_exception_type((TimeoutException, WebDriverException, IndexError, Exception)), # 增加对 IndexEror 和更广泛的 Exception 的重试
        before_sleep=lambda retry_state: logger.info(f"重试基金 {retry_state.args[1]}，第 {retry_state.attempt_number} 次")
    )
    def _get_fund_data_from_eastmoney(self, fund_code):
        """使用 Selenium 从 fund.eastmoney.com 抓取基金历史净值数据（含URL参数翻页）"""
        logger.info("正在获取基金 %s 的净值数据...", fund_code)
        
        latest_local_date, local_df = self._get_latest_local_date(fund_code)
        
        driver = None
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-infobars')
            options.add_argument('--disable-software-rasterizer')
            options.add_argument('--enable-javascript-i18n-api')
            options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36')
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            
            all_data = []
            page_index = 1
            max_pages = 200
            
            found_new_data = False
            
            while page_index <= max_pages:
                try:
                    url = f"http://fundf10.eastmoney.com/jjjz_{fund_code}.html?p={page_index}"
                    driver.set_page_load_timeout(40)
                    driver.get(url)
                    logger.info("访问URL: %s", url)
                    
                    wait = WebDriverWait(driver, 30)
                    wait.until(EC.visibility_of_element_located((By.ID, 'jztable')))
                    logger.info("第 %d 页: 历史净值表格容器加载完成并可见", page_index)
                    
                    # 尝试解析表格，如果失败则捕获 IndexEror
                    try:
                        table_html = driver.find_element(By.ID, 'jztable').get_attribute('innerHTML')
                        df_list = pd.read_html(StringIO(table_html), flavor='lxml')
                    except IndexError as e:
                        logger.error("基金 %s 第 %d 页解析表格失败: %s", fund_code, page_index, str(e))
                        # 尝试捕获是否有反爬页面
                        if "验证码" in driver.page_source or "Access Denied" in driver.page_source:
                            logger.error("检测到反爬措施，跳过此基金。")
                            raise Exception("反爬措施")
                        # 如果没有反爬措施，页面可能只是没有表格，继续正常处理
                        df_list = []
                        
                    if not df_list or df_list[0].empty:
                        logger.warning("第 %d 页: 表格内容为空，可能已无更多数据", page_index)
                        break

                    df = df_list[0]
                    df = df.iloc[:, [0, 1]].copy()
                    df.columns = ['date', 'net_value']
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
                    df = df.dropna(subset=['date', 'net_value'])
                    
                    if latest_local_date:
                        new_df = df[df['date'].dt.date > latest_local_date]
                        if not new_df.empty:
                            all_data.append(new_df)
                            found_new_data = True
                            logger.info("第 %d 页: 发现 %d 行新数据", page_index, len(new_df))
                        
                        if new_df.empty and len(df) == 20:
                            logger.info("基金 %s 已获取到最新数据，爬取结束", fund_code)
                            break
                    else:
                        all_data.append(df)
                        found_new_data = True

                    if len(df) < 20:
                        logger.info("基金 %s 数据量不足20行，翻页结束", fund_code)
                        break
                        
                    page_index += 1
                    time.sleep(random.uniform(2, 4))

                except TimeoutException as e:
                    logger.error("基金 %s 页面加载超时: %s", fund_code, str(e))
                    raise
                except Exception as e:
                    logger.error("基金 %s 翻页失败: %s", fund_code, str(e))
                    break

            if all_data:
                new_full_df = pd.concat(all_data, ignore_index=True)
                new_full_df = new_full_df.drop_duplicates(subset=['date']).sort_values(by='date', ascending=True)

                if found_new_data:
                    self._save_to_local_file(fund_code, new_full_df)

                if not local_df.empty:
                    df_combined = pd.concat([local_df, new_full_df]).drop_duplicates(subset=['date']).sort_values(by='date', ascending=True)
                else:
                    df_combined = new_full_df
                    
                if len(df_combined) < 100:
                    logger.warning("基金 %s 总数据量不足，仅获取 %d 行", fund_code, len(df_combined))
                
                df_combined = df_combined.tail(100)
                logger.info("成功解析基金 %s 的数据，共获取 %d 页，总行数: %d, 最新日期: %s, 最新净值: %.4f", 
                                 fund_code, page_index - 1, len(df_combined), df_combined['date'].iloc[-1].strftime('%Y-%m-%d'), df_combined['net_value'].iloc[-1])
                return df_combined[['date', 'net_value']]
            else:
                if not local_df.empty:
                    logger.info("基金 %s 无新数据，使用本地历史数据", fund_code)
                    df_combined = local_df.tail(100)
                    return df_combined[['date', 'net_value']]
                else:
                    raise ValueError("未获取到任何有效数据")

        except Exception as e:
            logger.error("Selenium 抓取基金 %s 失败: %s", fund_code, str(e))
            if driver:
                try:
                    driver.save_screenshot(f"error_screenshot_{fund_code}.png")
                    with open(f"error_page_{fund_code}.html", "w", encoding="utf-8") as f:
                        f.write(driver.page_source)
                    logger.info("错误截图和页面已保存到 error_screenshot_%s.png 和 error_page_%s.html", fund_code, fund_code)
                except:
                    logger.warning("无法保存错误截图或页面源码")
            raise
        finally:
            if driver:
                driver.quit()

    def process_fund(self, fund_code):
        """处理单个基金，用于多线程调用"""
        try:
            df = self._get_fund_data_from_eastmoney(fund_code)
            
            if df is not None and not df.empty and len(df) >= 14:
                df = df.sort_values(by='date', ascending=True)
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

                return {
                    'fund_code': fund_code,
                    'latest_net_value': latest_net_value,
                    'rsi': latest_rsi,
                    'ma_ratio': latest_ma50_ratio
                }
            else:
                logger.warning("基金 %s 数据获取失败或数据不足，跳过计算 (数据行数: %s)", fund_code, len(df) if df is not None else 0)
                return None
        except Exception as e:
            logger.error("处理基金 %s 时发生异常: %s", fund_code, str(e))
            return None
        finally:
            for handler in logger.handlers:
                handler.flush()

    def get_fund_data(self):
        """使用多线程获取所有基金的数据"""
        logger.info("开始使用多线程获取 %d 个基金的数据...", len(self.fund_codes))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(self.process_fund, self.fund_codes))

        self.fund_data = {
            result['fund_code']: {
                'latest_net_value': result['latest_net_value'],
                'rsi': result['rsi'],
                'ma_ratio': result['ma_ratio']
            }
            for result in results if result is not None
        }

        if len(self.fund_data) > 0:
            logger.info("所有基金数据处理完成。")
        else:
            logger.error("所有基金数据均获取失败。")

    def generate_report(self):
        """生成市场情绪与技术指标监控报告"""
        logger.info("正在生成市场监控报告...")
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 市场情绪与技术指标监控报告\n\n")
            f.write(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## 推荐基金技术指标 (处理基金数: {len(self.fund_codes)})\n")
            f.write("| 基金代码 | 最新净值 | RSI | 净值/MA50 | 投资建议 |\n")
            f.write("|----------|----------|-----|-----------|----------|\n")
            
            if not self.fund_codes:
                f.write("| 无 | 无数据 | - | - | 请检查 analysis_report.md 是否包含有效基金代码 |\n")
            else:
                for fund_code in self.fund_codes:
                    if fund_code in self.fund_data and self.fund_data[fund_code] is not None:
                        data = self.fund_data[fund_code]
                        rsi = data['rsi']
                        ma_ratio = data['ma_ratio']

                        rsi_str = f"{rsi:.2f}" if not np.isnan(rsi) else "N/A"
                        ma_ratio_str = f"{ma_ratio:.2f}" if not np.isnan(ma_ratio) else "N/A"

                        advice = (
                            "等待回调" if not np.isnan(rsi) and rsi > 70 or not np.isnan(ma_ratio) and ma_ratio > 1.2 else
                            "可分批买入" if (np.isnan(rsi) or 30 <= rsi <= 70) and (np.isnan(ma_ratio) or 0.8 <= ma_ratio <= 1.2) else
                            "可加仓" if not np.isnan(rsi) and rsi < 30 else "观察"
                        )
                        f.write(f"| {fund_code} | {data['latest_net_value']:.4f} | {rsi_str} | {ma_ratio_str} | {advice} |\n")
                    else:
                        f.write(f"| {fund_code} | 数据获取失败 | - | - | 观察 |\n")
        
        logger.info("报告生成完成: %s", self.output_file)
        with open(self.output_file, 'r', encoding='utf-8') as f:
            logger.info("market_monitor_report.md 内容: %s", f.read())
        for handler in logger.handlers:
            handler.flush()

if __name__ == "__main__":
    try:
        logger.info("脚本启动")
        monitor = MarketMonitor()
        monitor._parse_report()
        monitor.get_fund_data()
        monitor.generate_report()
        logger.info("脚本执行完成")
    except Exception as e:
        logger.error("脚本运行失败: %s", e)
        for handler in logger.handlers:
            handler.flush()
        raise
