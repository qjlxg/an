import pandas as pd
import numpy as np
import re
import os
import logging
from datetime import datetime, timedelta
import time
import random
from io import StringIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException, StaleElementReferenceException
import requests
import tenacity

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
            
            # 使用更精确的正则表达式来匹配基金代码
            # 1. 匹配表格中的基金代码，例如：| 007509 |
            # 2. 匹配详细分析中的基金代码，例如：### 基金 001407 -
            pattern = re.compile(r'(?:^\| +(\d{6})|### 基金 (\d{6}))', re.M)
            matches = pattern.findall(content)

            extracted_codes = set()
            for match in matches:
                # findall 返回的是一个元组，我们需要提取非空的那个
                code = match[0] if match[0] else match[1]
                extracted_codes.add(code)
            
            # 将集合转换为列表并进行排序
            sorted_codes = sorted(list(extracted_codes))
            self.fund_codes = sorted_codes[:10]  # 限制前10个有效代码
            
            if not self.fund_codes:
                logger.warning("未提取到任何有效基金代码，请检查 analysis_report.md")
            else:
                logger.info("提取到 %d 个基金（测试限制前10个）: %s", len(self.fund_codes), self.fund_codes)
            for handler in logger.handlers:
                handler.flush()
            
        except Exception as e:
            logger.error("解析报告文件失败: %s", e)
            raise

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_fixed(2),
        retry=tenacity.retry_if_exception_type((TimeoutException, WebDriverException)),
        before_sleep=lambda retry_state: logger.info(f"重试基金 {retry_state.args[1]}，第 {retry_state.attempt_number} 次")
    )
    def _get_fund_data_from_eastmoney(self, fund_code):
        """使用 Selenium 从 fund.eastmoney.com 抓取基金历史净值数据（含翻页）"""
        logger.info("正在获取基金 %s 的净值数据...", fund_code)
        
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
            options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36')
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            
            url = f"http://fundf10.eastmoney.com/jjjz_{fund_code}.html"
            driver.set_page_load_timeout(15)  # 页面加载超时15秒
            driver.get(url)
            logger.info("访问URL: %s", url)

            all_data = []
            page_count = 1
            
            while True:
                try:
                    # 等待表格容器加载，并确保内容可见
                    wait = WebDriverWait(driver, 15)
                    wait.until(EC.visibility_of_element_located((By.ID, 'jztable')))
                    logger.info("第 %d 页: 历史净值表格容器加载完成并可见", page_count)

                    # 等待分页导航加载
                    wait.until(EC.presence_of_element_located((By.ID, 'pagebar')))
                    logger.info("第 %d 页: 分页导航容器加载完成", page_count)

                    # 解析当前页面表格
                    table_html = driver.find_element(By.ID, 'jztable').get_attribute('innerHTML')
                    df_list = pd.read_html(StringIO(table_html), flavor='lxml')
                    
                    if not df_list or df_list[0].empty:
                        logger.warning("第 %d 页: 表格内容为空，可能已无更多数据", page_count)
                        break

                    df = df_list[0]
                    df = df.iloc[:, [0, 1]].copy()  # 只取日期和单位净值
                    df.columns = ['date', 'net_value']
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
                    df = df.dropna(subset=['date', 'net_value'])
                    all_data.append(df)
                    logger.info("第 %d 页: 解析成功，获取 %d 行数据", page_count, len(df))

                    # 检查是否有下一页按钮
                    try:
                        # 使用更宽松的XPath，匹配包含“下一页”文本的元素
                        next_button_xpath = "//div[@id='pagebar']//a[contains(text(), '下一页')]"
                        wait.until(EC.element_to_be_clickable((By.XPATH, next_button_xpath)))
                        next_button = driver.find_element(By.XPATH, next_button_xpath)
                        
                        # 检查按钮是否可点击（通过检查class是否包含禁用标志）
                        button_class = next_button.get_attribute('class') or ''
                        if 'nolink' in button_class or 'disabled' in button_class:
                            logger.info("基金 %s 已到达最后一页，翻页结束", fund_code)
                            break
                        
                        # 使用JavaScript点击，增加可靠性
                        driver.execute_script("arguments[0].click();", next_button)
                        page_count += 1
                        time.sleep(random.uniform(2, 3))  # 增加延迟以等待页面加载

                    except (NoSuchElementException, StaleElementReferenceException):
                        logger.info("基金 %s 无下一页按钮，或按钮已失效，翻页结束", fund_code)
                        break
                    except TimeoutException:
                        logger.info("基金 %s 等待下一页按钮超时，可能已到达最后一页", fund_code)
                        break

                except TimeoutException:
                    logger.info("基金 %s 页面加载超时，可能已到达最后一页", fund_code)
                    break
                except Exception as e:
                    logger.error("翻页失败: %s", str(e))
                    break

            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                df = df.drop_duplicates(subset=['date']).sort_values(by='date', ascending=True)
                df = df.tail(100)  # 取最近 100 天
                logger.info("成功解析基金 %s 的数据，共获取 %d 页，总行数: %d, 最新日期: %s, 最新净值: %.4f", 
                            fund_code, page_count, len(df), df['date'].iloc[-1].strftime('%Y-%m-%d'), df['net_value'].iloc[-1])
                return df[['date', 'net_value']]
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

    def get_fund_data(self):
        """获取所有基金的数据"""
        logger.info("开始获取 %d 个基金的数据...", len(self.fund_codes))
        for i, fund_code in enumerate(self.fund_codes, 1):
            try:
                logger.info("处理第 %d/%d 个基金: %s", i, len(self.fund_codes), fund_code)
                df = self._get_fund_data_from_eastmoney(fund_code)
                
                if df is not None and not df.empty and len(df) >= 14:
                    df = df.sort_values(by='date', ascending=True)
                    delta = df['net_value'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=14, min_periods=1).mean()
                    avg_loss = loss.rolling(window=14, min_periods=1).mean()
                    
                    # 避免除以0
                    rs = avg_gain / avg_loss.replace(0, np.nan)
                    rsi = 100 - (100 / (1 + rs))
                    
                    ma50 = df['net_value'].rolling(window=min(50, len(df)), min_periods=1).mean()
                    
                    latest_data = df.iloc[-1]
                    latest_net_value = latest_data['net_value']
                    latest_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else np.nan
                    latest_ma50 = ma50.iloc[-1]
                    latest_ma50_ratio = latest_net_value / latest_ma50 if not pd.isna(latest_ma50) and latest_ma50 != 0 else np.nan

                    self.fund_data[fund_code] = {
                        'latest_net_value': latest_net_value,
                        'rsi': latest_rsi,
                        'ma_ratio': latest_ma50_ratio
                    }
                    logger.info("成功计算基金 %s 的技术指标: 净值=%.4f, RSI=%.2f, MA50比率=%.2f", 
                                fund_code, latest_net_value, latest_rsi, latest_ma50_ratio)
                else:
                    self.fund_data[fund_code] = None
                    logger.warning("基金 %s 数据获取失败或数据不足，跳过计算 (数据行数: %s)", fund_code, len(df) if df is not None else 0)
                
                for handler in logger.handlers:
                    handler.flush()
                # 在处理完一个基金后，随机延迟1到3秒
                time.sleep(random.uniform(1, 3))
            except Exception as e:
                logger.error("处理基金 %s 时发生异常: %s", fund_code, str(e))
                self.fund_data[fund_code] = None
                for handler in logger.handlers:
                    handler.flush()

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

                        # 使用更安全的格式化逻辑
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
