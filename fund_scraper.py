import asyncio
import aiohttp
from aiohttp import ClientSession
import re
import json5
import jsbeautifier
import pandas as pd
import os
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 配置常量：直接编码在脚本中 ---
FUND_LIST_FILE = "recommended_cn_funds.csv"
BASE_URL_INFO_JS = "http://fund.eastmoney.com/pingzhongdata/{fund_code}.js"
OUTPUT_DIR = 'fund_metrics'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'http://fund.eastmoney.com/',
}
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.5
MAX_CONCURRENT = 5
# --- 结束配置常量 ---

# 从 CSV 读取基金代码并补零 
def get_fund_codes_from_csv(file_path):
    """从 CSV 文件读取基金代码，并根据关键字过滤，将其填充为 6 位数字。"""
    EXCLUDE_KEYWORDS = ['币', '债', '持有', 'A', 'B']
    
    try:
        # 修复：显式指定编码为 'gb18030' 以兼容中文 CSV 文件
        df = pd.read_csv(file_path, encoding='gb18030')
        
        if 'code' not in df.columns or 'name' not in df.columns:
            logger.error(f"CSV file '{file_path}' 必须包含 'code' 和 'name' 列。")
            return []

        # 1. 过滤掉包含排除关键字的基金
        initial_count = len(df)
        mask = pd.Series([True] * initial_count)
        
        for keyword in EXCLUDE_KEYWORDS:
            mask &= ~df['name'].astype(str).str.contains(keyword, case=False, na=False)
            
        filtered_df = df[mask].copy()
        
        filtered_count = len(filtered_df)
        logger.info(f"读取到基金总数: {initial_count}. 排除基金数: {initial_count - filtered_count}. 排除关键字: {EXCLUDE_KEYWORDS}.")

        # 2. 格式化基金代码
        fund_codes = filtered_df['code'].astype(str).str.strip().str.zfill(6).tolist()
        
        # 3. 过滤掉非数字或不满足 6 位长度的无效代码
        fund_codes = [code for code in fund_codes if code.isdigit() and len(code) == 6]
        logger.info(f"最终保留并格式化了 {len(fund_codes)} 个有效的基金代码。")
        
        return fund_codes
        
    except Exception as e:
        logger.error(f"读取和过滤基金代码出错: {e}")
        return []

# 辅助函数：提取 JS 变量内容
def extract_js_variable_content(text, var_name):
    start_match = re.search(r'var\s+' + re.escape(var_name) + r'\s*=\s*', text)
    if not start_match:
        return None
    
    start_index = start_match.end()
    content_start_index = start_index
    while content_start_index < len(text) and text[content_start_index].isspace():
        content_start_index += 1
        
    if content_start_index >= len(text):
        return None

    start_char = text[content_start_index]
    if start_char not in ['[', '{']:
        match = re.search(r'var\s+' + re.escape(var_name) + r'\s*=\s*(.*?)\s*;', text, re.DOTALL)
        return match.group(1).strip() if match else None

    end_char = ']' if start_char == '[' else '}'
    balance = 0
    content_end_index = -1
    
    for i in range(content_start_index, len(text)):
        char = text[i]
        if char == start_char:
            balance += 1
        elif char == end_char:
            balance -= 1
        if balance == 0 and i > content_start_index:
            content_end_index = i
            break
            
    if content_end_index != -1:
        data_str = text[content_start_index : content_end_index + 1].strip()
        # 修复：显式指定 flags 参数，消除 DeprecationWarning
        data_str = re.sub(r'\s*/\*.*$', '', data_str, flags=re.DOTALL).strip()
        data_str = re.sub(r'\s*//.*$', '', data_str, flags=re.MULTILINE).strip()
        logger.debug(f"Successfully extracted variable {var_name}")
        return data_str
        
    logger.error(f"Failed to extract variable {var_name}: Stack counting failed.")
    return None

# 辅助函数：清理和解析 JS 对象
def clean_and_parse_js_object(js_text):
    if not js_text:
        return {}
    
    text = js_text.strip().lstrip('\ufeff')
    try:
        cleaned_js = jsbeautifier.beautify(text)
    except Exception as e:
        logger.warning(f"jsbeautifier failed: {e}. Falling back to raw text.")
        cleaned_js = text

    def replace_single_quotes(match):
        return '"' + match.group(1).replace('"', '\\"') + '"'
    cleaned_js = re.sub(r"'(.*?)'", replace_single_quotes, cleaned_js)
    
    def replace_unquoted_keys(match):
        return match.group(1) + '"' + match.group(2) + '":'
    
    # 关键修复：添加 'cleaned_js' 作为 'string' 参数
    cleaned_js = re.sub(r'([\{\,]\s*)([a-zA-Z_]\w*)\s*:', replace_unquoted_keys, cleaned_js)
    
    cleaned_js = cleaned_js.replace('True', 'true').replace('False', 'false').replace('Null', 'null').replace('NaN', 'null')
    
    try:
        data = json5.loads(cleaned_js)
        return data
    except Exception as e:
        logger.error(f"JSON5 parsing failed: {e}. Text snippet: {cleaned_js[:200]}...")
        return {}

# 异步函数：抓取 JS 页面
async def fetch_js_page(session, url):
    async with session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT) as response:
        if response.status != 200:
            raise aiohttp.ClientClientError(f"HTTP error: {response.status}")
        return await response.text()

# 异步函数：抓取基金指标
async def fetch_fund_metrics(fund_code, session, semaphore):
    logger.info(f"Fetching metrics for fund {fund_code}")
    url = BASE_URL_INFO_JS.format(fund_code=fund_code)
    
    default_result = {
        'fund_code': fund_code,
        'net_worth_trend': [],
        'performance_evaluation': {}
    }
    
    async with semaphore:
        try:
            await asyncio.sleep(REQUEST_DELAY)
            js_text = await fetch_js_page(session, url)
            
            net_worth_str = extract_js_variable_content(js_text, 'Data_netWorthTrend')
            net_worth_data = clean_and_parse_js_object(net_worth_str) if net_worth_str else []
            
            performance_str = extract_js_variable_content(js_text, 'Data_performanceEvaluation')
            performance_data = clean_and_parse_js_object(performance_str) if performance_str else {}
            
            result = {
                'fund_code': fund_code,
                'net_worth_trend': net_worth_data,
                'performance_evaluation': performance_data
            }
            
            logger.info(f"Successfully fetched metrics for fund {fund_code}")
            return fund_code, result
            
        except Exception as e:
            logger.error(f"Failed to fetch metrics for fund {fund_code}: {e}")
            return fund_code, default_result

# 保存数据到 CSV
def save_to_csv(fund_code, data):
    fund_output_dir = os.path.join(OUTPUT_DIR, fund_code)
    os.makedirs(fund_output_dir, exist_ok=True)
    
    try:
        # --- 1. 处理净值走势 (net_worth_trend) ---
        net_worth_df = pd.DataFrame(data['net_worth_trend'])
        if not net_worth_df.empty:
            net_worth_df['date'] = pd.to_datetime(net_worth_df['x'], unit='ms').dt.strftime('%Y-%m-%d')
            net_worth_df = net_worth_df[['date', 'y', 'equityReturn', 'unitMoney']]
            net_worth_df.columns = ['date', 'net_value', 'equity_return', 'unit_money']
            
            net_worth_path = os.path.join(fund_output_dir, f"{fund_code}_net_worth.csv")
            net_worth_df.to_csv(net_worth_path, index=False, encoding='utf-8')
            logger.info(f"Saved net worth trend for fund {fund_code} to {net_worth_path}")
        else:
            logger.warning(f"No net worth trend data for fund {fund_code}")

        # --- 2. 处理业绩评价 (performance_evaluation) ---
        performance_data = data['performance_evaluation']
        if performance_data and performance_data.get('categories') and performance_data.get('data'):
            performance_df = pd.DataFrame({
                'category': performance_data.get('categories', []),
                'score': performance_data.get('data', []),
                'description': performance_data.get('dsc', [])
            })
            
            performance_path = os.path.join(fund_output_dir, f"{fund_code}_performance.csv")
            performance_df.to_csv(performance_path, index=False, encoding='utf-8')
            logger.info(f"Saved performance evaluation for fund {fund_code} to {performance_path}")
        else:
            logger.warning(f"No performance evaluation data for fund {fund_code}")

        return True
    except Exception as e:
        logger.error(f"Failed to save metrics for fund {fund_code}: {e}")
        return False

# 主异步函数：抓取所有基金数据
async def fetch_all_funds(fund_codes):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    async with ClientSession() as session:
        tasks = [fetch_fund_metrics(code, session, semaphore) for code in fund_codes]
        
        for future in asyncio.as_completed(tasks):
            fund_code, result = await future
            
            if result.get('net_worth_trend') or result.get('performance_evaluation'):
                save_to_csv(fund_code, result)
            else:
                logger.warning(f"No valid data fetched for fund {fund_code}")
        
# 主函数
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fund_codes = get_fund_codes_from_csv(FUND_LIST_FILE)
    
    if not fund_codes:
        logger.error(f"No fund codes found in {FUND_LIST_FILE}. Exiting.")
        return
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    loop.run_until_complete(fetch_all_funds(fund_codes))
    logger.info("Fund metrics scraping completed.")

if __name__ == "__main__":
    main()