import pandas as pd
import requests
import os
import time
import asyncio
import aiohttp
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import re
import math
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_message
import concurrent.futures
# V7 核心依赖：使用 json5 代替内置 json
import json5 
import logging
import jsbeautifier

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义文件路径和目录
INPUT_FILE = 'C类.txt' # 请确保您的基金代码文件名为 C类.txt
OUTPUT_DIR = 'fund_data'
BASE_URL_NET_VALUE = "http://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page={page_index}&per=20"
BASE_URL_INFO_JS = "http://fund.eastmoney.com/pingzhongdata/{fund_code}.js"
INFO_CACHE_FILE = 'fund_info.csv'

# 设置请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'http://fund.eastmoney.com/',
}

REQUEST_TIMEOUT = 30
REQUEST_DELAY = 3.5
MAX_CONCURRENT = 5
MAX_FUNDS_PER_RUN = 0 
PAGE_SIZE = 20

# --------------------------------------------------------------------------------------
# 辅助函数：JS变量提取和解析 (V7 - 引入 JSON5)
# --------------------------------------------------------------------------------------

def extract_simple_var(text, var_name):
    """提取简单的字符串或数字变量，如 fS_name, fund_Rate"""
    match = re.search(r'var\s+' + re.escape(var_name) + r'\s*=\s*(.*?);', text, re.DOTALL)
    if match:
        value = match.group(1).strip()
        # 清理引号和 BOM
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1].lstrip('\ufeff')
        return value.lstrip('\ufeff')
    return None

def extract_js_variable_content_v7(text, var_name):
    """
    V7 核心：使用栈计数提取复杂的 JS 变量内容 (内容本身)。
    """
    # 1. 定位到变量赋值的起始位置
    start_match = re.search(r'var\s+' + re.escape(var_name) + r'\s*=\s*', text)
    if not start_match:
        return None
    
    start_index = start_match.end()
    
    # 2. 找到变量内容实际开始的位置（跳过空格，定位到 [ 或 {）
    content_start_index = start_index
    while content_start_index < len(text) and text[content_start_index].isspace():
        content_start_index += 1
        
    if content_start_index >= len(text):
        return None

    start_char = text[content_start_index] # 应该是 [ 或 {
    
    if start_char not in ['[', '{']:
        # 如果不是标准的对象或数组开头，尝试回退到宽泛正则
        match = re.search(r'var\s+' + re.escape(var_name) + r'\s*=\s*(.*?)\s*;', text, re.DOTALL)
        return match.group(1).strip() if match else None

    # --- 使用栈计数法 (确保提取的结构完整性) ---
    end_char = ']' if start_char == '[' else '}'
    balance = 0
    content_end_index = -1
    
    # 增加一个检查，确保我们只在最外层计数
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
        
        # 清理尾部可能的注释 /*...*/ 或 //...
        data_str = re.sub(r'\s*/\*.*$', '', data_str, re.DOTALL).strip()
        data_str = re.sub(r'\s*//.*$', '', data_str, re.MULTILINE).strip()
        
        logger.info(f"成功使用栈计数提取变量 {var_name}")
        return data_str
    
    logger.error(f"变量 {var_name} 提取失败：栈计数失败。")
    return None

def clean_and_parse_js_object_v7(js_text):
    """
    V7 核心解析函数：使用 jsbeautifier 清理 JS 对象字面量，并使用 json5 解析。
    """
    if not js_text:
        return {}
    
    text = js_text.strip()
    
    # 1. 移除 BOM (Byte Order Mark)
    text = text.lstrip('\ufeff')
    
    # 2. **核心步骤：** 使用 jsbeautifier 格式化/清理 JS 文本
    try:
        # 尝试美化，移除注释等
        cleaned_js = jsbeautifier.beautify(text)
    except Exception as e:
        logger.warning(f"jsbeautifier 美化失败: {e}. 尝试不美化进行手动清理。")
        cleaned_js = text # 回退到原始文本

    # 3. **激进替换：** 将 JS 单引号字符串替换为 JSON 双引号字符串
    # 确保内部的双引号被转义
    def replace_single_quotes(match):
        return '"' + match.group(1).replace('"', '\\"') + '"'
    
    # 匹配 '...' 格式的字符串
    cleaned_js = re.sub(r"'(.*?)'", replace_single_quotes, cleaned_js)
    
    # 4. **核心修复：** 使用正则将无引号的键名替换为带双引号的键名 ("key":)
    def replace_unquoted_keys(match):
        # match.group(1) 是定界符 { 或 ,
        # match.group(2) 是键名 (字母数字下划线开头)
        return match.group(1) + '"' + match.group(2) + '":'
    
    # 匹配 { 或 , 之后跟着一个合法的 JS 标识符作为键名
    # 注意：json5 理论上能处理这个，但我们在这里提前修正，以提高鲁棒性。
    cleaned_js = re.sub(r'([\{\,]\s*)([a-zA-Z_]\w*)\s*:', replace_unquoted_keys, cleaned_js)
    
    # 5. 替换 JS 的 true/false/null 为标准 JSON 的 true/false/null (小写)
    # json5 也能处理，但确保万无一失
    cleaned_js = cleaned_js.replace('True', 'true').replace('False', 'false').replace('Null', 'null')
    
    # 6. 最终解析 **(V7 关键)**
    try:
        # 使用 json5.loads 进行最宽容的解析
        data = json5.loads(cleaned_js)
        
        # 处理 Data_fund_info 的常见格式：数组包含单个对象 [ {...} ]
        if isinstance(data, list) and data and isinstance(data[0], dict):
             return data[0]

        return data
        
    except Exception as e:
        # 捕获所有解析错误
        logger.error(f"最终 JSON5 解析失败: {e}. 请检查数据结构。文本片段: {cleaned_js[:100]}...")
        return {}


# --------------------------------------------------------------------------------------
# 基金信息抓取核心逻辑 (V7 - 采用多变量自适应提取策略)
# --------------------------------------------------------------------------------------

async def fetch_fund_info(fund_code, session, semaphore):
    """异步抓取基金基本信息，使用 V7 多变量自适应提取逻辑"""
    print(f"    -> 开始抓取基金 {fund_code} 的基本信息")
    url = BASE_URL_INFO_JS.format(fund_code=fund_code)
    
    # 默认值
    default_result = {
        '代码': fund_code, '名称': '抓取失败', '类型': '未知', '成立日期': '未知', '基金经理': '未知', 
        '公司名称': '未知', '基金简称': '未知', '基金类型': '未知', '发行时间': '未知', '估值日期': '未知', 
        '估值涨幅': '未知', '累计净值': '未知', '单位净值': '未知', '申购净值': '未知', '申购报价': '未知', 
        '净值日期': '未知', '申购状态': '未知', '赎回状态': '未知', '费率': '未知', '申购步长': '未知'
    }
    
    async with semaphore:
        try:
            await asyncio.sleep(REQUEST_DELAY * 0.5)
            # 使用 V7 的提取器
            js_text = await fetch_js_page(session, url)
            
            # 1. 提取简单变量
            fund_name = extract_simple_var(js_text, 'fS_name') or '未知名称'
            fund_code_in_data = extract_simple_var(js_text, 'fS_code') or fund_code
            fund_type_raw = extract_simple_var(js_text, 'FTyp')
            fund_rate = extract_simple_var(js_text, 'fund_Rate')
            fund_minsg = extract_simple_var(js_text, 'fund_minsg')
            
            # 初始化数据源字典
            data_info = {}
            data_main = {}
            data_manager = {}
            
            # 2. **V7 核心：** 尝试从所有可能的变量中提取数据 (使用 V7 解析器)
            
            # A. 提取并解析 Data_managerInfo (经理信息最可靠来源)
            data_manager_str = extract_js_variable_content_v7(js_text, 'Data_managerInfo')
            if data_manager_str:
                 data_manager = clean_and_parse_js_object_v7(data_manager_str)
            
            # B. 提取并解析 apidata (估值、公司、成立日期等信息)
            data_main_str = extract_js_variable_content_v7(js_text, 'apidata')
            if data_main_str:
                 data_main = clean_and_parse_js_object_v7(data_main_str)

            # C. 提取并解析 Data_fund_info (传统信息来源，作为补充)
            data_info_str = extract_js_variable_content_v7(js_text, 'Data_fund_info')
            if data_info_str:
                 data_info = clean_and_parse_js_object_v7(data_info_str)


            # 3. **V7 核心：** 整合最终结果，按优先级合并信息
            
            manager_name = '未知'
            # 优先级1: Data_managerInfo (数组)
            if isinstance(data_manager, list) and data_manager and isinstance(data_manager[0], dict):
                 manager_name = data_manager[0].get('name', '未知')
            # 优先级2: apidata (有时包含单个经理名)
            elif data_main.get('jjjl') and isinstance(data_main['jjjl'], str):
                 manager_name = data_main['jjjl']
            # 优先级3: Data_fund_info (旧结构)
            elif 'FundManager' in data_info:
                 if isinstance(data_info['FundManager'], list) and data_info['FundManager']:
                     # 有些是 {'name': '张三'}，有些是 ['张三']
                     if isinstance(data_info['FundManager'][0], dict):
                         manager_name = data_info['FundManager'][0].get('name', '未知')
                     elif isinstance(data_info['FundManager'][0], str):
                         manager_name = data_info['FundManager'][0]

            
            # 公司名称：apidata > Data_fund_info
            company_name = data_main.get('jjgs', data_info.get('FundCompany', '未知'))
            # 成立日期：apidata > Data_fund_info
            establish_date = data_main.get('qjdate', data_info.get('EstablishDate', '未知'))

            result = {
                '代码': fund_code_in_data,
                '名称': fund_name,
                
                '基金经理': manager_name,
                '公司名称': company_name,
                '成立日期': establish_date,
                
                '基金简称': data_info.get('jjjc', fund_name.split('(')[0].strip() if '(' in fund_name else fund_name.strip()),
                '基金类型': data_info.get('FundType', fund_type_raw or data_main.get('FTyp', '未知')),
                '发行时间': data_info.get('IssueDate', '未知'),
                '类型': data_info.get('FundType', fund_type_raw or data_main.get('FTyp', '未知')),
                
                '估值日期': data_main.get('gzrq', '未知'),
                '估值涨幅': data_main.get('gszzl', '未知'),
                '累计净值': data_main.get('ljjz', '未知'),
                '单位净值': data_main.get('dwjz', '未知'),
                '净值日期': data_main.get('jzrq', '未知'),
                '申购状态': data_main.get('sgzt', '未知'),
                '赎回状态': data_main.get('shzt', '未知'),
                
                '费率': fund_rate if fund_rate else '未知',
                '申购步长': fund_minsg if fund_minsg else '未知',
                
                '申购净值': '未知',
                '申购报价': '未知',
            }
            
            # 检查核心字段是否成功提取
            if result['基金经理'] != '未知' and result['公司名称'] != '未知' and result['名称'] != '抓取失败':
                 print(f"    -> 基金 {fund_code} 基本信息抓取成功 (名称: {result['名称']} | 公司: {result['公司名称']} | 经理: {result['基金经理']})")
            else:
                 print(f"    -> 基金 {fund_code} 基本信息抓取完成，但核心字段缺失 (经理: {result['基金经理']} | 公司: {result['公司名称']} | 名称: {result['名称']})")

            return fund_code, result

        except Exception as e:
            error_message = str(e)
            print(f"    -> 基金 {fund_code} [信息抓取失败]: {error_message}")
            logger.error(f"基金 {fund_code} 抓取失败的详情: {e}")
            return fund_code, default_result


# 以下代码与 V6 保持一致，无需更改（包括文件读取、缓存、净值抓取等逻辑）...

def get_all_fund_codes(file_path):
    """从 C类.txt 文件中读取基金代码"""
    print(f"尝试读取基金代码文件: {file_path}")
    if not os.path.exists(file_path):
        print(f"[错误] 文件 {file_path} 不存在。")
        return []

    if os.path.getsize(file_path) == 0:
        print(f"[错误] 文件 {file_path} 为空。")
        return []

    encodings_to_try = ['utf-8', 'utf-8-sig', 'gbk', 'latin-1']
    df = None

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding, header=None, dtype=str)
            df.columns = ['code']
            print(f"  -> 成功使用 {encoding} 编码读取文件，找到 {len(df)} 个基金代码。")
            break
        except UnicodeDecodeError as e:
            print(f"  -> 使用 {encoding} 编码读取失败: {e}")
            continue
        except Exception as e:
            print(f"  -> 读取文件时发生错误: {e}")
            continue

    if df is None:
        print("[错误] 无法读取文件，请检查文件格式和编码。")
        return []

    codes = df['code'].dropna().astype(str).str.strip().unique().tolist()
    valid_codes = [code for code in codes if re.match(r'^\d{6}$', code)]
    if not valid_codes:
        print("[错误] 没有找到有效的6位基金代码。")
        return []
    print(f"  -> 找到 {len(valid_codes)} 个有效基金代码。")
    return valid_codes

def load_info_cache():
    """加载基金基本信息缓存 (从 CSV 文件)"""
    if not os.path.exists(INFO_CACHE_FILE):
        print(f"[信息] 缓存文件 {INFO_CACHE_FILE} 不存在，将创建新缓存。")
        return {}

    try:
        file_to_load = INFO_CACHE_FILE
        
        df = pd.read_csv(file_to_load, dtype={'代码': str}, encoding='utf-8')
        if df.empty:
            print(f"[警告] 缓存文件 {file_to_load} 为空。")
            return {}
        df.set_index('代码', inplace=True)
        info_cache = df.to_dict('index')
        print(f"  -> 成功从 {file_to_load} 加载 {len(info_cache)} 条缓存记录。")
        return info_cache
    except Exception as e:
        print(f"[警告] 加载基本信息缓存失败: {e}。将从头抓取。")
        return {}

def save_info_cache(cache):
    """保存基金基本信息缓存 (到 CSV 文件)"""
    if not cache:
        print("[警告] 基本信息缓存为空，不保存。")
        return

    try:
        df = pd.DataFrame.from_dict(cache, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': '代码'}, inplace=True)
        
        # 固定列顺序，使用中文列名
        cols = ['代码', '名称', '类型', '成立日期', '基金经理', '公司名称', '基金简称', '基金类型', '发行时间', '估值日期', '估值涨幅', '累计净值', '单位净值', '申购净值', '申购报价', '净值日期', '申购状态', '赎回状态', '费率', '申购步长']
        for col in cols:
            if col not in df.columns:
                df[col] = '未知'
        df = df[cols]
        
        df['代码'] = df['代码'].astype(str)
        df.to_csv(INFO_CACHE_FILE, index=False, encoding='utf-8')
        print(f"  -> 成功保存基本信息缓存到 {INFO_CACHE_FILE}")
    except Exception as e:
        print(f"[错误] 保存基本信息缓存失败: {e}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_message(match='Frequency Capped')
)
async def fetch_js_page(session, url):
    """异步请求 JS 页面，带重试机制"""
    async with session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT) as response:
        if response.status == 514:
            raise aiohttp.ClientError("Frequency Capped")
        if response.status != 200:
            print(f"    -> HTTP 状态码: {response.status}")
            text = await response.text()
            print(f"    -> 响应内容（前100字符）: {text[:100]}")
            raise aiohttp.ClientError(f"HTTP 错误: {response.status}")
        return await response.text()

async def fetch_and_cache_fund_info(fund_codes):
    """主函数：检查缓存，对未缓存的基金进行并发抓取"""
    print("\n======== 开始抓取基金基本信息（静态数据）========\n")
    
    loop = asyncio.get_event_loop()
    info_cache = await loop.run_in_executor(None, load_info_cache)
    
    # 重新抓取条件：代码缺失或名称/经理/公司为 '未知' 或 '抓取失败'
    codes_to_fetch = [
        code for code in fund_codes if code not in info_cache or 
        info_cache[code].get('名称') in ['未知名称', '抓取失败'] or 
        info_cache[code].get('基金经理') == '未知' or 
        info_cache[code].get('公司名称') == '未知'
    ]
    
    if not codes_to_fetch:
        print("所有基金的基本信息都已在缓存中。")
        return info_cache

    print(f"发现 {len(codes_to_fetch)} 个基金信息缺失/需要重新抓取，开始抓取...")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    async with ClientSession() as session:
        fetch_tasks = [fetch_fund_info(code, session, semaphore) for code in codes_to_fetch]
        
        # 实时更新缓存，而不是等待所有任务完成
        for future in asyncio.as_completed(fetch_tasks):
            code, result = await future
            info_cache[code] = result
            
    await loop.run_in_executor(None, save_info_cache, info_cache)
    print("\n======== 基金基本信息抓取完成 ========\n")
    return info_cache

def load_latest_date(fund_code):
    """从本地 CSV 文件中读取现有最新日期"""
    output_path = os.path.join(OUTPUT_DIR, f"{fund_code}.csv")
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path, parse_dates=['date'], encoding='utf-8')
            if not df.empty and 'date' in df.columns:
                latest_date = df['date'].max().to_pydatetime().date()
                logger.info(f"  -> 基金 {fund_code} 现有最新日期: {latest_date.strftime('%Y-%m-%d')}")
                return latest_date
        except Exception as e:
            logger.warning(f"  -> 加载 {fund_code} CSV 失败: {e}")
    return None

async def fetch_page(session, url):
    """异步请求页面，不带重试，由外部 fetch_net_values 控制"""
    async with session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT) as response:
        if response.status != 200:
            raise aiohttp.ClientError(f"HTTP 错误: {response.status}")
        return await response.text()

async def fetch_net_values(fund_code, session, semaphore):
    """使用“最新日期”作为停止条件，实现智能增量更新"""
    print(f"-> [START] 基金代码 {fund_code}")
    
    async with semaphore:
        all_records = []
        page_index = 1
        total_pages = 1
        first_run = True
        dynamic_delay = REQUEST_DELAY
        
        loop = asyncio.get_event_loop()
        latest_date = await loop.run_in_executor(None, load_latest_date, fund_code)

        while page_index <= total_pages:
            url = BASE_URL_NET_VALUE.format(fund_code=fund_code, page_index=page_index)
            
            try:
                if page_index > 1:
                    await asyncio.sleep(dynamic_delay)
                
                text = await fetch_page(session, url)
                
                if first_run:
                    total_pages_match = re.search(r'pages:(\d+)', text)
                    total_pages = int(total_pages_match.group(1)) if total_pages_match else 1
                    records_match = re.search(r'records:(\d+)', text)
                    total_records = int(records_match.group(1)) if records_match else '未知'
                    logger.info(f"    基金 {fund_code} 信息：总页数 {total_pages}，总记录数 {total_records}。")
                    
                    if total_records == '未知' or (isinstance(total_records, int) and total_records == 0):
                        logger.warning(f"    基金 {fund_code} [跳过]：API 返回总记录数为 0 或未知。")
                        return fund_code, "API返回记录数为0或代码无效"
                        
                    first_run = False
                
                soup = BeautifulSoup(text, 'lxml')
                table = soup.find('table')
                if not table:
                    logger.warning(f"    基金 {fund_code} [警告]：页面 {page_index} 无表格数据。提前停止。")
                    break

                rows = table.find_all('tr')[1:]
                if not rows:
                    logger.info(f"    基金 {fund_code} 第 {page_index} 页无数据行。停止抓取。")
                    break

                page_records = []
                stop_fetch = False
                latest_api_date = None
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) < 7:
                        continue
                        
                    date_str = cols[0].text.strip()
                    net_value_str = cols[1].text.strip()
                    cumulative_net_value = cols[2].text.strip()
                    daily_growth_rate = cols[3].text.strip()
                    purchase_status = cols[4].text.strip()
                    redemption_status = cols[5].text.strip()
                    dividend = cols[6].text.strip()
                    
                    if not date_str or not net_value_str:
                        continue
                        
                    try:
                        date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        if not latest_api_date or date > latest_api_date:
                            latest_api_date = date
                        
                        if latest_date and date <= latest_date:
                            stop_fetch = True
                            break
                            
                        page_records.append({
                            'date': date_str,
                            'net_value': net_value_str,
                            'cumulative_net_value': cumulative_net_value,
                            'daily_growth_rate': daily_growth_rate,
                            'purchase_status': purchase_status,
                            'redemption_status': redemption_status,
                            'dividend': dividend
                        })
                    except ValueError:
                        continue
                
                if latest_api_date:
                    logger.info(f"    基金 {fund_code} API 返回最新日期: {latest_api_date.strftime('%Y-%m-%d')}")
                
                all_records.extend(page_records)
                
                if stop_fetch:
                    print(f"    基金 {fund_code} [增量停止]：页面 {page_index} 遇到旧数据 ({latest_date.strftime('%Y-%m-%d')})，停止抓取。")
                    break
                
                page_index += 1
                dynamic_delay = max(REQUEST_DELAY, dynamic_delay * 0.9)

            except aiohttp.ClientError as e:
                if "Frequency Capped" in str(e):
                    dynamic_delay = min(dynamic_delay * 2, 5.0)
                    print(f"    基金 {fund_code} [警告]：频率限制，延迟调整为 {dynamic_delay} 秒，重试第 {page_index} 页")
                    continue
                print(f"    基金 {fund_code} [错误]：请求 API 时发生网络错误 (超时/连接) 在第 {page_index} 页: {e}")
                return fund_code, f"网络错误: {e}"
            except Exception as e:
                print(f"    基金 {fund_code} [错误]：处理数据时发生意外错误在第 {page_index} 页: {e}")
                return fund_code, f"数据处理错误: {e}"
            
        print(f"-> [COMPLETE] 基金 {fund_code} 数据抓取完毕，共获取 {len(all_records)} 条新记录。")
        if not all_records:
            return fund_code, "数据已是最新，无新数据"
        return fund_code, all_records

def save_to_csv(fund_code, data):
    """将历史净值数据以增量更新方式保存为 CSV 文件"""
    output_path = os.path.join(OUTPUT_DIR, f"{fund_code}.csv")
    if not isinstance(data, list) or not data:
        print(f"    基金 {fund_code} 无新数据可保存。")
        return False, 0

    new_df = pd.DataFrame(data)

    try:
        new_df['net_value'] = pd.to_numeric(new_df['net_value'], errors='coerce').round(4)
        new_df['cumulative_net_value'] = pd.to_numeric(new_df['cumulative_net_value'], errors='coerce').round(4)
        new_df['daily_growth_rate'] = new_df['daily_growth_rate'].replace('--', '0').str.rstrip('%').astype(float) / 100.0
        new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
        new_df.dropna(subset=['date', 'net_value'], inplace=True)
        if new_df.empty:
            print(f"    基金 {fund_code} 数据无效或为空，跳过保存。")
            return False, 0
    except Exception as e:
        print(f"    基金 {fund_code} 数据转换失败: {e}")
        return False, 0
    
    old_record_count = 0
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path, parse_dates=['date'], encoding='utf-8')
            old_record_count = len(existing_df)
            combined_df = pd.concat([new_df, existing_df])
        except Exception as e:
            print(f"    读取现有 CSV 文件 {output_path} 失败: {e}。仅保存新数据。")
            combined_df = new_df
    else:
        combined_df = new_df
        
    final_df = combined_df.drop_duplicates(subset=['date'], keep='first')
    final_df = final_df.sort_values(by='date', ascending=False)
    final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
    
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final_df.to_csv(output_path, index=False, encoding='utf-8')
        new_record_count = len(final_df)
        newly_added = new_record_count - old_record_count
        print(f"    -> 基金 {fund_code} [保存完成]：总记录数 {new_record_count} (新增 {max(0, newly_added)} 条)。")
        return True, max(0, newly_added)
    except Exception as e:
        print(f"    基金 {fund_code} 保存 CSV 文件 {output_path} 失败: {e}")
        return False, 0

async def fetch_all_funds(fund_codes):
    """异步获取所有基金数据，并在任务完成时立即保存数据"""
    print("\n======== 开始基金净值数据抓取（动态数据）========\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    loop = asyncio.get_event_loop()

    async with ClientSession() as session:
        fetch_tasks = [fetch_net_values(fund_code, session, semaphore) for fund_code in fund_codes]
        
        success_count = 0
        total_new_records = 0
        failed_codes = []
        
        for future in asyncio.as_completed(fetch_tasks):
            print("-" * 30)
            fund_code = "UNKNOWN"
            try:
                fund_code, net_values = await future
            except Exception as e:
                print(f"[错误] 处理基金数据时发生顶级异步错误: {e}")
                failed_codes.append("UNKNOWN")
                continue

            if isinstance(net_values, list):
                try:
                    success, new_records = await loop.run_in_executor(None, save_to_csv, fund_code, net_values)
                    if success:
                        success_count += 1
                        total_new_records += new_records
                    else:
                        failed_codes.append(fund_code)
                except Exception as e:
                    print(f"[错误] 基金 {fund_code} 的保存任务在线程中发生错误: {e}")
                    failed_codes.append(fund_code)
            else:
                print(f"    基金 {fund_code} [抓取失败/跳过]：{net_values}")
                if not str(net_values).startswith('数据已是最新'):
                    failed_codes.append(fund_code)

        return success_count, total_new_records, failed_codes

def main():
    """主函数：集成静态信息抓取和动态净值抓取"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"确保输出目录存在: {OUTPUT_DIR}")

    fund_codes = get_all_fund_codes(INPUT_FILE)
    if not fund_codes:
        print("[错误] 没有可处理的基金代码，脚本结束。")
        return

    if MAX_FUNDS_PER_RUN > 0 and len(fund_codes) > MAX_FUNDS_PER_RUN:
        print(f"限制本次运行最多处理 {MAX_FUNDS_PER_RUN} 个基金。")
        processed_codes = fund_codes[:MAX_FUNDS_PER_RUN]
    else:
        processed_codes = fund_codes
    print(f"本次处理 {len(processed_codes)} 个基金。")

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(fetch_and_cache_fund_info(processed_codes))
    success_count, total_new_records, failed_codes = loop.run_until_complete(fetch_all_funds(processed_codes))
    
    print(f"\n======== 本次更新总结 ========")
    print(f"成功处理 {success_count} 个基金，新增/更新 {total_new_records} 条记录，失败 {len(set(failed_codes))} 个基金。")
    if failed_codes:
        print(f"失败的基金代码: {', '.join(set(failed_codes))}")
    if total_new_records == 0:
        print("[警告] 未新增任何记录，可能是数据已是最新，或 API 无新数据。")
    print(f"==============================")

if __name__ == "__main__":
    main()
