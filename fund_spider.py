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
import json5 
import logging
import jsbeautifier 
from functools import partial 

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义文件路径和目录
INPUT_FILE = 'C类.txt' 
OUTPUT_DIR = 'fund_data'
# 仅保留净值 API
BASE_URL_NET_VALUE = "http://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page={page_index}&per=20"

# 设置请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'http://fund.eastmoney.com/',
}

REQUEST_TIMEOUT = 30
# 优化 2: 减少基础延迟，提高请求频率 (原 3.5)
REQUEST_DELAY = 3.5 
# 优化 1: 增加最大并发数 (原 5)
MAX_CONCURRENT = 15 
MAX_FUNDS_PER_RUN = 0 #限制0个
PAGE_SIZE = 20

# --------------------------------------------------------------------------------------
# 文件和代码读取 (保留)
# --------------------------------------------------------------------------------------

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
            # 使用 Python 原生文件读取，如果 pandas 慢的话
            with open(file_path, 'r', encoding=encoding) as f:
                codes = [line.strip() for line in f if line.strip()]
            
            df = pd.DataFrame(codes, columns=['code'])
            print(f"  -> 成功使用 {encoding} 编码读取文件，找到 {len(df)} 个基金代码。")
            break
        except UnicodeDecodeError as e:
            # print(f"  -> 使用 {encoding} 编码读取失败: {e}")
            continue
        except Exception as e:
            # print(f"  -> 读取文件时发生错误: {e}")
            continue

    if df is None or df.empty:
        print("[错误] 无法读取文件，请检查文件格式和编码。")
        return []

    codes = df['code'].dropna().astype(str).str.strip().unique().tolist()
    valid_codes = [code for code in codes if re.match(r'^\d{6}$', code)]
    if not valid_codes:
        print("[错误] 没有找到有效的6位基金代码。")
        return []
    print(f"  -> 找到 {len(valid_codes)} 个有效基金代码。")
    return valid_codes

# --------------------------------------------------------------------------------------
# 基金净值抓取核心逻辑 (加速本地 I/O 和网络请求)
# --------------------------------------------------------------------------------------

# 优化 4: 使用 ThreadPoolExecutor 来运行 load_latest_date，加速 I/O 密集型操作
def load_latest_date(fund_code):
    """
    
    如果读取失败，返回 None 触发全量抓取。
    """
    output_path = os.path.join(OUTPUT_DIR, f"{fund_code}.csv")
    if os.path.exists(output_path):
        # 尝试多种编码和日期格式
        encodings_to_try = ['utf-8', 'gbk', 'utf-8-sig'] 
        
        for encoding in encodings_to_try:
            try:
                # 仅读取必要的 'date' 列，加速 Pandas 加载
                # 明确指定日期格式，以提高解析速度和准确性
                df = pd.read_csv(
                    output_path, 
                    usecols=['date'], 
                    # 明确 date_parser，确保日期解析准确
                    # 注意：pd.read_csv在指定parse_dates时，会尝试多种格式，不需要显式指定date_parser
                    parse_dates=['date'], 
                    encoding=encoding
                )
                
                if not df.empty and 'date' in df.columns:
                    # 确保只包含日期对象
                    df.dropna(subset=['date'], inplace=True)
                    if not df.empty:
                        latest_date = df['date'].max().to_pydatetime().date()
                        # 【修改 1】：增加日志输出，确认本地最新日期
                        logger.info(f"  -> 基金 {fund_code} 现有最新日期: {latest_date.strftime('%Y-%m-%d')} (使用 {encoding} 编码)")
                        return latest_date
            except Exception as e:
                # print(f"  -> 加载 {fund_code} CSV 失败 (编码 {encoding}): {e}")
                continue

        logger.warning(f"  -> 基金 {fund_code} [重要警告]：无法准确读取本地 CSV 文件中的最新日期，将从头开始抓取！")
    
    return None

async def fetch_page(session, url):
    """异步请求页面，不带重试，由外部 fetch_net_values 控制"""
    # 增加连接池限制，确保不会突然创建过多连接，但这里 ClientSession 已经处理了
    async with session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT) as response:
        if response.status != 200:
            # 检查是否为频率限制（514通常是）
            if response.status == 514:
                raise aiohttp.ClientError("Frequency Capped (HTTP 514)")
            raise aiohttp.ClientError(f"HTTP 错误: {response.status}")
        return await response.text()

async def fetch_net_values(fund_code, session, semaphore, executor):
    """使用“最新日期”作为停止条件，实现智能增量更新"""
    print(f"-> [START] 基金代码 {fund_code}")
    
    async with semaphore:
        all_records = []
        dynamic_delay = REQUEST_DELAY
        
        # 优化 4: 使用线程池异步加载最新日期
        latest_date = await asyncio.get_event_loop().run_in_executor(executor, load_latest_date, fund_code)
        
        # 【新增】：先抓取第一页，获取 API 最新日期和总页数
        url_page1 = BASE_URL_NET_VALUE.format(fund_code=fund_code, page_index=1)
        try:
            text = await fetch_page(session, url_page1)
            soup = BeautifulSoup(text, 'lxml')
            table = soup.find('table')
            if not table:
                logger.warning(f"    基金 {fund_code} [警告]：第一页无表格数据。")
                return fund_code, "API返回记录数为0或代码无效"
            
            rows = table.find_all('tr')[1:]
            if not rows:
                logger.info(f"    基金 {fund_code} 第一页无数据行。停止抓取。")
                return fund_code, "API返回记录数为0或代码无效"
            
            # 解析总页数和记录数
            total_pages_match = re.search(r'pages:(\d+)', text)
            total_pages = int(total_pages_match.group(1)) if total_pages_match else 1
            records_match = re.search(r'records:(\d+)', text)
            total_records = int(records_match.group(1)) if records_match else 0
            logger.info(f"    基金 {fund_code} 信息：总页数 {total_pages}，总记录数 {total_records}。")
            
            if total_records == 0:
                return fund_code, "API返回记录数为0或代码无效"
            
            # 获取 API 最新日期（第一页第一行）
            first_tds = rows[0].find_all('td')
            latest_api_date_str = first_tds[0].text.strip()
            latest_api_date = datetime.strptime(latest_api_date_str, '%Y-%m-%d').date()
            
            # 【新增比较】：如果 API 最新日期 <= 本地最新日期，跳过更新
            if latest_date and latest_api_date <= latest_date:
                logger.info(f"    基金 {fund_code} 数据已是最新 ({latest_date.strftime('%Y-%m-%d')})，API 最新为 {latest_api_date.strftime('%Y-%m-%d')}，无新数据。")
                return fund_code, f"数据已是最新 ({latest_date.strftime('%Y-%m-%d')})，无新数据"
            
            # 如果本地落后，继续增量抓取所有页，但只收集 date > latest_date 的记录
            page_index = 1
            stop_fetch = False
            while page_index <= total_pages and not stop_fetch:
                if page_index > 1:
                    url = BASE_URL_NET_VALUE.format(fund_code=fund_code, page_index=page_index)
                    await asyncio.sleep(dynamic_delay)
                    text = await fetch_page(session, url)
                    soup = BeautifulSoup(text, 'lxml')
                    table = soup.find('table')
                    if not table:
                        logger.info(f"    基金 {fund_code} [停止]：第 {page_index} 页无表格数据。提前停止。")
                        break
                    rows = table.find_all('tr')[1:]
                    if not rows:
                        logger.info(f"    基金 {fund_code} 第 {page_index} 页无数据行。停止抓取。")
                        break
                
                # 解析当前页记录
                page_records = []
                for row in rows:
                    tds = row.find_all('td')
                    if len(tds) < 7:
                        continue
                    date_str = tds[0].text.strip()
                    try:
                        row_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    except ValueError:
                        continue
                    
                    # 如果 latest_date 存在，且当前日期 <= latest_date，停止抓取该页及后续页
                    if latest_date and row_date <= latest_date:
                        stop_fetch = True
                        break
                    
                    # 收集新记录
                    net_value = tds[1].text.strip()
                    cumulative_net_value = tds[2].text.strip()
                    daily_growth_rate = tds[3].text.strip()
                    purchase_status = tds[4].text.strip()
                    redemption_status = tds[5].text.strip()
                    dividend = tds[6].text.strip() if len(tds) > 6 else ''
                    
                    page_records.append({
                        'date': date_str,
                        'net_value': net_value,
                        'cumulative_net_value': cumulative_net_value,
                        'daily_growth_rate': daily_growth_rate,
                        'purchase_status': purchase_status,
                        'redemption_status': redemption_status,
                        'dividend': dividend
                    })
                
                all_records.extend(page_records)
                page_index += 1
            
            # 如果收集到新记录，返回它们
            if all_records:
                logger.info(f"    基金 {fund_code} 抓取到 {len(all_records)} 条新记录。")
                return fund_code, all_records
            else:
                # 无新数据的情况
                if latest_api_date and latest_date and latest_api_date == latest_date:
                    return fund_code, f"数据已是最新 ({latest_date.strftime('%Y-%m-%d')})，无新数据"
                else:
                    return fund_code, "未获取到新数据（可能是API未更新或基金已停售）"
        except Exception as e:
            logger.error(f"    基金 {fund_code} 抓取失败: {e}")
            return fund_code, f"抓取失败: {str(e)}"

def save_to_csv(fund_code, data):
    """
 
    """
    output_path = os.path.join(OUTPUT_DIR, f"{fund_code}.csv")
    if not isinstance(data, list) or not data:
        # print(f"    基金 {fund_code} 无新数据可保存。")
        return False, 0

    new_df = pd.DataFrame(data)

    try:
        # 数据类型转换和清洗
        new_df['net_value'] = pd.to_numeric(new_df['net_value'], errors='coerce').round(4)
        new_df['cumulative_net_value'] = pd.to_numeric(new_df['cumulative_net_value'], errors='coerce').round(4)
        
        # 【修改 2】：使用更健壮的 apply 函数处理 daily_growth_rate
        def clean_growth_rate(rate_str):
            rate_str = str(rate_str).strip()
            if rate_str in ['--', '']:
                return 0.0
            try:
                # 去除 % 并转换为浮点数，然后除以 100
                return float(rate_str.rstrip('%')) / 100.0
            except ValueError:
                return None # 标记为无效数据
        
        new_df['daily_growth_rate'] = new_df['daily_growth_rate'].apply(clean_growth_rate)
        
        new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce', format='%Y-%m-%d')
        
        # 丢弃无效的核心记录
        new_df.dropna(subset=['date', 'net_value', 'daily_growth_rate'], inplace=True)
        if new_df.empty:
            # print(f"    基金 {fund_code} 数据无效或为空，跳过保存。")
            return False, 0
    except Exception as e:
        print(f"    基金 {fund_code} 数据转换失败: {e}")
        return False, 0
    
    old_record_count = 0
    if os.path.exists(output_path):
        try:
            # 读取老数据时，同样使用 robust 的日期解析，确保一致性
            existing_df = pd.read_csv(
                output_path, 
                parse_dates=['date'], 
                # 这里不需要显式的 date_parser，pd 默认的行为已经很好了
                encoding='utf-8' # 保持与保存时一致的编码
            )
            old_record_count = len(existing_df)
            combined_df = pd.concat([new_df, existing_df])
        except Exception as e:
            print(f"    读取现有 CSV 文件 {output_path} 失败: {e}。仅保存新数据。")
            combined_df = new_df
    else:
        combined_df = new_df
        
    # 去重：以日期为准，保留最新的记录 (新数据在前，所以 keep='first')
    final_df = combined_df.drop_duplicates(subset=['date'], keep='first')
    # 排序：按日期降序
    final_df = final_df.sort_values(by='date', ascending=False)
    # 格式化日期为字符串，以便保存
    final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
    
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # 写入时使用 UTF-8 编码
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
    
    # 优化 4: 创建线程池执行器，用于加速本地 I/O 操作 (读取和写入 CSV)
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2 + 1) as executor:

        # aiohttp 连接器设置：限制并发连接数，与 semaphore 配合
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5) 

        async with ClientSession(connector=connector) as session:
            # 传递 executor 给 fetch_net_values
            fetch_tasks = [fetch_net_values(fund_code, session, semaphore, executor) for fund_code in fund_codes]
            
            success_count = 0
            total_new_records = 0
            # 使用集合存储，避免重复记录失败的基金代码
            failed_codes = set() 
            
            # 使用 loop.run_in_executor 包装 save_to_csv
            save_executor = partial(loop.run_in_executor, executor, save_to_csv)

            for future in asyncio.as_completed(fetch_tasks):
                print("-" * 30)
                fund_code = "UNKNOWN"
                try:
                    # 统一捕获 fetch_net_values 的返回
                    result = await future
                    if isinstance(result, tuple) and len(result) == 2:
                        fund_code, net_values = result
                    else:
                        raise Exception("Fetch task returned unexpected format.")
                except Exception as e:
                    # 捕获任务执行中的异常
                    print(f"[错误] 处理基金数据时发生顶级异步错误: {e}")
                    # 如果 fund_code 无法确定，则无法加入失败列表，但可以记录异常
                    failed_codes.add(fund_code if fund_code != "UNKNOWN" else "UNKNOWN_ERROR")
                    continue


                if isinstance(net_values, list):
                    try:
                        # 优化 4: 在线程池中执行耗时的保存操作，避免阻塞主循环
                        success, new_records = await save_executor(fund_code, net_values) 
                        if success:
                            success_count += 1
                            total_new_records += new_records
                        else:
                            failed_codes.add(fund_code)
                    except Exception as e:
                        print(f"[错误] 基金 {fund_code} 的保存任务在线程中发生错误: {e}")
                        failed_codes.add(fund_code)
                else:
                    # 抓取失败或跳过 (数据已最新)
                    if not str(net_values).startswith('数据已是最新'):
                        failed_codes.add(fund_code)

        return success_count, total_new_records, list(failed_codes) # 转换为列表返回

def main():
    """主函数：现在只执行动态净值抓取"""
    print(f"加速设置：并发数={MAX_CONCURRENT}，基础延迟={REQUEST_DELAY}秒")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"确保输出目录存在: {OUTPUT_DIR}")

    start_time = time.time()
    
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
        # Windows 环境下推荐使用 ProactorEventLoop
        if os.name == 'nt':
            try:
                loop = asyncio.ProactorEventLoop()
                asyncio.set_event_loop(loop)
            except Exception:
                # 兼容旧版本 Python/Windows
                loop = asyncio.get_event_loop()
        else:
             loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    success_count, total_new_records, failed_codes = loop.run_until_complete(fetch_all_funds(processed_codes))
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n======== 本次更新总结 ========")
    print(f"总耗时: {duration:.2f} 秒")
    print(f"成功处理 {success_count} 个基金，新增/更新 {total_new_records} 条记录，失败 {len(failed_codes)} 个基金。")
    if failed_codes:
        print(f"失败的基金代码: {', '.join(failed_codes)}")
    if total_new_records == 0:
        print("[警告] 未新增任何记录，可能是数据已是最新，或 API 无新数据。")
    print(f"==============================")

if __name__ == "__main__":
    main()
