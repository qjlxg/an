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
from functools import partial # 引入 partial 用于线程池执行

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
MAX_FUNDS_PER_RUN = 0 
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
    """从本地 CSV 文件中读取现有最新日期"""
    output_path = os.path.join(OUTPUT_DIR, f"{fund_code}.csv")
    if os.path.exists(output_path):
        try:
            # 仅读取必要的列，加速 Pandas 加载
            df = pd.read_csv(output_path, usecols=['date'], parse_dates=['date'], encoding='utf-8')
            if not df.empty and 'date' in df.columns:
                latest_date = df['date'].max().to_pydatetime().date()
                # logger.info(f"  -> 基金 {fund_code} 现有最新日期: {latest_date.strftime('%Y-%m-%d')}")
                return latest_date
        except Exception as e:
            logger.warning(f"  -> 加载 {fund_code} CSV 失败: {e}")
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
        page_index = 1
        total_pages = 1
        first_run = True
        dynamic_delay = REQUEST_DELAY
        
        # 优化 4: 使用线程池异步加载最新日期
        latest_date = await asyncio.get_event_loop().run_in_executor(executor, load_latest_date, fund_code)

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
                # 优化 3: 成功抓取后，减少延迟以加速。
                dynamic_delay = max(REQUEST_DELAY * 0.5, dynamic_delay * 0.9) 

            except aiohttp.ClientError as e:
                if "Frequency Capped" in str(e):
                    # 优化 3: 频率限制时，更激进地增加延迟 (原 * 2, max 5.0)
                    dynamic_delay = min(dynamic_delay * 2.5, 10.0) 
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
        # print(f"    基金 {fund_code} 无新数据可保存。")
        return False, 0

    new_df = pd.DataFrame(data)

    try:
        new_df['net_value'] = pd.to_numeric(new_df['net_value'], errors='coerce').round(4)
        new_df['cumulative_net_value'] = pd.to_numeric(new_df['cumulative_net_value'], errors='coerce').round(4)
        new_df['daily_growth_rate'] = new_df['daily_growth_rate'].replace('--', '0').str.rstrip('%').astype(float) / 100.0
        new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
        new_df.dropna(subset=['date', 'net_value'], inplace=True)
        if new_df.empty:
            # print(f"    基金 {fund_code} 数据无效或为空，跳过保存。")
            return False, 0
    except Exception as e:
        print(f"    基金 {fund_code} 数据转换失败: {e}")
        return False, 0
    
    old_record_count = 0
    if os.path.exists(output_path):
        try:
            # 仅读取必要的列，加速 Pandas 加载
            existing_df = pd.read_csv(output_path, parse_dates=['date'], encoding='utf-8')
            old_record_count = len(existing_df)
            combined_df = pd.concat([new_df, existing_df])
        except Exception as e:
            # print(f"    读取现有 CSV 文件 {output_path} 失败: {e}。仅保存新数据。")
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
    
    # 优化 4: 创建线程池执行器，用于加速本地 I/O 操作 (读取和写入 CSV)
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2 + 1) as executor:

        # aiohttp 连接器设置：限制并发连接数，与 semaphore 配合
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5) 

        async with ClientSession(connector=connector) as session:
            # 传递 executor 给 fetch_net_values
            fetch_tasks = [fetch_net_values(fund_code, session, semaphore, executor) for fund_code in fund_codes]
            
            success_count = 0
            total_new_records = 0
            failed_codes = []
            
            # 使用 loop.run_in_executor 包装 save_to_csv
            save_executor = partial(loop.run_in_executor, executor, save_to_csv)

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
                        # 优化 4: 在线程池中执行耗时的保存操作，避免阻塞主循环
                        success, new_records = await save_executor(fund_code, net_values) 
                        if success:
                            success_count += 1
                            total_new_records += new_records
                        else:
                            failed_codes.append(fund_code)
                    except Exception as e:
                        print(f"[错误] 基金 {fund_code} 的保存任务在线程中发生错误: {e}")
                        failed_codes.append(fund_code)
                else:
                    # print(f"    基金 {fund_code} [抓取失败/跳过]：{net_values}")
                    if not str(net_values).startswith('数据已是最新'):
                        failed_codes.append(fund_code)

        return success_count, total_new_records, failed_codes

def main():
    """主函数：现在只执行动态净值抓取"""
    print(f"加速设置：并发数={MAX_CONCURRENT}，基础延迟={REQUEST_DELAY}秒")
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
    
    print(f"\n======== 本次更新总结 ========")
    print(f"成功处理 {success_count} 个基金，新增/更新 {total_new_records} 条记录，失败 {len(set(failed_codes))} 个基金。")
    if failed_codes:
        print(f"失败的基金代码: {', '.join(set(failed_codes))}")
    if total_new_records == 0:
        print("[警告] 未新增任何记录，可能是数据已是最新，或 API 无新数据。")
    print(f"==============================")

if __name__ == "__main__":
    main()
