import pandas as pd
import requests
import os
import time
import asyncio
import aiohttp
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import re
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_message
import json
import concurrent.futures # 引入 concurrent.futures

# 定义文件路径和目录
INPUT_FILE = 'C类.txt'
OUTPUT_DIR = 'fund_data'
BASE_URL = "http://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page={page_index}&per=20"

# 设置请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/533.36',
    'Referer': 'http://fund.eastmoney.com/',
}

REQUEST_TIMEOUT = 30 
REQUEST_DELAY = 0.5  # 初始延迟，动态调整
MAX_CONCURRENT = 5  # 最大并发基金数量
FORCE_UPDATE = False  # 是否强制重新抓取（忽略缓存）

# 保持上次新增的限制逻辑，以便在 GitHub Actions 中控制批次
MAX_FUNDS_PER_RUN = 100  # 每次运行脚本最多处理的基金代码数量 (0 表示不限制)

def get_all_fund_codes(file_path):
    """从 C类.txt 文件中读取基金代码（单列无标题，UTF-8 编码）"""
    print(f"尝试读取基金代码文件: {file_path}")
    encodings_to_try = ['utf-8', 'utf-8-sig', 'gbk', 'latin-1']
    df = None
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding, header=None, dtype=str)
            df.columns = ['code']  # 直接将第一列命名为 'code'
            print(f"  -> 成功使用 {encoding} 编码读取文件，找到 {len(df)} 个基金代码。")
            break
        except UnicodeDecodeError as e:
            print(f"  -> 使用 {encoding} 编码读取失败: {e}")
            continue
        except Exception as e:
            print(f"  -> 读取文件时发生错误: {e}")
            continue
            
    if df is None:
        print("  -> 无法读取文件，请检查文件格式和编码。")
        return []

    codes = df['code'].dropna().astype(str).unique().tolist()
    valid_codes = [code for code in codes if code.isdigit() and len(code) >= 3]
    print(f"  -> 找到 {len(valid_codes)} 个有效基金代码。")
    return valid_codes

def load_cache(fund_code):
    """加载缓存，获取已爬取的页面"""
    if FORCE_UPDATE:
        print(f"  -> 强制更新模式，忽略缓存 {fund_code}_cache.json。")
        return 0
    cache_file = os.path.join(OUTPUT_DIR, f"{fund_code}_cache.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
                last_page = cache.get('last_page', 0)
                # 移除此处过多的日志打印
                return last_page
        except Exception as e:
            print(f"  -> 加载缓存 {cache_file} 失败: {e}")
            return 0
    return 0

def save_cache(fund_code, last_page):
    """保存缓存，记录已爬取的页面"""
    cache_file = os.path.join(OUTPUT_DIR, f"{fund_code}_cache.json")
    try:
        with open(cache_file, 'w') as f:
            json.dump({'last_page': last_page}, f)
        # 移除此处过多的日志打印
    except Exception as e:
        print(f"  -> 保存缓存 {cache_file} 失败: {e}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_message(match='Frequency Capped')
)
async def fetch_page(session, url):
    """异步请求单页数据，带重试机制"""
    async with session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT) as response:
        if response.status == 514:
            raise aiohttp.ClientError("Frequency Capped")
        response.raise_for_status()
        return await response.text()

async def fetch_net_values(fund_code, session, semaphore):
    """使用天天基金 API 获取指定基金代码的【所有】历史净值数据 (分页，异步)"""
    # 仅在开始时打印
    print(f"-> [START] 基金代码 {fund_code}")
    
    async with semaphore:  # 控制并发
        all_records = []
        page_index = load_cache(fund_code) + 1  # 从缓存的下一页开始
        page_size = 20
        total_pages = 1
        first_run = True
        dynamic_delay = REQUEST_DELAY

        while page_index <= total_pages:
            url = BASE_URL.format(fund_code=fund_code, page_index=page_index, page_size=page_size)
            
            try:
                if page_index > 1:
                    await asyncio.sleep(dynamic_delay) 
                
                text = await fetch_page(session, url)
                soup = BeautifulSoup(text, 'lxml')
                
                if first_run:
                    total_pages_match = re.search(r'pages:(\d+)', text)
                    total_pages = int(total_pages_match.group(1)) if total_pages_match else 1
                    records_match = re.search(r'records:(\d+)', text)
                    total_records = int(records_match.group(1)) if records_match else '未知'
                    print(f"   基金 {fund_code} 信息：总页数 {total_pages}，总记录数 {total_records}。")
                    if page_index > total_pages:
                        print(f"   基金 {fund_code} [跳过]：缓存页数 ({page_index-1}) >= 总页数 ({total_pages})。")
                        return fund_code, f"缓存跳过: 缓存页数 {page_index-1} >= 总页数 {total_pages}"
                    first_run = False
                
                table = soup.find('table') 
                if not table:
                    print(f"   基金 {fund_code} 第 {page_index} 页未找到表格数据，可能无效或无数据。")
                    return fund_code, f"无表格数据: 基金代码可能无效"

                rows = table.find_all('tr')[1:] 
                if not rows:
                    print(f"   基金 {fund_code} 第 {page_index} 页无数据行，停止抓取。")
                    break

                new_records_count = 0
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        date_str = cols[0].text.strip()
                        net_value_str = cols[1].text.strip() 
                        
                        if date_str and net_value_str and net_value_str != '-':
                            all_records.append({'date': date_str, 'net_value': net_value_str})
                            new_records_count += 1
                        # else:
                        #     print(f"   基金 {fund_code} 第 {page_index} 页数据无效: 日期={date_str}, 净值={net_value_str}") # 移除此行，减少日志
                
                # 移除此处过多的日志打印，只通过缓存更新来追踪进度
                save_cache(fund_code, page_index)  # 保存缓存
                page_index += 1
                dynamic_delay = max(REQUEST_DELAY, dynamic_delay * 0.9)  # 动态减少延迟

            except aiohttp.ClientError as e:
                if "Frequency Capped" in str(e):
                    dynamic_delay = min(dynamic_delay * 2, 5.0)  # 频率限制时增加延迟
                    print(f"   基金 {fund_code} [警告]：频率限制，延迟调整为 {dynamic_delay} 秒，重试第 {page_index} 页")
                    continue
                print(f"   基金 {fund_code} [错误]：请求 API 时发生网络错误 (超时/连接) 在第 {page_index} 页: {e}")
                return fund_code, f"网络错误: {e}"
            except Exception as e:
                print(f"   基金 {fund_code} [错误]：处理数据时发生意外错误在第 {page_index} 页: {e}")
                return fund_code, f"数据处理错误: {e}"
            
        print(f"-> [COMPLETE] 基金 {fund_code} 数据抓取完毕，共获取 {len(all_records)} 条记录。")
        return fund_code, all_records

def save_to_csv(fund_code, data):
    """将历史净值数据以增量更新方式保存为 CSV 文件，格式为 date,net_value"""
    output_path = os.path.join(OUTPUT_DIR, f"{fund_code}.csv")
    if not isinstance(data, list):
        print(f"   基金 {fund_code} 数据无效: {data}")
        return False, 0

    new_df = pd.DataFrame(data)
    if new_df.empty:
        print(f"   基金 {fund_code} 无新数据可保存。")
        return False, 0

    try:
        new_df['net_value'] = pd.to_numeric(new_df['net_value'], errors='coerce').round(4)
        new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
        new_df.dropna(subset=['date', 'net_value'], inplace=True)
    except Exception as e:
        print(f"   基金 {fund_code} 数据转换失败: {e}")
        return False, 0
    
    old_record_count = 0
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path, parse_dates=['date'], dtype={'net_value': float}, encoding='utf-8')
            old_record_count = len(existing_df)
            combined_df = pd.concat([new_df, existing_df])
        except Exception as e:
            print(f"   读取现有 CSV 文件 {output_path} 失败: {e}")
            combined_df = new_df
    else:
        combined_df = new_df
        
    final_df = combined_df.drop_duplicates(subset=['date'], keep='first')
    final_df = final_df.sort_values(by='date', ascending=False)
    final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
    
    try:
        final_df.to_csv(output_path, index=False, encoding='utf-8')
        new_record_count = len(final_df)
        # 精简保存成功的日志
        print(f"   -> 基金 {fund_code} [保存完成]：总记录数 {new_record_count} (新增 {new_record_count - old_record_count} 条)。")
        return True, new_record_count - old_record_count
    except Exception as e:
        print(f"   基金 {fund_code} 保存 CSV 文件 {output_path} 失败: {e}")
        return False, 0

async def fetch_all_funds(fund_codes):
    """异步获取所有基金数据，并在任务完成时立即保存数据（实现边下载边保存）"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    async with ClientSession() as session:
        # 1. 创建所有异步抓取任务
        fetch_tasks = [fetch_net_values(fund_code, session, semaphore) for fund_code in fund_codes]
        
        # 2. 获取当前事件循环，用于运行同步的 save_to_csv (必须在单独的线程中运行)
        loop = asyncio.get_event_loop()
        
        success_count = 0
        total_new_records = 0
        failed_codes = []
        
        # 3. 使用 as_completed 迭代已完成的任务，立即处理结果并保存
        for future in asyncio.as_completed(fetch_tasks):
            print("-" * 30) # 保持分割线，用于区分不同基金的处理结果
            try:
                # 等待任务完成并获取结果
                result = await future 
            except Exception as e:
                # 捕获在 fetch_net_values 之外发生的顶级异常（例如任务取消）
                print(f"处理基金数据时发生顶级异步错误: {e}")
                failed_codes.append("未知基金代码")
                continue # 继续处理下一个已完成的任务

            # 结果处理 (与原 main() 逻辑类似)
            if isinstance(result, tuple) and len(result) == 2:
                fund_code, net_values = result
                
                if isinstance(net_values, list):
                    # 在单独的线程中运行同步的 save_to_csv，避免阻塞事件循环
                    try:
                        success, new_records = await loop.run_in_executor(
                            None, 
                            save_to_csv, 
                            fund_code, 
                            net_values
                        )
                        
                        if success:
                            success_count += 1
                            total_new_records += new_records
                        else:
                            failed_codes.append(fund_code)
                    except concurrent.futures.CancelledError:
                        print(f"基金 {fund_code} 的保存任务被取消。")
                        failed_codes.append(fund_code)
                    except Exception as e:
                        print(f"基金 {fund_code} 的保存任务在线程中发生错误: {e}")
                        failed_codes.append(fund_code)
                        
                else:
                    # 打印抓取失败信息
                    print(f"   基金 {fund_code} [抓取失败/跳过]：{net_values}")
                    # 只有抓取失败才计入 failed_codes，缓存跳过不计入
                    if not str(net_values).startswith('缓存跳过'):
                        failed_codes.append(fund_code)
            
        # 4. 返回最终统计结果
        return success_count, total_new_records, failed_codes

def main():
    """主函数"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    fund_codes = get_all_fund_codes(INPUT_FILE)
    if not fund_codes:
        print("没有可处理的基金代码，脚本结束。")
        return

    # 限制本次运行处理的基金数量
    if MAX_FUNDS_PER_RUN > 0 and len(fund_codes) > MAX_FUNDS_PER_RUN:
        print(f"限制本次运行最多处理 {MAX_FUNDS_PER_RUN} 个基金。")
        # 只取列表的前 MAX_FUNDS_PER_RUN 个基金代码
        processed_codes = fund_codes[:MAX_FUNDS_PER_RUN]
        print(f"本次实际处理的基金数量: {len(processed_codes)}")
    else:
        processed_codes = fund_codes
        print(f"本次处理所有 {len(processed_codes)} 个基金。")

    print(f"找到 {len(processed_codes)} 个基金代码，开始获取历史净值...")
    
    # 异步运行 fetch_all_funds，它现在包含了结果处理和保存逻辑
    success_count, total_new_records, failed_codes = asyncio.run(fetch_all_funds(processed_codes))
    
    # 打印总结
    print(f"\n======== 本次更新总结 ========")
    print(f"本次基金历史净值数据获取和保存完成。")
    print(f"总结: 成功处理 {success_count} 个基金，新增 {total_new_records} 条记录，失败 {len(failed_codes)} 个基金。")
    if failed_codes:
        print(f"失败的基金代码: {', '.join(failed_codes)}")
    if total_new_records == 0:
        print("警告: 未新增任何记录，可能是缓存跳过或无新数据，请检查 FORCE_UPDATE 设置或 API 响应。")
    print(f"==============================")


if __name__ == "__main__":
    main()
