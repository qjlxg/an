import requests
from bs4 import BeautifulSoup
import re
import os
import glob
import time
import sys
from datetime import datetime
import pandas as pd
import concurrent.futures
import random # 新增依赖，用于生成随机延迟

# 定义基金数据存放的目录
FUND_DATA_DIR = 'fund_data'
# 定义最大并发线程数。
MAX_WORKERS = 8 
# 定义每次请求后的随机延迟范围（秒）
MIN_DELAY = 1.0
MAX_DELAY = 3.0

def get_fund_codes_from_files(directory):
    """
    从指定目录下的CSV文件名中提取基金代码。
    """
    codes = set()
    for filepath in glob.glob(os.path.join(directory, '*.csv')):
        filename = os.path.basename(filepath)
        match = re.search(r'(\d+)', filename)
        if match:
            codes.add(match.group(1))
    return sorted(list(codes))

def fetch_fund_profile(fund_code):
    """
    从天天基金网获取单个基金的基本概况信息，并返回结果。
    """
    base_url = f"http://fund.eastmoney.com/{fund_code}.html"
    headers = {
        # 使用更完整的浏览器头部信息
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6',
        'Connection': 'keep-alive',
    }
    
    # 线程启动后，先进行随机延迟，模拟真实用户行为
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    print(f"--- [INFO] 线程启动: 基金代码 {fund_code}，延迟 {delay:.2f}s")
    time.sleep(delay)

    try:
        response = requests.get(base_url, headers=headers, timeout=15)
        response.raise_for_status() 
        response.encoding = 'utf-8' 
        soup = BeautifulSoup(response.text, 'html.parser')
        
        fund_data = {'基金代码': fund_code, '抓取时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        # 抓取页面主要文本内容，用于正则匹配费用信息
        # 尝试查找包含核心信息的区块，防止反爬页面影响
        main_info_container = soup.select_one('.bs_jjcc_cont') 
        full_text = main_info_container.text if main_info_container else response.text
        
        # 1. 查找基金名称、经理、规模等基础信息 
        title_tag = soup.select_one('.fundDetail-tit .dataName')
        if title_tag:
            name_match = re.search(r'(.+?)\((\d+)\)', title_tag.text.strip())
            if name_match:
                fund_data['基金名称'] = name_match.group(1).strip()
            # 否则，可能是反爬页面，或者名称解析失败
            else:
                fund_data['基金名称'] = title_tag.text.strip()
        
        # 检查是否成功抓取到主要信息，否则标记为失败页面
        if not fund_data.get('基金名称') or '天天基金' in fund_data.get('基金名称', ''):
             # 检查页面是否被重定向或内容缺失（反爬页面的典型特征）
            if not soup.select_one('.info.w100'): 
                raise Exception("页面内容缺失或被反爬机制拦截")


        info_list = soup.select('.info.w100 li')
        if info_list:
            for item in info_list:
                text = item.text.strip().replace('\xa0', ' ')
                if '成立日期' in text:
                    fund_data['成立日期'] = text.split('：')[-1].split('(')[0].strip()
                elif '基金规模' in text:
                    fund_data['基金规模'] = text.split('：')[-1].split('（截止')[0].strip()
                elif '基金管理人' in text:
                    manager_tag = item.find('a')
                    if manager_tag:
                        fund_data['基金管理人'] = manager_tag.text.strip()
                    else:
                        fund_data['基金管理人'] = text.split('：')[-1].strip()
        
        manager_div = soup.select_one('.manager_item p a')
        if manager_div:
            fund_data['基金经理'] = manager_div.text.strip()
            
        
        # 2. 抓取费用信息 (使用正则匹配)
        fee_patterns = {
            '管理费率': r'管理费率([\d\.]+%)',
            '托管费率': r'托管费率([\d\.]+%)',
            '销售服务费率': r'销售服务费率([\d\.]+%)',
        }

        for key, pattern in fee_patterns.items():
            match = re.search(pattern, full_text)
            fund_data[key] = match.group(1).strip() if match else 'N/A'

        fund_data['状态'] = '成功'
        
        # 详细日志输出
        print(f"--- [成功] 基金代码 {fund_code}: 名称: {fund_data.get('基金名称', 'N/A')}")
        print(f"   [费用] 管理费率: {fund_data.get('管理费率', 'N/A')}, 托管费率: {fund_data.get('托管费率', 'N/A')}")
        
        return fund_data

    except requests.exceptions.HTTPError as e:
        print(f"--- [失败] HTTP 错误 ({e.response.status_code}): 代码 {fund_code} 请求失败。")
        return {'基金代码': fund_code, '状态': f'失败: HTTP {e.response.status_code}'}
    except requests.exceptions.RequestException as e:
        print(f"--- [失败] 请求异常: 代码 {fund_code} 网络错误或超时。")
        return {'基金代码': fund_code, '状态': f'失败: 请求异常'}
    except Exception as e:
        if "页面内容缺失" in str(e):
             print(f"--- [失败] 反爬拦截: 代码 {fund_code} 页面内容被拦截。")
             return {'基金代码': fund_code, '状态': '失败: 反爬拦截'}
        else:
             print(f"--- [失败] 解析异常: 代码 {fund_code} 数据解析失败。错误: {e}")
             return {'基金代码': fund_code, '状态': f'失败: 解析异常'}

def main(output_file_path):
    if not os.path.isdir(FUND_DATA_DIR):
        print(f"致命错误: 找不到目录 {FUND_DATA_DIR}。")
        return

    fund_codes = get_fund_codes_from_files(FUND_DATA_DIR)
    
    if not fund_codes:
        print("未找到任何基金代码 (.csv 文件)。脚本结束。")
        return

    print(f"\n[START] 准备使用 {MAX_WORKERS} 个并发线程抓取 {len(fund_codes)} 个基金...\n")
    
    all_profiles = []
    
    # 使用 ThreadPoolExecutor 实现并行抓取
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_code = {executor.submit(fetch_fund_profile, code): code for code in fund_codes}
        total_tasks = len(fund_codes)
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_code)):
            code = future_to_code[future]
            try:
                profile = future.result()
                all_profiles.append(profile)
                
                print(f"[PROGRESS] {i+1}/{total_tasks} 任务完成。当前基金: {code}")
            except Exception as exc:
                print(f"[ERROR] 线程异常: 基金 {code} 在线程中产生异常: {exc}")
                all_profiles.append({'基金代码': code, '状态': f'失败: 线程异常 {exc}'})
                
    # --- 保存为 CSV 文件 ---
    df = pd.DataFrame(all_profiles)
    
    if df.empty:
        print("\n[ERROR] 未抓取到任何有效数据，无法生成 CSV 文件。")
        return
        
    # 确保列顺序，新增了三个费用列
    column_order = [
        '基金代码', '基金名称', '基金经理', '基金管理人', 
        '管理费率', '托管费率', '销售服务费率',
        '基金规模', '成立日期', '状态', '抓取时间'
    ]
    final_columns = [col for col in column_order if col in df.columns]
    df = df[final_columns]
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # 保存结果到 CSV 文件
    try:
        df.to_csv(output_file_path, index=False, encoding='utf-8-sig') 
        print(f"\n[DONE] 抓取完成。结果已保存为 CSV 到 {output_file_path}")
        print(f"[DONE] CSV 包含 {len(df)} 条记录。")
    except Exception as e:
        print(f"\n[ERROR] 写入 CSV 文件失败: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python scrape_all_funds.py <output_file_path>")
        sys.exit(1)
        
    output_path = sys.argv[1]
    main(output_path)
