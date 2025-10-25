# strategy_script.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pytz
import requests
from bs4 import BeautifulSoup 
import time
import random
import re 
import json 

# --- 配置 ---
DATA_DIR = 'fund_data'
OUTPUT_DIR = 'fetched_pb_data'  # 输出目录名称
REQUEST_TIMEOUT = 10 # 请求超时时间

# --- PB 数据获取函数 (使用 资产配置页面 修复) ---
def fetch_pb_from_eastmoney(fund_code):
    """
    【最终修复：切换到资产配置页面】解析 F10 资产配置页面，从中搜索 PB 值。
    """
    # **【关键修复点：URL 切换到 zcpz 接口】**
    # 资产配置 (zcpz) 页面最有可能包含 PB 数据
    url = f"http://fundf10.eastmoney.com/zcpz_{fund_code}.html" 
    
    # 伪装请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': f'http://fundf10.eastmoney.com/{fund_code}.html'
    }
    
    # 随机延迟
    time.sleep(random.uniform(0.5, 1.5))
    
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT) 
        # 很多 F10 页面使用 GBK 编码，如果 UTF-8 乱码则自动检测
        if 'charset=gb2312' in response.text.lower() or 'charset=gbk' in response.text.lower():
            response.encoding = 'gbk'
        else:
            response.encoding = 'utf-8'
            
        html_text = response.text
        
        soup = BeautifulSoup(html_text, 'html.parser')
        
        # 查找包含 '市净率' 文本的 <th> 或 <td> 单元格
        pb_label = soup.find(['th', 'td'], text=re.compile(r'市净率'))
        
        if pb_label:
            # PB 值通常在标签的下一个兄弟单元格中，或者在同一个表格的下一行
            
            # 尝试获取标签的下一个兄弟节点（最常见的情况）
            pb_value_cell = pb_label.find_next_sibling(['td', 'th'])
            
            if pb_value_cell:
                pb_text = pb_value_cell.text.strip()
                if pb_text and pb_text != '-': 
                    pb_value = float(pb_text)
                    return pb_value
        
        return None
            
    except requests.exceptions.Timeout:
        print(f"警告：基金 {fund_code} 请求超时 ({REQUEST_TIMEOUT}s)。")
        return None
    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None

# main 函数保持不变，只负责读取文件、调用 PB 函数和保存结果。
def main():
    print("--- 策略已移除，开始执行：基金资料 (PB) 抓取 ---")
    
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(shanghai_tz) 
    
    date_dir = now.strftime('%Y%m%d')
    full_output_dir = os.path.join(OUTPUT_DIR, date_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    all_funds_data = []

    if not os.path.exists(DATA_DIR):
        print(f"错误: 必需的 {DATA_DIR} 目录不存在。")
        return

    # 从 fund_data 目录下读取 CSV 文件名作为基金代码
    fund_codes = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.csv'):
            fund_codes.append(filename.replace('.csv', '').zfill(6))

    if not fund_codes:
        print("错误: fund_data 目录下没有找到任何 CSV 文件。")
        return

    # 循环处理每个基金
    for fund_code in fund_codes:
        try:
            # 1. 获取 PB 数据 
            latest_pb = fetch_pb_from_eastmoney(fund_code)
            
            # 2. 收集数据 (只收集 PB 数据)
            all_funds_data.append({
                '基金代码': fund_code,
                '最新PB': latest_pb if latest_pb is not None else 'N/A',
            })
            
        except Exception:
            continue

    if not all_funds_data:
        print(f"未能抓取到任何基金资料。")
        return

    # 3. 保存结果 
    results_df = pd.DataFrame(all_funds_data)

    timestamp = now.strftime('%H%M%S')
    output_filename = os.path.join(full_output_dir, f'基金PB数据_{timestamp}.csv')
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"基金资料 (PB) 抓取完成。结果已保存到：{output_filename}")
    print("--- 脚本执行结束 ---")

if __name__ == '__main__':
    main()
