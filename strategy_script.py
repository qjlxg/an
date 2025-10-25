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
import re # 确保导入 re 模块

# --- 配置 ---
DATA_DIR = 'fund_data'
OUTPUT_DIR = 'fetched_pb_data'  # 输出目录名称

# --- PB 数据获取函数 (使用 FundArchivesDatas.aspx?type=guzhi 接口修复) ---
def fetch_pb_from_eastmoney(fund_code):
    """
    【最终尝试：切换到 guzhi 估值接口】通过 FundArchivesDatas.aspx 接口的 guzhi (估值) 类型获取最新的 PB 值。
    """
    # **【关键修复点：URL 切换到 guzhi 接口】**
    # 估值 (guzhi) 页面最有可能包含 PB 数据
    url = f"https://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=guzhi&code={fund_code}"
    
    # 伪装请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': f'http://fundf10.eastmoney.com/guzhi_{fund_code}.html' # 模仿来自 F10 估值页面的请求
    }
    
    # 随机延迟
    time.sleep(random.uniform(0.5, 1.5))
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        # 使用你原始代码中成功的提取模式：从 content:"" 中提取 HTML 片段
        html_match = re.findall(r"content:\s*\"(.*?)\"\s*};", response.text, re.DOTALL)
        
        if not html_match:
            return None

        # 解析提取到的 HTML 片段
        # 注意：这里我们使用 BeautifulSoup 的内置功能来处理 HTML，避免复杂的字符串替换
        rise_html = html_match[0].replace('\\"', '"').replace('\\n', '\n') 
        rise_soup = BeautifulSoup(rise_html, 'html.parser')
        
        # 查找包含 '市净率' 文本的 <th> 或 <td> 单元格
        pb_label = rise_soup.find(['th', 'td'], text=re.compile(r'市净率'))
        
        if pb_label:
            # PB 值通常在标签的下一个兄弟单元格中（我们假设是表格结构）
            pb_value_cell = pb_label.find_next_sibling(['td', 'th'])
            
            if pb_value_cell:
                # 提取文本并清理
                pb_text = pb_value_cell.text.strip()
                # 确保它不是空字符串或 '-'
                if pb_text and pb_text != '-': 
                    pb_value = float(pb_text)
                    return pb_value
        
        return None
            
    except Exception as e:
        return None

# 移除了 calculate_3day_fall 函数

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

    # 保持参数来源不变：从 fund_data 目录下读取 CSV 文件名作为基金代码
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
            
        except Exception as e:
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
