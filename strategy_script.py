# strategy_script.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import pytz
import requests
from bs4 import BeautifulSoup 
import time
import random
import re 

# --- 配置 ---
DATA_DIR = 'fund_data'
OUTPUT_DIR = '推荐结果' # 策略输出目录
PB_THRESHOLD = 1.2        # PB < 1.2 的筛选条件
TOP_N_RANK = 1            # 跌幅Top1
FALLBACK_DAYS = 3         # 近3日跌幅

# CSV 文件中的列名
DATE_COL = 'date'
NAV_COL = 'net_value'
REQUEST_TIMEOUT = 10 

# --- PB 数据获取函数 (使用 tsdata 页面及正则搜索) ---
def fetch_pb_from_eastmoney(fund_code):
    """
    【最终修复】目标: 特色数据页面 (tsdata)，采用正则搜索整个 HTML 文本获取 PB 值。
    """
    # **【最终目标页面】**
    url = f"http://fundf10.eastmoney.com/tsdata_{fund_code}.html" 
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'http://fundf10.eastmoney.com/'
    }
    
    time.sleep(random.uniform(0.5, 1.5))
    
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT) 
        
        # 尝试GBK解码，应对旧版F10页面
        try:
            response.encoding = response.apparent_encoding if response.apparent_encoding else 'utf-8'
            if response.encoding.lower() not in ['utf-8', 'gbk', 'gb2312']:
                response.encoding = 'utf-8' 
        except:
             response.encoding = 'utf-8'

        html_text = response.text
        
        # **【核心修复：正则搜索】**
        # 搜索 "市净率" 附近是否有浮点数 (PB值通常是浮点数)
        pb_match = re.search(r'市净率.*?(\d+\.\d+)', html_text, re.DOTALL)
        
        if pb_match:
            pb_value = float(pb_match.group(1))
            return pb_value
        
        # 如果正则没找到，退化到 BeautifulSoup 表格搜索
        soup = BeautifulSoup(html_text, 'html.parser')
        pb_label = soup.find(['th', 'td'], text=re.compile(r'市净率'))
        
        if pb_label:
            pb_value_cell = pb_label.find_next_sibling(['td', 'th'])
            if pb_value_cell:
                pb_text = pb_value_cell.text.strip()
                if pb_text and pb_text != '-': 
                    return float(pb_text)
        
        return None
            
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None

# --- 净值跌幅计算函数 ---
def calculate_3day_fall(file_path):
    """
    计算近 FALLBACK_DAYS 的跌幅 (今日净值 - FALLBACK_DAYS前净值) / FALLBACK_DAYS前净值 * 100
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        if df.empty or len(df) < FALLBACK_DAYS:
            return None

        # 确保列名正确
        df.columns = [DATE_COL, NAV_COL]
        
        # 将日期转换为 datetime 对象并排序，确保最新的在最上面
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df.sort_values(by=DATE_COL, ascending=False).reset_index(drop=True)
        
        # 获取最新净值和 FALLBACK_DAYS 前的净值
        latest_nav = df.loc[0, NAV_COL]
        old_nav = df.loc[FALLBACK_DAYS - 1, NAV_COL]
        
        if np.isnan(latest_nav) or np.isnan(old_nav) or old_nav == 0:
            return None
            
        # 跌幅计算: (最新净值 - 历史净值) / 历史净值 * 100
        fall_rate = (latest_nav - old_nav) / old_nav * 100
        
        return fall_rate # 跌幅为负数表示下跌
        
    except Exception as e:
        return None

# --- Main Logic ---
def main():
    print("--- 正在执行基金筛选策略 ---")
    
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(shanghai_tz) 
    
    date_dir = now.strftime('%Y%m%d')
    full_output_dir = os.path.join(OUTPUT_DIR, date_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    all_funds_data = []

    if not os.path.exists(DATA_DIR):
        print(f"错误: 必需的 {DATA_DIR} 目录不存在。")
        return

    # 获取所有基金代码 (从 fund_data 目录下的 CSV 文件名获取)
    fund_codes = [f.replace('.csv', '').zfill(6) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    if not fund_codes:
        print("错误: fund_data 目录下没有找到任何 CSV 文件。")
        return

    # 循环处理每个基金
    for fund_code in fund_codes:
        file_path = os.path.join(DATA_DIR, f'{fund_code}.csv')
        
        try:
            # 1. 获取 PB 数据
            latest_pb = fetch_pb_from_eastmoney(fund_code)
            
            if latest_pb is None:
                continue 

            # 2. 筛选 PB < PB_THRESHOLD
            if latest_pb >= PB_THRESHOLD:
                continue 

            # 3. 计算跌幅
            fall_rate = calculate_3day_fall(file_path)
            
            if fall_rate is None:
                continue 

            # 4. 收集数据 (PB < 1.2 且有跌幅数据的基金)
            all_funds_data.append({
                '基金代码': fund_code,
                f'近{FALLBACK_DAYS}日跌幅(%)': fall_rate,
                '最新PB': latest_pb,
            })
            
        except Exception:
            continue

    if not all_funds_data:
        print(f"没有基金同时满足 PB < {PB_THRESHOLD} 条件和足够的净值数据。")
        return

    # 5. 筛选 跌幅 Top1 (数值越小越跌得多，即负数绝对值越大)
    results_df = pd.DataFrame(all_funds_data)
    sort_column = f'近{FALLBACK_DAYS}日跌幅(%)'
    
    # 按照跌幅升序排列 (负数越小，跌幅越大)
    sorted_df = results_df.sort_values(
        by=sort_column, 
        ascending=True 
    ).dropna(subset=[sort_column])

    final_recommendation = sorted_df.head(TOP_N_RANK)
    
    if final_recommendation.empty:
        print("满足 PB 条件的基金中，没有足够的跌幅数据来筛选 Top1。")
        return

    # 6. 保存结果
    final_recommendation = final_recommendation[[\
        '基金代码', 
        sort_column,
        '最新PB'
    ]]
    final_recommendation[sort_column] = final_recommendation[sort_column].round(4)
    final_recommendation['最新PB'] = final_recommendation['最新PB'].round(4)

    timestamp = now.strftime('%H%M%S')
    output_filename = os.path.join(full_output_dir, f'基金推荐结果_{timestamp}.csv')
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    final_recommendation.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"基金筛选策略执行完成。推荐结果已保存到：{output_filename}")
    print("--- 脚本执行结束 ---")

if __name__ == '__main__':
    main()
