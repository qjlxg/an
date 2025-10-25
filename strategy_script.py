# strategy_script.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pytz
import requests
import re
import time
import random

# --- 配置 ---
DATA_DIR = 'fund_data'
OUTPUT_DIR = '推荐结果'
PB_THRESHOLD = 1.2        # PB < 1.2 的筛选条件
TOP_N_RANK = 1            # 跌幅Top1
FALLBACK_DAYS = 3         # 近3日跌幅

# CSV 文件中的列名 (匹配您提供的文件格式)
DATE_COL = 'date'
NAV_COL = 'net_value'
# CUM_NAV_COL 移除，不再用于筛选

# --- PB 数据获取函数 ---
def fetch_pb_from_eastmoney(fund_code):
    """尝试从天天基金的估值页面获取最新的 PB 值。"""
    # 针对 PB/PE 数据的估值页面 URL
    url = f"http://fundf10.eastmoney.com/jzzs_{fund_code}.html"
    
    # 伪装请求头，防止被网站屏蔽
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'http://fundf10.eastmoney.com/'
    }
    
    # 随机延迟，防止请求过于频繁
    time.sleep(random.uniform(0.5, 1.5))
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        html_text = response.text
        
        # 尝试通过正则表达式在 HTML 中查找 PB 值
        # 目标结构大致为：<td>市净率</td><td>1.2345</td>
        # 或者在表格中的特定行/列
        
        # 查找包含 '市净率' 文本的行，并尝试提取其后的数字
        # r'市净率.*?(\d+\.\d+)' 查找 '市净率' 后面的第一个浮点数
        match = re.search(r'市净率.*?(\d+\.\d+)', html_text)
        
        if match:
            pb_value = float(match.group(1))
            return pb_value
        else:
            print(f"警告：基金 {fund_code} 页面结构已变化或无 PB 数据。")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"错误：获取基金 {fund_code} 的 PB 数据网络请求失败: {e}")
        return None
    except Exception as e:
        print(f"错误：解析基金 {fund_code} 的 PB 数据失败: {e}")
        return None

def calculate_3day_fall(df):
    """计算近3日跌幅（最新一日相比3日前）"""
    # 确保有足够的净值数据用于计算
    if len(df) < FALLBACK_DAYS + 1:
        return np.nan
    
    # 确保日期降序排列，最新数据在顶部
    df_sorted = df.sort_values(by=DATE_COL, ascending=False)
    
    if len(df_sorted) <= FALLBACK_DAYS:
        return np.nan
        
    latest_nav = df_sorted[NAV_COL].iloc[0]
    nav_3days_ago = df_sorted[NAV_COL].iloc[FALLBACK_DAYS]
    
    if nav_3days_ago == 0 or pd.isna(latest_nav) or pd.isna(nav_3days_ago):
        return np.nan
        
    fall_percentage = (latest_nav / nav_3days_ago - 1) * 100
    
    return fall_percentage

def main():
    print("--- 策略开始执行：PB < 1.2 + 近3日跌幅 Top1 ---")
    
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(shanghai_tz) 
    
    date_dir = now.strftime('%Y%m%d')
    full_output_dir = os.path.join(OUTPUT_DIR, date_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    all_funds_data = []

    # 1. 遍历 fund_data 目录并计算
    if not os.path.exists(DATA_DIR):
        print(f"错误: 必需的 {DATA_DIR} 目录不存在。")
        return

    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.csv'):
            fund_code = filename.replace('.csv', '').zfill(6) 
            file_path = os.path.join(DATA_DIR, filename)
            
            try:
                # 1. 获取 PB 数据 (Web Scraping)
                latest_pb = fetch_pb_from_eastmoney(fund_code)
                
                # --- 策略筛选（PB < 1.2） ---
                if latest_pb is None or latest_pb >= PB_THRESHOLD:
                    print(f"基金 {fund_code} 跳过：PB数据缺失或 PB ({latest_pb}) >= {PB_THRESHOLD}")
                    continue 
                
                # 2. 读取 CSV 文件并计算跌幅
                df = pd.read_csv(file_path)
                df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
                df[NAV_COL] = pd.to_numeric(df[NAV_COL], errors='coerce')

                fall_3d = calculate_3day_fall(df.dropna(subset=[NAV_COL, DATE_COL]))
                
                all_funds_data.append({
                    '基金代码': fund_code,
                    f'近{FALLBACK_DAYS}日跌幅(%)': fall_3d,
                    '最新PB': latest_pb,
                })
                
            except Exception as e:
                print(f"处理文件 {filename} 时发生未知错误: {e}")
                continue

    if not all_funds_data:
        print(f"没有基金同时满足 PB < {PB_THRESHOLD} 条件和足够的净值数据。")
        return

    # 2. 筛选 近3日跌幅 Top1
    results_df = pd.DataFrame(all_funds_data)
    sort_column = f'近{FALLBACK_DAYS}日跌幅(%)'
    
    # 排序：按跌幅升序（跌幅最大，即值最小/最负）
    sorted_df = results_df.sort_values(
        by=sort_column, 
        ascending=True 
    ).dropna(subset=[sort_column])

    final_recommendation = sorted_df.head(TOP_N_RANK)
    
    if final_recommendation.empty:
        print("满足 PB 条件的基金中，没有足够的跌幅数据来筛选 Top1。")
        return

    # 3. 保存结果
    final_recommendation = final_recommendation[[
        '基金代码', 
        sort_column,
        '最新PB'
    ]]
    final_recommendation[sort_column] = final_recommendation[sort_column].round(4)
    final_recommendation['最新PB'] = final_recommendation['最新PB'].round(4)

    timestamp = now.strftime('%H%M%S')
    output_filename = os.path.join(full_output_dir, f'推荐_{timestamp}.csv')
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    final_recommendation.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"策略筛选完成。推荐结果已保存到：{output_filename}")
    print("推荐基金:")
    print(final_recommendation)
    print("--- 策略执行结束 ---")

if __name__ == '__main__':
    main()
