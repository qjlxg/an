import requests
from bs4 import BeautifulSoup
import re
import json
import os
import glob
import time
import sys

# 定义基金数据存放的目录
FUND_DATA_DIR = 'fund_data'

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
    从天天基金网获取单个基金的基本概况信息。
    """
    base_url = f"http://fund.eastmoney.com/{fund_code}.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # 移除 print(f"-> 正在抓取代码 {fund_code}...") 以保持 Actions 日志简洁

    try:
        response = requests.get(base_url, headers=headers, timeout=10)
        response.raise_for_status() 
        response.encoding = 'utf-8' 
        soup = BeautifulSoup(response.text, 'html.parser')
        
        fund_data = {'基金代码': fund_code}
        
        # 1. 查找基金名称
        title_tag = soup.select_one('.fundDetail-tit .dataName')
        if title_tag:
            name_match = re.search(r'(.+?)\((\d+)\)', title_tag.text.strip())
            if name_match:
                fund_data['基金名称'] = name_match.group(1).strip()

        # 2. 查找概况表格（成立日期、规模等）
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
        
        # 3. 查找基金经理
        manager_div = soup.select_one('.manager_item p a')
        if manager_div:
            fund_data['基金经理'] = manager_div.text.strip()
            
        return fund_data

    except requests.exceptions.RequestException as e:
        return {'基金代码': fund_code, '状态': f'抓取失败: {str(e)}'}
    except Exception as e:
        return {'基金代码': fund_code, '状态': f'解析失败: {str(e)}'}

def main(output_file_path):
    if not os.path.isdir(FUND_DATA_DIR):
        print(f"警告: 找不到目录 {FUND_DATA_DIR}。")
        return

    fund_codes = get_fund_codes_from_files(FUND_DATA_DIR)
    
    if not fund_codes:
        print("未找到任何基金代码。脚本结束。")
        return

    print(f"找到 {len(fund_codes)} 个基金代码进行抓取。")
    
    all_profiles = []
    
    for code in fund_codes:
        profile = fetch_fund_profile(code)
        all_profiles.append(profile)
        # 增加延迟
        time.sleep(2) 
        
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        # 使用 os.makedirs(..., exist_ok=True) 确保即使目录已存在也不会报错
        os.makedirs(output_dir, exist_ok=True)
        
    # 保存结果到 JSON 文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_profiles, f, ensure_ascii=False, indent=4)
        
    print(f"\n--- 抓取完成 ---")
    print(f"抓取结果已保存到 {output_file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python scrape_all_funds.py <output_file_path>")
        sys.exit(1)
        
    output_path = sys.argv[1]
    main(output_path)
