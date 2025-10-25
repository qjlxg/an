import requests
from bs4 import BeautifulSoup
import re
import json
import os
import glob
import time

# 定义基金数据存放的目录
FUND_DATA_DIR = 'fund_data'
# 定义抓取结果的输出文件
OUTPUT_FILE = 'fund_profiles_latest.json'

def get_fund_codes_from_files(directory):
    """
    从指定目录下的CSV文件名中提取基金代码。
    """
    codes = set()
    # 查找目录中所有的.csv文件
    for filepath in glob.glob(os.path.join(directory, '*.csv')):
        filename = os.path.basename(filepath)
        # 提取文件名中数字部分作为代码
        match = re.search(r'(\d+)', filename)
        if match:
            codes.add(match.group(1))
    return sorted(list(codes))

def fetch_fund_profile(fund_code):
    """
    从天天基金网获取单个基金的基本概况信息。
    注意: 此处为基础爬虫逻辑，可能需要根据网站反爬机制调整。
    """
    base_url = f"http://fund.eastmoney.com/{fund_code}.html"
    # 模拟浏览器访问，添加User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"-> 正在抓取代码 {fund_code}...")
    
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
                text = item.text.strip().replace('\xa0', ' ') # 清理空白字符
                if '成立日期' in text:
                    fund_data['成立日期'] = text.split('：')[-1].split('(')[0].strip()
                elif '基金规模' in text:
                    fund_data['基金规模'] = text.split('：')[-1].split('（截止')[0].strip()
                elif '基金管理人' in text:
                    # 仅获取第一个链接的文本作为管理人
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
        print(f"   [失败] 代码 {fund_code} 请求失败: {e}")
        return {'基金代码': fund_code, '状态': f'抓取失败: {str(e)}'}
    except Exception as e:
        print(f"   [失败] 代码 {fund_code} 解析失败: {e}")
        return {'基金代码': fund_code, '状态': f'解析失败: {str(e)}'}

def main():
    # 确保 fund_data 目录存在，虽然在 GitHub Actions 中不需要，但在本地调试时很有用
    if not os.path.isdir(FUND_DATA_DIR):
        print(f"警告: 找不到目录 {FUND_DATA_DIR}，请确保您的基金 CSV 文件放在该目录中。")
    
    # 1. 获取所有基金代码
    fund_codes = get_fund_codes_from_files(FUND_DATA_DIR)
    
    if not fund_codes:
        print("未在 fund_data 目录中找到任何基金代码（如 014194.csv）。脚本结束。")
        return

    print(f"找到以下基金代码进行抓取: {', '.join(fund_codes)}")
    
    all_profiles = []
    
    # 2. 迭代抓取每个基金
    for code in fund_codes:
        profile = fetch_fund_profile(code)
        all_profiles.append(profile)
        # 增加延迟，避免触发反爬虫
        time.sleep(2) 
        
    # 3. 保存结果到 JSON 文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_profiles, f, ensure_ascii=False, indent=4)
        
    print(f"\n--- 抓取完成 ---")
    print(f"成功抓取 {len(all_profiles)} 个基金信息，结果已保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
