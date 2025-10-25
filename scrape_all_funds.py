import requests
from bs4 import BeautifulSoup
import re
import os
import glob
import time
import sys
from datetime import datetime
import pandas as pd # 新增依赖，用于生成 CSV

# 定义基金数据存放的目录
FUND_DATA_DIR = 'fund_data'
# 每次抓取后增加的延迟时间（秒）
REQUEST_DELAY = 2

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
    从天天基金网获取单个基金的基本概况信息，并打印抓取状态及部分内容。
    """
    base_url = f"http://fund.eastmoney.com/{fund_code}.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"--- [INFO] 开始抓取基金代码: {fund_code} ---")

    try:
        response = requests.get(base_url, headers=headers, timeout=10)
        response.raise_for_status() 
        response.encoding = 'utf-8' 
        soup = BeautifulSoup(response.text, 'html.parser')
        
        fund_data = {'基金代码': fund_code, '抓取时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
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
        
        # --- 详细日志输出 ---
        if '基金名称' in fund_data:
            print(f"   [成功] 名称: {fund_data.get('基金名称', 'N/A')}")
            print(f"   [成功] 经理: {fund_data.get('基金经理', 'N/A')}")
            print("   [数据预览 (部分)]: ")
            print(f"     - 基金规模: {fund_data.get('基金规模', 'N/A')}")
            print(f"     - 成立日期: {fund_data.get('成立日期', 'N/A')}")
        else:
            print("   [警告] 抓取到页面但未能解析出核心数据。")
            
        fund_data['状态'] = '成功'
        return fund_data

    except requests.exceptions.HTTPError as e:
        print(f"   [失败] HTTP 错误 ({e.response.status_code}): 请求被拒绝或页面不存在。")
        return {'基金代码': fund_code, '状态': f'失败: HTTP {e.response.status_code}'}
    except requests.exceptions.RequestException as e:
        print(f"   [失败] 请求异常: 发生网络错误或超时。")
        return {'基金代码': fund_code, '状态': f'失败: 请求异常'}
    except Exception as e:
        print(f"   [失败] 解析异常: 解析数据失败。错误: {e}")
        return {'基金代码': fund_code, '状态': f'失败: 解析异常'}

def main(output_file_path):
    if not os.path.isdir(FUND_DATA_DIR):
        print(f"致命错误: 找不到目录 {FUND_DATA_DIR}。")
        return

    fund_codes = get_fund_codes_from_files(FUND_DATA_DIR)
    
    if not fund_codes:
        print("未找到任何基金代码 (.csv 文件)。脚本结束。")
        return

    print(f"\n[START] 准备抓取 {len(fund_codes)} 个基金代码: {', '.join(fund_codes)}\n")
    
    all_profiles = []
    
    for i, code in enumerate(fund_codes):
        profile = fetch_fund_profile(code)
        all_profiles.append(profile)
        
        if i < len(fund_codes) - 1:
            print(f"   [INFO] 暂停 {REQUEST_DELAY} 秒，避免触发反爬...")
            time.sleep(REQUEST_DELAY) 
        
    # --- 保存为 CSV 文件 ---
    df = pd.DataFrame(all_profiles)
    
    if df.empty:
        print("\n[ERROR] 未抓取到任何有效数据，无法生成 CSV 文件。")
        return
        
    # 确保列顺序
    column_order = ['基金代码', '基金名称', '基金经理', '基金管理人', '基金规模', '成立日期', '状态', '抓取时间']
    final_columns = [col for col in column_order if col in df.columns]
    df = df[final_columns]
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # 保存结果到 CSV 文件 (使用 utf-8-sig 编码，兼容 Excel 中文显示)
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
