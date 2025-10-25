import os
import requests
from bs4 import BeautifulSoup
import csv
import datetime
import pytz
import subprocess
from concurrent.futures import ThreadPoolExecutor # 引入并发模块

# 设置上海时区
shanghai_tz = pytz.timezone('Asia/Shanghai')

# 获取当前日期和时间（上海时区）
now = datetime.datetime.now(shanghai_tz)
date_dir = now.strftime('%Y%m%d')
timestamp = now.strftime('%Y%m%d_%H%M%S')

# 基金数据目录
fund_data_dir = 'fund_data'

# 输出目录（仓库中年月日目录）
output_base_dir = date_dir
os.makedirs(output_base_dir, exist_ok=True)

# 定义一个函数来处理单个基金的抓取和保存
def scrape_and_save_fund(fund_code):
    """抓取单个基金的基本概况信息并保存为CSV。"""
    
    # 构建URL
    url = f"https://fundf10.eastmoney.com/jbgk_{fund_code}.html"
    
    try:
        # 抓取页面内容，设置超时以防请求卡死
        response = requests.get(url, timeout=10) # 增加10秒超时
        if response.status_code != 200:
            print(f"[{fund_code}] Failed to fetch {url}. Status code: {response.status_code}")
            return # 失败则跳过
        
        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 找到基本概况表格（class="info w790"）
        table = soup.find('table', class_='info w790')
        if not table:
            print(f"[{fund_code}] No table found.")
            return
        
        # 提取表格数据
        data = []
        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all(['th', 'td'])
            row_data = [col.text.strip() for col in cols]
            data.append(row_data)
        
        # 保存为CSV文件
        output_filename = f"basic_info_{fund_code}_{timestamp}.csv"
        output_path = os.path.join(output_base_dir, output_filename)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
        
        print(f"[{fund_code}] Saved {output_path}")

    except requests.exceptions.Timeout:
        print(f"[{fund_code}] Request timed out for {url}")
    except requests.exceptions.RequestException as e:
        print(f"[{fund_code}] An error occurred: {e}")
    except Exception as e:
        print(f"[{fund_code}] An unexpected error occurred: {e}")

# --- 主执行逻辑 ---

# 收集所有基金代码
fund_codes_to_process = []
for filename in os.listdir(fund_data_dir):
    if filename.endswith('.csv'):
        fund_code = filename.split('.')[0]
        fund_codes_to_process.append(fund_code)

print(f"Found {len(fund_codes_to_process)} funds to scrape. Starting parallel process...")

# 使用线程池进行并行处理
# MAX_WORKERS可以根据实际情况调整，通常20到50在I/O密集型任务中效果最好
MAX_WORKERS = 30 
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # 提交所有任务
    executor.map(scrape_and_save_fund, fund_codes_to_process)

print("All scraping tasks finished.")

# Git推送结果到仓库（保持不变）
# 注意：这里我们使用'git add .'来确保添加了新创建的日期目录
subprocess.run(['git', 'add', output_base_dir])
# 提交前检查是否有改动，可以简化为：
commit_result = subprocess.run(['git', 'commit', '-m', f"Add basic info files for {date_dir}"], capture_output=True, text=True)
if "nothing to commit" in commit_result.stdout or "nothing to commit" in commit_result.stderr:
    print("No changes to commit after scraping.")
else:
    print(commit_result.stdout)
    subprocess.run(['git', 'push'])
    print("Push finished.")
