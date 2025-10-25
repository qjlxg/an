import os
import requests
from bs4 import BeautifulSoup
import csv
import datetime
import pytz
import subprocess
from concurrent.futures import ThreadPoolExecutor
import re

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

# 定义全局变量用于保存字段映射
FIELD_MAPPING = {}

# 定义一个函数来处理单个基金的抓取和解析
def scrape_and_parse_fund(fund_code):
    """抓取单个基金的基本概况信息，并返回一个包含键值对的字典。"""
    
    # 构建URL
    url = f"https://fundf10.eastmoney.com/jbgk_{fund_code}.html"
    result = {'基金代码': fund_code, '状态': '成功'}
    
    try:
        # 抓取页面内容，设置超时
        response = requests.get(url, timeout=15) # 适当增加超时时间
        if response.status_code != 200:
            result['状态'] = f"抓取失败: Status code {response.status_code}"
            return result
        
        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 找到基本概况表格（class="info w790"）
        table = soup.find('table', class_='info w790')
        if not table:
            result['状态'] = "抓取失败: 未找到表格"
            return result
        
        # 提取表格数据并转换为字典
        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all(['th', 'td'])
            
            # 基础格式：每行是 <th>Label</th> <td>Value</td>
            if len(cols) >= 2:
                label = cols[0].text.strip().replace('：', '').replace(':', '')
                value = cols[1].text.strip()
                result[label] = value
                
            # 扩展格式：每行可能是 <th>Label1</th> <td>Value1</td> <th>Label2</th> <td>Value2</td>
            if len(cols) == 4:
                label2 = cols[2].text.strip().replace('：', '').replace(':', '')
                value2 = cols[3].text.strip()
                result[label2] = value2
                
    except requests.exceptions.Timeout:
        result['状态'] = "抓取失败: 请求超时"
    except requests.exceptions.RequestException as e:
        result['状态'] = f"抓取失败: 网络错误 ({e})"
    except Exception as e:
        result['状态'] = f"抓取失败: 未知错误 ({e})"

    print(f"[{fund_code}] 状态: {result.get('状态')}")
    return result

# --- 主执行逻辑 ---

# 收集所有基金代码
fund_codes_to_process = []
for filename in os.listdir(fund_data_dir):
    # 确保只处理有效的基金代码文件（例如，不包含中文或日期等）
    if filename.endswith('.csv') and re.match(r'^\d+\.csv$', filename):
        fund_code = filename.split('.')[0]
        fund_codes_to_process.append(fund_code)

print(f"找到 {len(fund_codes_to_process)} 个基金准备开始并行抓取...")

# 使用线程池进行并行处理
MAX_WORKERS = 40 # 增加线程数以进一步加速
all_fund_data = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # 使用 list() 来等待所有结果返回
    results = executor.map(scrape_and_parse_fund, fund_codes_to_process)
    all_fund_data = list(results)

print("所有抓取任务完成。")

# --- 合并数据并写入单个 CSV 文件 ---

# 1. 收集所有唯一的字段名作为 CSV 的表头
all_keys = set()
for fund_data in all_fund_data:
    all_keys.update(fund_data.keys())

# 将字段排序，确保 "基金代码" 和 "状态" 在最前面
sorted_keys = ['基金代码', '状态']
for key in sorted(list(all_keys)):
    if key not in sorted_keys:
        sorted_keys.append(key)

output_filename = f"basic_info_all_funds_{timestamp}.csv"
output_path = os.path.join(output_base_dir, output_filename)

print(f"开始写入合并文件: {output_path}")

try:
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted_keys)
        
        # 写入表头
        writer.writeheader()
        
        # 写入所有行数据
        for fund_data in all_fund_data:
            # 填充缺失的字段，确保行与表头匹配
            writer.writerow({k: v for k, v in fund_data.items() if k in sorted_keys})

    print(f"成功保存合并文件: {output_path}")

except Exception as e:
    print(f"写入 CSV 文件时发生错误: {e}")

# Git推送结果到仓库
subprocess.run(['git', 'add', output_base_dir])
commit_result = subprocess.run(['git', 'commit', '-m', f"Add basic info file {output_filename} for {date_dir}"], capture_output=True, text=True)
if "nothing to commit" in commit_result.stdout or "nothing to commit" in commit_result.stderr:
    print("没有文件变动需要提交。")
else:
    print(commit_result.stdout)
    subprocess.run(['git', 'push'])
    print("Git 推送完成。")
