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

# 定义一个函数来处理单个基金的所有抓取和解析任务
def scrape_and_parse_fund(fund_code):
    """
    抓取单个基金的基本概况信息和费率信息，并返回一个包含键值对的字典。
    并行执行，保证速度。
    """
    
    result = {'基金代码': fund_code, '状态': '成功'}
    
    # --- 1. 抓取基本概况信息 (jbgk) ---
    jbgk_url = f"https://fundf10.eastmoney.com/jbgk_{fund_code}.html"
    try:
        response = requests.get(jbgk_url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', class_='info w790')
            
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all(['th', 'td'])
                    
                    # 提取基本概况表格数据 (Label: Value)
                    if len(cols) >= 2:
                        label = cols[0].text.strip().replace('：', '').replace(':', '')
                        value = cols[1].text.strip()
                        result[label] = value
                    if len(cols) == 4:
                        label2 = cols[2].text.strip().replace('：', '').replace(':', '')
                        value2 = cols[3].text.strip()
                        result[label2] = value2
            else:
                result['状态_概况'] = "抓取警告: 未找到概况表格"
        else:
            result['状态_概况'] = f"抓取失败: 概况页Status code {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        result['状态_概况'] = f"抓取失败: 概况页网络错误 ({e})"


    # --- 2. 抓取费率信息 (jjfl) ---
    fee_url = f"https://fundf10.eastmoney.com/jjfl_{fund_code}.html"
    try:
        fee_response = requests.get(fee_url, timeout=15)
        if fee_response.status_code == 200:
            fee_soup = BeautifulSoup(fee_response.text, 'html.parser')
            
            # a) 提取运作费用 (管理费、托管费、销售服务费)
            run_fee_h4 = fee_soup.find('h4', string=lambda t: t and '运作费用' in t)
            if run_fee_h4:
                run_fee_table = run_fee_h4.find_next('table', class_='comm jjfl')
                if run_fee_table:
                    # 运作费用的表格结构是 <th>Label</th><td>Value</td> ... 重复三次
                    run_fee_tds = run_fee_table.find_all(['td', 'th'])
                    if len(run_fee_tds) == 6:
                        # 确保键名统一并去掉"费率"
                        result[run_fee_tds[0].text.strip().replace('费率', '')] = run_fee_tds[1].text.strip()
                        result[run_fee_tds[2].text.strip().replace('费率', '')] = run_fee_tds[3].text.strip()
                        result[run_fee_tds[4].text.strip().replace('费率', '')] = run_fee_tds[5].text.strip()

            # b) 提取赎回费率
            shfl_h4 = fee_soup.find('h4', string=lambda t: t and '赎回费率' in t)
            if shfl_h4:
                shfl_table = shfl_h4.find_next('table', class_='comm jjfl')
                
                if shfl_table and shfl_table.find('tbody'):
                    fee_rows = shfl_table.find('tbody').find_all('tr')
                    
                    for row in fee_rows:
                        cols = row.find_all(['td', 'th'])
                        if len(cols) >= 3:
                            # 适用期限 (Period)
                            period = cols[1].text.strip()
                            # 赎回费率 (Rate)
                            rate = cols[2].text.strip()
                            
                            # 格式化键名，如 "赎回费率_小于7天"
                            key = f"赎回费率_{period.replace(' ', '')}"
                            result[key] = rate
                else:
                    result['状态_费率'] = "抓取警告: 未找到赎回费率表格"
            
        else:
            result['状态_费率'] = f"抓取失败: 费率页Status code {fee_response.status_code}"
            
    except requests.exceptions.RequestException as e:
        result['状态_费率'] = f"抓取失败: 费率页网络错误 ({e})"

    print(f"[{fund_code}] 状态: {result.get('状态', '未知')}. 概况: {result.get('状态_概况', 'OK')}. 费率: {result.get('状态_费率', 'OK')}")
    # 将最终的统一状态设为'失败'如果任何一个子任务失败
    if '失败' in result.get('状态_概况', '') or '失败' in result.get('状态_费率', ''):
        result['状态'] = '失败'
        
    return result

# --- 主执行逻辑 ---

# 收集所有基金代码
fund_codes_to_process = []
for filename in os.listdir(fund_data_dir):
    # 确保只处理有效的基金代码文件
    if filename.endswith('.csv') and re.match(r'^\d+\.csv$', filename):
        fund_code = filename.split('.')[0]
        fund_codes_to_process.append(fund_code)

print(f"找到 {len(fund_codes_to_process)} 个基金准备开始并行抓取 (包含费率信息)...")

# 使用线程池进行并行处理
MAX_WORKERS = 40 # 保持较高的线程数以维持加速
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
# 将费率相关的字段也放在前面，便于查看
fee_keys = [k for k in all_keys if '费' in k or '费率' in k]
fee_keys = sorted(list(set(fee_keys)))

# 重新组织表头顺序: 基金代码, 状态, 费率相关, 其他概况信息
final_keys = sorted_keys + fee_keys
for key in sorted(list(all_keys)):
    if key not in final_keys:
        final_keys.append(key)

output_filename = f"basic_info_and_fees_all_funds_{timestamp}.csv"
output_path = os.path.join(output_base_dir, output_filename)

print(f"开始写入合并文件: {output_path}")

try:
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=final_keys)
        
        # 写入表头
        writer.writeheader()
        
        # 写入所有行数据
        for fund_data in all_fund_data:
            writer.writerow({k: v for k, v in fund_data.items() if k in final_keys})

    print(f"成功保存合并文件: {output_path}")

except Exception as e:
    print(f"写入 CSV 文件时发生错误: {e}")

# Git推送结果到仓库
subprocess.run(['git', 'add', output_base_dir])
commit_result = subprocess.run(['git', 'commit', '-m', f"Add combined info and fee file {output_filename} for {date_dir}"], capture_output=True, text=True)
if "nothing to commit" in commit_result.stdout or "nothing to commit" in commit_result.stderr:
    print("没有文件变动需要提交。")
else:
    print(commit_result.stdout)
    subprocess.run(['git', 'push'])
    print("Git 推送完成。")
