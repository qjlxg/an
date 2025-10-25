import os
import requests
from bs4 import BeautifulSoup
import csv
import datetime
import pytz
import subprocess
from concurrent.futures import ThreadPoolExecutor
import re

# --- 配置 ---
# 设置上海时区
shanghai_tz = pytz.timezone('Asia/Shanghai')

# 获取当前日期和时间（上海时区）
now = datetime.datetime.now(shanghai_tz)
date_dir = now.strftime('%Y%m%d')
timestamp = now.strftime('%Y%m%d_%H%M%S')

# 基金数据目录
fund_data_dir = 'fund_data'

# 输出目录
output_base_dir = date_dir
os.makedirs(output_base_dir, exist_ok=True)

# 定义筛选的低费率阈值
MAX_MANAGEMENT_FEE = 1.00 # 管理费率不超过 1.00%
MAX_CUSTODIAN_FEE = 0.20   # 托管费率不超过 0.20%
# 最关键：持有7天以上的赎回费率不超过 0.00% (即为 0%)
MAX_REDEMPTION_FEE_7D = 0.00 
# --- end 配置 ---


# --- 工具函数 ---

def clean_fee_rate(fee_str):
    """从费率字符串中提取数值（百分比形式）并转换为浮点数。"""
    if not fee_str:
        return None
    
    # 查找所有数字和百分号
    match = re.search(r'(\d+\.?\d*)%', fee_str)
    if match:
        try:
            # 返回百分数形式的浮点数 (如 1.50%)
            return float(match.group(1))
        except ValueError:
            return None
    return None

# --- 抓取和解析函数 (与上一版本相同) ---

def scrape_and_parse_fund(fund_code):
    """
    抓取单个基金的基本概况信息和费率信息，并返回一个包含键值对的字典。
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
                    run_fee_tds = run_fee_table.find_all(['td', 'th'])
                    if len(run_fee_tds) == 6:
                        result[run_fee_tds[0].text.strip().replace('费率', '')] = run_fee_tds[1].text.strip()
                        result[run_fee_tds[2].text.strip().replace('费率', '')] = run_fee_tds[3].text.strip()
                        result[run_fee_tds[4].text.strip().replace('费率', '')] = run_fee_tds[5].text.strip()

            # b) 提取赎回费率 (关键)
            shfl_h4 = fee_soup.find('h4', string=lambda t: t and '赎回费率' in t)
            if shfl_h4:
                shfl_table = shfl_h4.find_next('table', class_='comm jjfl')
                
                if shfl_table and shfl_table.find('tbody'):
                    fee_rows = shfl_table.find('tbody').find_all('tr')
                    
                    for row in fee_rows:
                        cols = row.find_all(['td', 'th'])
                        if len(cols) >= 3:
                            period = cols[1].text.strip()
                            rate = cols[2].text.strip()
                            
                            key = f"赎回费率_{period.replace(' ', '')}"
                            result[key] = rate
                else:
                    result['状态_费率'] = "抓取警告: 未找到赎回费率表格"
            
        else:
            result['状态_费率'] = f"抓取失败: 费率页Status code {fee_response.status_code}"
            
    except requests.exceptions.RequestException as e:
        result['状态_费率'] = f"抓取失败: 费率页网络错误 ({e})"

    print(f"[{fund_code}] 状态: {result.get('状态', '未知')}. 概况: {result.get('状态_概况', 'OK')}. 费率: {result.get('状态_费率', 'OK')}")
    # 统一设置最终状态
    if '失败' in result.get('状态_概况', '') or '失败' in result.get('状态_费率', ''):
        result['状态'] = '失败'
    elif '警告' in result.get('状态_概况', '') or '警告' in result.get('状态_费率', ''):
         result['状态'] = '警告'
        
    return result

# --- 主执行逻辑 ---

# 收集所有基金代码
fund_codes_to_process = []
for filename in os.listdir(fund_data_dir):
    if filename.endswith('.csv') and re.match(r'^\d+\.csv$', filename):
        fund_code = filename.split('.')[0]
        fund_codes_to_process.append(fund_code)

print(f"找到 {len(fund_codes_to_process)} 个基金准备开始并行抓取...")

# 使用线程池进行并行处理
MAX_WORKERS = 10
all_fund_data = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    results = executor.map(scrape_and_parse_fund, fund_codes_to_process)
    all_fund_data = list(results)

print("所有抓取任务完成。")

# --- 1. 合并数据并写入单个 CSV 文件 ---

# 1. 收集所有唯一的字段名作为 CSV 的表头
all_keys = set()
for fund_data in all_fund_data:
    all_keys.update(fund_data.keys())

# 2. 组织 CSV 文件的最终表头
sorted_keys = ['基金代码', '状态']
fee_keys = [k for k in all_keys if '费' in k or '费率' in k]
fee_keys = sorted(list(set(fee_keys)))

final_keys = sorted_keys + fee_keys
for key in sorted(list(all_keys)):
    if key not in final_keys:
        final_keys.append(key)

output_filename = f"basic_info_and_fees_all_funds_{timestamp}.csv"
output_path = os.path.join(output_base_dir, output_filename)

try:
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=final_keys)
        writer.writeheader()
        for fund_data in all_fund_data:
            writer.writerow({k: v for k, v in fund_data.items() if k in final_keys})

    print(f"成功保存合并文件: {output_path}")

except Exception as e:
    print(f"写入主 CSV 文件时发生错误: {e}")

# --- 2. 筛选和生成低费率报告文件 (核心目的) ---

low_fee_funds = []
print("\n--- 低费率基金筛选结果 (开始筛选) ---")
print(f"筛选条件：管理费率 <= {MAX_MANAGEMENT_FEE}%, 托管费率 <= {MAX_CUSTODIAN_FEE}%, 赎回费率(>=7天) <= {MAX_REDEMPTION_FEE_7D}%")

for fund in all_fund_data:
    if fund.get('状态') != '成功':
        continue
        
    mgmt_fee = clean_fee_rate(fund.get('管理费', ''))
    cust_fee = clean_fee_rate(fund.get('托管费', ''))
    redemption_fee_7d = clean_fee_rate(fund.get('赎回费率_大于等于7天', ''))
    
    is_low_fee = True
    
    if mgmt_fee is None or mgmt_fee > MAX_MANAGEMENT_FEE:
        is_low_fee = False
    
    if cust_fee is None or cust_fee > MAX_CUSTODIAN_FEE:
        is_low_fee = False
        
    if redemption_fee_7d is None or redemption_fee_7d > MAX_REDEMPTION_FEE_7D:
        is_low_fee = False
    
    if is_low_fee:
        fund_name = fund.get('基金全称', fund.get('基金简称', ''))
        low_fee_funds.append({
            '代码': fund['基金代码'],
            '名称': fund_name,
            '管理费': f"{mgmt_fee:.2f}%" if mgmt_fee is not None else 'N/A',
            '托管费': f"{cust_fee:.2f}%" if cust_fee is not None else 'N/A',
            '赎回费(大于等于7天)': f"{redemption_fee_7d:.2f}%" if redemption_fee_7d is not None else 'N/A',
        })

# 写入低费率报告文件
report_filename = f"low_fee_funds_report_{timestamp}.csv"
report_path = os.path.join(output_base_dir, report_filename)

if low_fee_funds:
    report_keys = ['代码', '名称', '管理费', '托管费', '赎回费(大于等于7天)']
    try:
        with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=report_keys)
            writer.writeheader()
            writer.writerows(low_fee_funds)
        print(f"\n找到 {len(low_fee_funds)} 个低费率基金，已保存到报告文件: {report_path}")
    except Exception as e:
        print(f"\n写入低费率报告文件时发生错误: {e}")
else:
    print("\n没有找到完全符合低费率条件的基金。")

# --- 3. Git推送结果到仓库 ---

# git add . 会同时添加主数据文件和低费率报告文件
subprocess.run(['git', 'add', output_base_dir]) 
commit_result = subprocess.run(['git', 'commit', '-m', f"Add combined data and low-fee report for {date_dir}"], capture_output=True, text=True)
if "nothing to commit" in commit_result.stdout or "nothing to commit" in commit_result.stderr:
    print("\n没有文件变动需要提交。")
else:
    print(commit_result.stdout)
    subprocess.run(['git', 'push'])
    print("Git 推送完成。")
