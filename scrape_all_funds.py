import os
import requests
from bs4 import BeautifulSoup
import csv
import datetime
import pytz
import subprocess

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

# 循环处理fund_data目录下的所有CSV文件
for filename in os.listdir(fund_data_dir):
    if filename.endswith('.csv'):
        # 从文件名提取基金代码（如014194.csv -> 014194）
        fund_code = filename.split('.')[0]
        
        # 构建URL
        url = f"https://fundf10.eastmoney.com/jbgk_{fund_code}.html"
        
        # 抓取页面内容
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            continue
        
        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 找到基本概况表格（class="info w790"）
        table = soup.find('table', class_='info w790')
        if not table:
            print(f"No table found for {fund_code}")
            continue
        
        # 提取表格数据
        data = []
        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all(['th', 'td'])
            row_data = [col.text.strip() for col in cols]
            data.append(row_data)
        
        # 保存为CSV文件，文件名加上时间戳
        output_filename = f"basic_info_{fund_code}_{timestamp}.csv"
        output_path = os.path.join(output_base_dir, output_filename)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
        
        print(f"Saved {output_path}")

# Git推送结果到仓库（假设仓库已配置好远程）
subprocess.run(['git', 'add', output_base_dir])
subprocess.run(['git', 'commit', '-m', f"Add basic info files for {date_dir}"])
subprocess.run(['git', 'push'])
