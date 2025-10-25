# -*- coding: utf-8 -*-
import os
import requests
from bs4 import BeautifulSoup
import csv
import datetime
import pytz
import random
import time
from concurrent.futures import ThreadPoolExecutor
import re

# ==================== 配置 ====================
shanghai_tz = pytz.timezone('Asia/Shanghai')
now = datetime.datetime.now(shanghai_tz)
date_only_str = now.strftime('%Y%m%d')
month_dir = now.strftime('%Y%m')
timestamp = now.strftime('%Y%m%d_%H%M%S')

fund_data_dir = 'fund_data'
output_base_dir = month_dir
os.makedirs(output_base_dir, exist_ok=True)

# 只要 7 天以上赎回费 = 0.00%，管理费/托管费放宽
MAX_MANAGEMENT_FEE = 99.0      # 实际上不限制
MAX_CUSTODIAN_FEE = 0.20       # 仍保留 0.20% 限制
MAX_REDEMPTION_FEE_7D = 0.00    # 必须 0%

# 防反爬请求头
HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/128.0.0.0 Safari/537.36')
}
# ==============================================

def clean_fee_rate(fee_str):
    """提取百分比数字（如 '1.50%' → 1.5）"""
    if not fee_str:
        return None
    m = re.search(r'(\d+\.?\d*)%', fee_str)
    return float(m.group(1)) if m else None


def scrape_and_parse_fund(fund_code):
    """抓取单只基金的概况 + 费率，返回 dict"""
    # 随机延时防封
    time.sleep(random.uniform(0.15, 0.6))

    result = {'基金代码': fund_code, '状态': '成功'}

    # ---------- 1. 基本概况 ----------
    jbgk_url = f"https://fundf10.eastmoney.com/jbgk_{fund_code}.html"
    try:
        r = requests.get(jbgk_url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            result['状态_概况'] = f"抓取失败: 概况页 {r.status_code}"
        else:
            soup = BeautifulSoup(r.text, 'html.parser')
            tbl = soup.find('table', class_='info w790')
            if tbl:
                for row in tbl.find_all('tr'):
                    cols = row.find_all(['th', 'td'])
                    if len(cols) >= 2:
                        k = cols[0].get_text(strip=True).rstrip('：:')
                        v = cols[1].get_text(strip=True)
                        result[k] = v
                    if len(cols) == 4:
                        k2 = cols[2].get_text(strip=True).rstrip('：:')
                        v2 = cols[3].get_text(strip=True)
                        result[k2] = v2
            else:
                result['状态_概况'] = "抓取警告: 未找到概况表格"
    except requests.RequestException as e:
        result['状态_概况'] = f"抓取失败: 网络错误 ({e})"

    # ---------- 2. 费率 ----------
    fee_url = f"https://fundf10.eastmoney.com/jjfl_{fund_code}.html"
    try:
        r = requests.get(fee_url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            result['状态_费率'] = f"抓取失败: 费率页 {r.status_code}"
        else:
            soup = BeautifulSoup(r.text, 'html.parser')

            # a) 运作费用（管理/托管/销售服务）
            h4 = soup.find('h4', string=lambda t: t and '运作费用' in t)
            if h4:
                tbl = h4.find_next('table', class_='comm jjfl')
                if tbl:
                    tds = tbl.find_all(['td', 'th'])
                    if len(tds) == 6:
                        for i in range(0, 6, 2):
                            k = tds[i].get_text(strip=True).replace('费率', '')
                            v = tds[i+1].get_text(strip=True)
                            result[k] = v

            # b) 赎回费率（多写法统一键名）
            h4 = soup.find('h4', string=lambda t: t and '赎回费率' in t)
            if h4:
                tbl = h4.find_next('table', class_='comm jjfl')
                if tbl and tbl.find('tbody'):
                    for row in tbl.find('tbody').find_all('tr'):
                        cols = row.find_all(['td', 'th'])
                        if len(cols) < 3:
                            continue
                        period = cols[1].get_text(strip=True)
                        rate   = cols[2].get_text(strip=True)

                        # 统一键名：大于等于7天 → 赎回费率_7D0
                        if any(p in period for p in
                               ['大于等于7天', '7天及以上', '≥7天', '7天以上', '7天-1年', '7天＜持有期限＜1年']):
                            result['赎回费率_7D0'] = rate
                        else:
                            result[f"赎回费率_{period.replace(' ', '')}"] = rate
                else:
                    result['状态_费率'] = "抓取警告: 未找到赎回费率表格"
    except requests.RequestException as e:
        result['状态_费率'] = f"抓取失败: 网络错误 ({e})"

    # 统一状态
    if any('失败' in result.get(k, '') for k in ('状态_概况', '状态_费率')):
        result['状态'] = '失败'
    elif any('警告' in result.get(k, '') for k in ('状态_概况', '状态_费率')):
        result['状态'] = '警告'
    return result


# ==================== 主逻辑 ====================
fund_codes = []
for fn in os.listdir(fund_data_dir):
    if fn.endswith('.csv') and re.match(r'^\d+\.csv$', fn):
        fund_codes.append(fn.split('.')[0])

print(f"[{now.strftime('%H:%M:%S')}] 共 {len(fund_codes)} 只基金，准备并行抓取...")
MAX_WORKERS = 40
all_data = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    all_data = list(pool.map(scrape_and_parse_fund, fund_codes))

print(f"[{datetime.datetime.now(shanghai_tz).strftime('%H:%M:%S')}] 抓取完毕")

# ---------- 1. 合并全量 CSV ----------
all_keys = set()
for d in all_data:
    all_keys.update(d.keys())

sorted_keys = ['基金代码', '状态']
fee_keys = sorted({k for k in all_keys if '费' in k or '费率' in k})
final_keys = sorted_keys + fee_keys + sorted({k for k in all_keys if k not in sorted_keys + fee_keys})

out_path = os.path.join(output_base_dir, f"basic_info_and_fees_all_funds_{timestamp}.csv")
with open(out_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=final_keys)
    w.writeheader()
    for d in all_data:
        w.writerow({k: d.get(k, '') for k in final_keys})
print(f"[{datetime.datetime.now(shanghai_tz).strftime('%H:%M:%S')}] 全量文件已保存 → {out_path}")

# ---------- 2. 低费率（7天免赎回）报告 ----------
print(f"[{datetime.datetime.now(shanghai_tz).strftime('%H:%M:%S')}] 开始筛选【7天以上免赎回费】基金...")
report_funds = []

for d in all_data:
    if d.get('状态') != '成功':
        continue

    mgmt = clean_fee_rate(d.get('管理费', ''))
    cust = clean_fee_rate(d.get('托管费', ''))
    red7 = clean_fee_rate(d.get('赎回费率_7D0', ''))   # 统一键

    # 管理费放宽，托管费仍限制，赎回必须 0%
    if (mgmt is not None and mgmt > MAX_MANAGEMENT_FEE) or \
       (cust is not None and cust > MAX_CUSTODIAN_FEE) or \
       (red7 is None or red7 > MAX_REDEMPTION_FEE_7D):
        continue

    name = d.get('基金全称') or d.get('基金简称') or ''
    svc  = clean_fee_rate(d.get('销售服务费', ''))

    report_funds.append({
        '代码': d['基金代码'],
        '名称': name,
        '管理费': f"{mgmt:.2f}%" if mgmt is not None else 'N/A',
        '托管费': f"{cust:.2f}%" if cust is not None else 'N/A',
        '销售服务费': f"{svc:.2f}%" if svc is not None else '0.00%',
        '赎回费(>=7天)': '0.00%',
        '基金类型': d.get('基金类型', ''),
        '成立日期': d.get('成立日期', '')
    })

report_path = os.path.join(output_base_dir, f"zero_redemption_funds_{timestamp}.csv")
report_cols = ['代码','名称','管理费','托管费','销售服务费','赎回费(>=7天)','基金类型','成立日期']

if report_funds:
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=report_cols)
        w.writeheader()
        w.writerows(report_funds)
    print(f"[{datetime.datetime.now(shanghai_tz).strftime('%H:%M:%S')}] 找到 {len(report_funds)} 只【7天免赎回】基金 → {report_path}")
else:
    print(f"[{datetime.datetime.now(shanghai_tz).strftime('%H:%M:%S')}] 未找到符合条件的基金")

print(f"[{datetime.datetime.now(shanghai_tz).strftime('%H:%M:%S')}] 全部完成")
