# In[]:
#!/usr/bin/env python
# coding: utf-8
# encoding=utf-8
import pandas as pd
import requests
from lxml import etree
import re
import time
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import numpy as np

# --- 1. 读取 C 类基金代码列表 ---
try:
    with open('C类.txt', 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f if line.strip()]
    
    # 移除可能的列标题 'code'
    if all_lines and all_lines[0].lower() in ['code', '基金代码']:
        c_class_codes = all_lines[1:]
    else:
        c_class_codes = all_lines
        
except FileNotFoundError:
    print("错误：未找到 'C类.txt' 文件，请确保文件位于脚本运行目录下！")
    sys.exit(1)
except Exception as e:
    print(f"读取文件时发生错误: {e}")
    sys.exit(1)

if not c_class_codes:
    print("基金代码列表为空，脚本将提前退出。")
    sys.exit(1)

# --- 2. 参数设置 ---
season = 1  # 爬取最新一期（即 div[1]）的持仓数据
MAX_WORKERS = 20  # 并发线程数
total = len(c_class_codes)

# 爬虫 headers
head = {
    "Cookie": "EMFUND1=null; EMFUND2=null; EMFUND3=null; EMFUND4=null; EMFUND5=null; EMFUND6=null; EMFUND7=null; EMFUND8=null; EMFUND0=null; st_si=44023331838789; st_asi=delete; EMFUND9=08-16 22:04:25@#$%u4E07%u5BB6%u65B0%u5229%u7075%u6DF7%u5408@%23%24519191; ASP.NET_SessionId=45qdofapdlm1hlgxapxuxhe1; st_pvi=87492384111747; st_sp=2020-08-16%2000%3A05%3A17; st_inirUrl=http%3A%2F%2Ffund.eastmoney.com%2Fdata%2Ffundranking.html; st_sn=12; st_psi=2020081622103685-0-6169905557",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36"
}

def fetch_fund_holdings(code, season, head):
    """
    爬取单个基金的持仓数据和简称，并解析。
    集成了最新的简称提取逻辑（XPath定位）和持仓数据提取逻辑。
    """
    url = f"http://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=jjcc&code={code}&topline=10&year=&month=&rt=0.5032668912422176"
    fund_name = '简称缺失'
    
    # 尝试重试机制，增加稳定性
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=head, timeout=10)
            text = response.text
            
            # --- 1. 基金简称提取 (最终稳定版: XPath定位) ---
            
            # 提取 HTML 内容块 (包含基金简称和持仓表格)
            div_match = re.findall('content:\\"(.*)\\",arryear', text)
            
            # 失败时尝试备用方案 (从 name:'...' 模式提取)
            if not div_match:
                name_match = re.search(r"name:'(.*?)'", text)
                if name_match:
                    fund_name = name_match.group(1).strip()
                break # 退出重试循环，返回空持仓
            
            div = div_match[0]
            # 构造完整HTML，便于 lxml 解析
            html_body = '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>test</title></head><body>%s</body></html>' % (div)
            html = etree.HTML(html_body)
            
            # XPath: //h4/label[@class='left']/a[1] 定位基金简称
            name_list = html.xpath('//h4/label/a[1]/text()')
            if name_list:
                fund_name = name_list[0].strip()
            # -----------------------------------------------------
            
            # --- 2. 提取持仓数据 ---
            xpath_base = f'//div[{season}]/div/table/tbody/tr'
            rows = html.xpath(xpath_base)
            
            stock_one_fund = []
            for row in rows:
                stock_name_list = row.xpath('./td[3]/a/text()')
                if not stock_name_list:
                    continue
                
                stock_name = stock_name_list[0].strip()
                
                # 股票数据提取逻辑：定位最后四列中的最后三列
                data_tds = row.xpath('./td[position() >= last()-3]') 
                
                money_data = []
                for td in data_tds:
                    text = "".join(td.xpath('.//text()')).strip()
                    text = text.replace('---','0').replace(',','').replace('%','')
                    try:
                        money_data.append(float(text))
                    except ValueError:
                        pass
                
                # 确保有足够的数字：占净值比例、持股数、持仓市值
                if len(money_data) >= 3:
                    stock_one_fund.append([stock_name, 
                                           money_data[-3], # 占净值比例
                                           money_data[-2], # 持股数_万
                                           money_data[-1]]) # 持仓市值_万
            
            return code, fund_name, stock_one_fund # 成功获取，退出重试
            
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2) # 超时重试
                continue
            else:
                return code, fund_name, []
        except Exception:
            break # 捕获其他解析错误，不重试，返回默认值
            
    return code, fund_name, []


# --- 3. 并发爬取持仓数据 ---
print(f"从 C类.txt 中读取到 {total} 支基金，开始并发爬取持仓...")
start_time = time.time()
futures = []
all_holdings = [] 
success_count = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for code in c_class_codes:
        future = executor.submit(fetch_fund_holdings, code, season, head)
        futures.append(future)

    for i, future in enumerate(as_completed(futures)):
        code, fund_name, holdings_list = future.result()
        
        if holdings_list:
            success_count += 1
            
        if total > 0 and (i + 1) % (total // 10 + 1) == 0:
            print(f"进度: {i+1}/{total} ({((i+1)/total)*100:.1f}%)")
            
        for holding in holdings_list:
            all_holdings.append([code, fund_name] + holding)


end_time = time.time()
print("\n" + "=" * 30 + " 并发获取基金持仓数据完成 " + "=" * 30)
print(f"成功获取 {success_count} 支基金的持仓数据，总耗时: {end_time - start_time:.2f} 秒")

# --- 4. 整合结果并保存 (优化版：增加计算列和排序) ---

# 创建包含所有持仓记录的 DataFrame
df_holdings = pd.DataFrame(
    all_holdings,
    columns=['基金代码', '基金简称', '股票简称', '占净值比例', '持股数_万', '持仓市值_万']
)

# 1. 数据清洗和类型转换
df_holdings['占净值比例'] = pd.to_numeric(df_holdings['占净值比例'], errors='coerce')
df_holdings['持股数_万'] = pd.to_numeric(df_holdings['持股数_万'], errors='coerce')
df_holdings['持仓市值_万'] = pd.to_numeric(df_holdings['持仓市值_万'], errors='coerce')
df_holdings.dropna(subset=['持仓市值_万'], inplace=True) 

# 2. 增加计算列：持仓市值（亿）
df_holdings['持仓市值_亿'] = df_holdings['持仓市值_万'] / 10000

# 3. 汇总计算：计算每支基金的总股票持仓市值 (作为基金规模估算)
df_fund_sum = df_holdings.groupby(['基金代码', '基金简称'])['持仓市值_万'].sum().reset_index()
df_fund_sum.rename(columns={'持仓市值_万': '总股票持仓市值_万'}, inplace=True)

# 4. 合并总市值到明细表中
df_holdings = pd.merge(df_holdings, df_fund_sum, on=['基金代码', '基金简称'], how='left')

# 5. 排序：按基金的总市值降序，再按持仓比例降序
df_holdings.sort_values(by=['总股票持仓市值_万', '占净值比例'], 
                        ascending=[False, False], 
                        inplace=True)

# 6. 数值格式化 (保留两位/四位小数)
df_holdings['占净值比例'] = df_holdings['占净值比例'].round(2)
df_holdings['持仓市值_亿'] = df_holdings['持仓市值_亿'].round(4)
df_holdings['持仓市值_万'] = df_holdings['持仓市值_万'].round(2)
df_holdings['总股票持仓市值_万'] = df_holdings['总股票持仓市值_万'].round(2)

# 7. 调整列顺序
final_columns = ['基金代码', '基金简称', '总股票持仓市值_万', '股票简称', '占净值比例', '持股数_万', '持仓市值_万', '持仓市值_亿']
df_holdings = df_holdings[final_columns]

# 保存文件
current_time_shanghai = datetime.now()
timestamp = current_time_shanghai.strftime("%Y%m%d_%H%M%S")
year_month = current_time_shanghai.strftime("%Y%m")
output_dir = os.path.join(year_month)
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(output_dir, f"C类基金持仓明细_优化_{timestamp}.csv")
df_holdings.to_csv(filename, encoding="utf_8_sig", index=False)
print(f"所有 C 类基金的持仓明细已保存至：{filename}")
