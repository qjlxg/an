import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# --- 配置常量 ---
BASE_URL = "http://fundf10.eastmoney.com/jjfl_{}.html"
OUTPUT_DIR = "fund_data"
OUTPUT_FILE = "fund_fee_result.csv"
FUND_CODES_FILE = "C类.txt"
# 调试抓取的基金数量 (GitHub Actions运行时，如需抓取全部，请将此值设为 None 或一个很大的数)
LIMIT_FUNDS = 601 

# --- 1. 抓取与解析函数 ---

def parse_fund_fees(html_content, fund_code):
    """
    解析天天基金网基金费率页面，使用直接定位父容器和标签文本的方法。
    """
    try:
        # 使用 lxml 解析器，如果未安装则回退到默认
        try:
            soup = BeautifulSoup(html_content, 'lxml') 
        except Exception:
            soup = BeautifulSoup(html_content, 'html.parser')
        
        # 1. 提取基金名称 
        fund_name = f"基金({fund_code})"
        try:
            # 查找页面的主标题区域
            name_tag = soup.find('div', class_='box_p')
            if name_tag:
                 name_link = name_tag.find('h4').find('a')
                 if name_link:
                    fund_name = name_link.text.split('(')[0].strip()
        except Exception:
            pass
        
        op_fees_data = {}
        redemption_fees = {}
        # C 类基金申购费率通常为 0.00%，设定默认值
        sub_fee_rate = '0.00%' 

        # 核心修正: 查找包含所有费率信息的 boxitem 容器
        fee_containers = soup.find_all('div', class_='boxitem')

        for container in fee_containers:
            # 查找容器内的标题 label
            title_label = container.find('label', class_='left')
            if not title_label:
                continue

            title_text = title_label.text.strip()
            table = container.find('table', class_='jjfl')
            if not table:
                continue

            # 2. 提取运作费用 (Management Fee, Custodian Fee, Sales Service Fee)
            if "运作费用" == title_text:
                rows = table.find_all('tr')
                if rows:
                    # 运作费用通常在第一个 tbody/tr 中
                    cols = rows[0].find_all('td')
                    if len(cols) >= 6: 
                        # 数据位于表格的第一行中的第 2, 4, 6 列
                        op_fees_data['管理费率（每年）'] = cols[1].text.strip()
                        op_fees_data['托管费率（每年）'] = cols[3].text.strip()
                        op_fees_data['销售服务费率（每年）'] = cols[5].text.strip()
            
            # 3. 提取赎回费率
            elif "赎回费率" == title_text:
                # 尝试获取 tbody，如果不存在，则从 tr 开始（跳过表头）
                rows = table.find('tbody').find_all('tr') if table.find('tbody') else table.find_all('tr')[1:] 
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 3:
                        # 期限在第 2 列，费率在第 3 列
                        term = cols[1].text.strip()
                        rate = cols[2].text.strip()
                        
                        if '小于7天' in term:
                            redemption_fees['赎回费率（小于7天）'] = rate
                        elif '大于等于7天' in term:
                            redemption_fees['赎回费率（大于等于7天）'] = rate
            
            # 4. 提取申购费率 (虽然已设默认值，但可以尝试确认申购费)
            # 申购费率（前端）表格也存在，但 C 类通常为 0.00%，此处沿用默认值。
            
        # 5. 整合数据
        data = {
            '基金代码': fund_code,
            '基金名称': fund_name,
            '申购费率（前端，优惠）': sub_fee_rate,
            **op_fees_data,
            **redemption_fees
        }

        # 检查关键字段是否成功获取
        if not data.get('管理费率（每年）'):
             print(f"警告：基金 {fund_code} 抓取数据不完整（缺少管理费率）。")
             return None

        return data

    except Exception as e:
        print(f"处理基金 {fund_code} 时发生错误: {e}")
        return None

# --- 2. 主执行逻辑 (保持不变，已包含失败时生成空 CSV 的逻辑) ---
def fetch_fund_data(fund_code):
    """
    从天天基金网获取单个基金的费率页面并解析数据。
    """
    url = BASE_URL.format(fund_code)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.encoding = 'utf-8'
        response.raise_for_status() 

        data = parse_fund_fees(response.text, fund_code)
        
        if data:
            print(f"✅ 成功抓取并解析基金 {fund_code}: {data.get('基金名称', '')}")
        
        return data

    except requests.exceptions.HTTPError:
        print(f"❌ 抓取基金 {fund_code} 失败: HTTP 错误 {response.status_code}")
    except requests.exceptions.RequestException:
        print(f"❌ 抓取基金 {fund_code} 失败: 请求超时或连接错误")
    except Exception:
        print(f"❌ 抓取基金 {fund_code} 发生未知错误")
        
    return None

def main():
    fund_codes = []
    try:
        with open(FUND_CODES_FILE, 'r', encoding='utf-8') as f:
            fund_codes = [line.strip() for line in f.readlines() if line.strip() and line.strip() != 'code']
    except FileNotFoundError:
        print(f"错误: 未找到文件 {FUND_CODES_FILE}。请确保文件存在。")
        return

    if not fund_codes:
        print("错误: 文件中未找到基金代码。")
        return
    
    codes_to_fetch = fund_codes[:LIMIT_FUNDS] if LIMIT_FUNDS else fund_codes
    print(f"成功读取 {len(fund_codes)} 个代码。开始并行抓取前 {len(codes_to_fetch)} 只基金的费率数据...")

    all_data = []
    with ThreadPoolExecutor(max_workers=5) as executor: 
        futures = [executor.submit(fetch_fund_data, code) for code in codes_to_fetch]
        
        for future in futures:
            result = future.result()
            if result:
                all_data.append(result)

    # 3. 数据处理与保存 (无论成功与否，都要创建文件)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    # 定义表头
    columns_order = [
        '基金代码', '基金名称', '管理费率（每年）', '托管费率（每年）', 
        '销售服务费率（每年）', '申购费率（前端，优惠）', '赎回费率（小于7天）', 
        '赎回费率（大于等于7天）'
    ]
    
    if not all_data:
        print("\n未能成功抓取任何数据。")
        df = pd.DataFrame(columns=columns_order)
        print(f"\n警告：抓取失败，已创建一个空文件 (含表头): {output_path}。")
    else:
        df = pd.DataFrame(all_data)
        
        final_columns = [col for col in columns_order if col in df.columns]
        df = df[final_columns]

        print("\n" + "="*50)
        print(f"✅ 抓取完成！共成功抓取 {len(all_data)} 只基金的数据。")
        print(f"文件已保存到: {output_path}")
        print("="*50)

    # 确保目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 写入 CSV 文件
    df.to_csv(output_path, index=False, encoding='utf_8_sig') 

if __name__ == '__main__':
    main()
