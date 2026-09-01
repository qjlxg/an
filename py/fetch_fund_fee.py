import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# --- 配置常量 ---
BASE_URL = "http://fundf10.eastmoney.com/jjfl_{}.html"
OUTPUT_DIR = "" # 保持为空，输出到根目录
OUTPUT_FILE = "fund_fee_result.csv"
FUND_CODES_FILE = "C类.txt"
LIMIT_FUNDS = None 

# --- 辅助函数：将百分比字符串转换为浮点数 ---
def parse_rate(rate_str):
    """从 'X.XX%（每年）' 格式的字符串中提取数值。"""
    if isinstance(rate_str, str):
        match = re.search(r'(\d+\.?\d*)%', rate_str)
        if match:
            return float(match.group(1)) / 100
    return 0.0

# --- 1. 抓取与解析函数 (名称提取逻辑已修正) ---

def parse_fund_fees(html_content, fund_code):
    """
    解析天天基金网基金费率页面，修正了基金名称的提取逻辑。
    """
    try:
        try:
            soup = BeautifulSoup(html_content, 'lxml') 
        except Exception:
            soup = BeautifulSoup(html_content, 'html.parser')
        
        # 1. ***修正的基金名称提取逻辑*** fund_name = f"基金({fund_code})"
        try:
            # 名称位于 class="bs_jz" 的 div 内，通常是 h4 标签下的 a 标签
            name_tag = soup.find('div', class_='bs_jz')
            if name_tag:
                 # 查找 h4 标签下的 a 标签 (名称和代码)
                 h4_tag = name_tag.find('h4', class_='title')
                 if h4_tag:
                    name_text = h4_tag.text.strip()
                    # 匹配 "基金名称 (基金代码)" 格式，只取名称部分
                    match = re.match(r'(.+?)\s*\(\s*' + re.escape(fund_code) + r'\s*\)', name_text)
                    if match:
                        fund_name = match.group(1).strip()
                    # 备选：如果格式不匹配，尝试直接从 a 标签的 title 属性获取
                    elif h4_tag.find('a'):
                         fund_name = h4_tag.find('a').get('title', fund_name).split('(')[0].strip()
        except Exception as e:
            # print(f"名称提取失败: {e}")
            pass # 保持默认值 fund_code
        
        op_fees_data = {}
        redemption_fees = {}
        sub_fee_rate = '0.00%' 

        # 2. 费率数据提取 (逻辑不变)
        fee_containers = soup.find_all('div', class_='boxitem')

        for container in fee_containers:
            title_label = container.find('label', class_='left')
            if not title_label:
                continue

            title_text = title_label.text.strip()
            table = container.find('table', class_='jjfl')
            if not table:
                continue

            # 运作费用
            if "运作费用" == title_text:
                rows = table.find_all('tr')
                if rows:
                    cols = rows[0].find_all('td')
                    if len(cols) >= 6: 
                        op_fees_data['管理费率（每年）'] = cols[1].text.strip()
                        op_fees_data['托管费率（每年）'] = cols[3].text.strip()
                        op_fees_data['销售服务费率（每年）'] = cols[5].text.strip()
            
            # 赎回费率
            elif "赎回费率" == title_text:
                rows = table.find('tbody').find_all('tr') if table.find('tbody') else table.find_all('tr')[1:] 
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 3:
                        term = cols[1].text.strip()
                        rate = cols[2].text.strip()
                        
                        if '小于7天' in term:
                            redemption_fees['赎回费率（小于7天）'] = rate
                        elif '大于等于7天' in term:
                            redemption_fees['赎回费率（大于等于7天）'] = rate
            
        # 3. 整合数据
        data = {
            '基金代码': fund_code,
            '基金名称': fund_name,
            '申购费率（前端，优惠）': sub_fee_rate,
            **op_fees_data,
            **redemption_fees
        }

        # 检查关键字段是否成功获取
        if not data.get('管理费率（每年）'):
             return None

        return data

    except Exception as e:
        print(f"处理基金 {fund_code} 时发生错误: {e}")
        return None

# --- 2. 主执行逻辑 (保持不变) ---

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

    output_path = OUTPUT_FILE
    
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
        
        # --- 计算总运作费率并排序 ---
        # 确保所有费率列都存在，如果不存在则填充 None
        for col in ['管理费率（每年）', '托管费率（每年）', '销售服务费率（每年）']:
             if col not in df.columns:
                 df[col] = '0.00%（每年）' # 如果缺少，则默认为0
                 
        df['管理费率（数值）'] = df['管理费率（每年）'].apply(parse_rate)
        df['托管费率（数值）'] = df['托管费率（每年）'].apply(parse_rate)
        df['销售服务费率（数值）'] = df['销售服务费率（每年）'].apply(parse_rate)
        
        df['运作总费率（每年）'] = df['管理费率（数值）'] + df['托管费率（数值）'] + df['销售服务费率（数值）']
        
        df = df.sort_values(by='运作总费率（每年）', ascending=True)

        # 重新组织列
        final_columns = [
            '基金代码',
            '基金名称',
            '运作总费率（每年）', 
            '管理费率（每年）', 
            '托管费率（每年）', 
            '销售服务费率（每年）', 
            '申购费率（前端，优惠）', 
            '赎回费率（小于7天）', 
            '赎回费率（大于等于7天）'
        ]
        
        df = df[[col for col in final_columns if col in df.columns]]

        print("\n" + "="*50)
        print(f"✅ 抓取完成！共成功抓取 {len(all_data)} 只基金的数据。")
        print(f"文件已保存到: {output_path} (根目录)，已按运作总费率升序排序。")
        print("="*50)

    # 写入 CSV 文件
    df.to_csv(output_path, index=False, encoding='utf_8_sig') 

if __name__ == '__main__':
    main()
