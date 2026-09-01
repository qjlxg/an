# fund_scraper_g.py (Final version with sorting)

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import concurrent.futures

# 定义请求头，模拟浏览器访问，提高成功率
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_fund_codes(filepath="C类.txt"):
    """
    从C类.txt文件中读取基金代码列表，并跳过可能的标题行。
    """
    if not os.path.exists(filepath):
        print(f"错误: 找不到文件 {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    codes = [line for line in lines if line and line.lower() != 'code']
    
    return [code for code in codes if len(code) >= 6]

def scrape_fund_info(fund_code):
    """
    抓取单个基金的基本概况信息。
    """
    base_url = "https://fundf10.eastmoney.com/jbgk_{}.html"
    url = base_url.format(fund_code)
    
    print(f"-> 正在抓取基金代码: {fund_code}")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        
        if response.status_code != 200:
            print(f"   警告: 基金 {fund_code} 状态码 {response.status_code}. 跳过.")
            return None

        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        info_table = soup.find('table', class_='info')
        
        if not info_table:
            print(f"   警告: 基金 {fund_code} 页面未找到基本概况信息表. 跳过.")
            return None

        fund_data = {'基金代码': fund_code}
        
        cells = [elem.get_text(strip=True) for elem in info_table.find_all(['th', 'td'])]
        
        for i in range(0, len(cells), 2):
            if i + 1 < len(cells):
                key = cells[i].strip('：')
                value = cells[i+1]
                
                # 清理冗余信息
                if key in ['基金代码', '最高申购费率', '最高赎回费率', '最高认购费率']:
                    value = value.split('（')[0].strip()
                    if '优惠费率' in value:
                        value = value.split('天天基金优惠费率')[0].strip()
                
                if key in ['份额规模', '成立来分红', '资产规模']:
                    if '（截止至：' in value:
                        value = value.split('（截止至：')[0].strip()
                    elif '（' in value and '次' in value:
                        value = value.split('（')[0].strip()
                
                fund_data[key] = value

        return fund_data

    except requests.RequestException as e:
        print(f"   错误: 基金 {fund_code} 抓取请求失败: {e}. 跳过.")
        raise e
    except Exception as e:
        print(f"   错误: 基金 {fund_code} 抓取过程中发生未知错误: {e}. 跳过.")
        return None

def main():
    fund_codes = get_fund_codes()
    if not fund_codes:
        print("未找到任何基金代码，脚本退出。")
        return

    print(f"共找到 {len(fund_codes)} 个基金代码，开始并发抓取...")
    
    MAX_WORKERS = 20 
    all_fund_data = []

    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_code = {executor.submit(scrape_fund_info, code): code for code in fund_codes}
        
        for future in concurrent.futures.as_completed(future_to_code):
            code = future_to_code[future]
            try:
                data = future.result()
                if data:
                    all_fund_data.append(data)
            except requests.RequestException:
                print(f"   严重错误: 基金 {code} 抓取请求失败，可能是网络问题或被反爬。")
            except Exception as exc:
                print(f"   致命错误: 基金 {code} 抓取过程中发生未预料的异常: {exc}")


    end_time = time.time()
    total_time = end_time - start_time
    
    if not all_fund_data:
        print("未成功抓取任何基金数据，未生成CSV文件。")
        return

    df = pd.DataFrame(all_fund_data)
    
    # 🌟 关键修改点：按基金代码排序，确保文件内容稳定
    df = df.sort_values(by='基金代码', ascending=True) 
    
    cols = ['基金代码'] + [col for col in df.columns if col != '基金代码']
    df = df[cols]

    output_filename = 'fund_basic_data_c_class.csv'

    df.to_csv(output_filename, index=False, encoding='utf_8_sig')
    
    print(f"\n✅ 数据抓取完成，已保存到文件: {output_filename}")
    print(f"   共抓取 {len(all_fund_data)} 条数据。")
    print(f"   总耗时: {total_time:.2f} 秒 (约 {total_time/60:.2f} 分钟)")

if __name__ == "__main__":
    main()
