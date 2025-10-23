import pandas as pd
import requests
from datetime import datetime
import os
import time

def fetch_fund_holdings(fund_code):
    """
    从东方财富网获取基金持仓信息
    """
    url = f"http://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=jjcc&code={fund_code}&topline=10&year=2024"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 简单解析HTML表格
        tables = pd.read_html(response.text, encoding='utf-8')
        if tables:
            holdings_table = tables[0]
            print(f"✅ 成功获取基金 {fund_code} 的持仓数据。")
            return holdings_table
        else:
            print(f"⚠️ 无法从 {fund_code} 获取表格数据。")
            return None
    except Exception as e:
        print(f"❌ 获取基金 {fund_code} 数据时出错：{e}")
        return None

def main():
    """
    主函数：读取基金代码并抓取数据
    """
    # 确定今天的日期和文件路径
    today_date = datetime.now().strftime('%Y%m%d')
    input_csv_path = f'data/买入信号基金_{today_date}.csv'
    
    print(f"🚀 正在检查输入文件路径: {input_csv_path}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_csv_path):
        print(f"❌ 输入文件 {input_csv_path} 不存在。请确认上游流程已成功生成文件并将其推送到仓库。")
        return
    else:
        print(f"✅ 找到输入文件：{input_csv_path}")
        
    # 创建输出目录
    output_dir = 'fund_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}")
        
    # 读取基金代码
    try:
        df = pd.read_csv(input_csv_path)
        fund_codes = df['fund_code'].unique()
        print(f"✅ 成功读取到 {len(fund_codes)} 个基金代码。")
    except Exception as e:
        print(f"❌ 读取 CSV 文件时出错：{e}")
        return

    # 遍历基金代码并抓取数据
    for code in fund_codes:
        print(f"----------------------------------------")
        print(f"🔍 正在处理基金代码: {code}")
        holdings_df = fetch_fund_holdings(str(code).zfill(6))
        
        if holdings_df is not None:
            output_path = os.path.join(output_dir, f'持仓_{code}_{today_date}.csv')
            holdings_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ 持仓数据已保存至 {output_path}")
            
        # 增加延迟以避免对网站造成压力
        time.sleep(2)
        
    print(f"----------------------------------------")
    print(f"✅ 所有基金处理完毕。")

if __name__ == "__main__":
    main()
