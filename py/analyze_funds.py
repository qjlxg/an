import pandas as pd
import numpy as np
import os
import re
import random
import time
import requests
import concurrent.futures
from bs4 import BeautifulSoup
from datetime import datetime
import warnings

# 忽略 SettingWithCopyWarning
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# --- 配置参数（支持环境变量，CI 友好）---
DATA_DIR = os.getenv('FUND_DATA_DIR', 'fund_data')
OUTPUT_FILE = os.getenv('OUTPUT_FILE', 'fund_analysis_summary.csv')
RISK_FREE_RATE = 0.02
TRADING_DAYS_PER_YEAR = 250
# 建议 MAX_THREADS 略微降低以缓解反爬压力，保留用户原始配置
MAX_THREADS = int(os.getenv('MAX_THREADS', '5')) 
FUND_INFO_CACHE = {}
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.3 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0'
]
MAX_RETRIES = 3 # 增加重试次数配置
REQUEST_TIMEOUT = 20

# --- 指标计算（不包含滚动收益）---
def calculate_metrics(df, start_date, end_date):
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    if len(df) < 2:
        return None

    df = df.sort_values(by='date').reset_index(drop=True)
    nav = pd.to_numeric(df['cumulative_net_value'], errors='coerce')

    # 修复异常小净值（原逻辑不变）
    if nav.min() < 0.1 and nav.max() < 10:
        nav *= 1000

    nav = nav.replace(0, np.nan).dropna()
    if len(nav) < 2:
        return None

    df = df.loc[nav.index].copy()
    nav = nav.reset_index(drop=True)

    # 1. 年化收益率（交易日）
    trading_days = len(nav)
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    day_diff = trading_days - 1
    annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / day_diff) - 1 if day_diff > 0 else np.nan

    # 2. 年化波动率
    daily_returns = nav.pct_change().dropna()
    annual_vol = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(daily_returns) > 0 else np.nan

    # 3. 最大回撤
    peak = nav.expanding().max()
    drawdown = nav / peak - 1
    mdd = drawdown.min()

    # 4. 夏普比率
    sharpe = (annual_return - RISK_FREE_RATE) / annual_vol if annual_vol is not np.nan and annual_vol > 1e-8 else np.nan

    return {
        '共同期年化收益率': annual_return,
        '共同期年化标准差': annual_vol,
        '共同期最大回撤(MDD)': mdd,
        '共同期夏普比率': sharpe,
    }


# --- 网络请求函数（增强健壮性，增加重试，并清理资产规模和费率）---
def fetch_fund_info(fund_code, max_retries=MAX_RETRIES):
    if fund_code in FUND_INFO_CACHE:
        return FUND_INFO_CACHE[fund_code]

    defaults = {
        'code': fund_code,
        'name': f'名称查找失败({fund_code})',
        'size': 'N/A',
        'type': 'N/A',
        'daily_growth': 'N/A',
        'net_value': 'N/A',
        'rate': 'N/A', # 管理费率
        'sales_rate': 'N/A', # 销售服务费率
        'custody_rate': 'N/A' # 托管费率
    }

    url = f'http://fundf10.eastmoney.com/jbgk_{fund_code}.html'
    
    for attempt in range(max_retries):
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        sleep_time = random.uniform(2.0, 4.0) 
        time.sleep(sleep_time)
        
        try:
            print(f"[{fund_code}] Fetching... (Attempt {attempt + 1}/{max_retries}, Sleep {sleep_time:.2f}s)")
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # 1. 基金简称
            title_tag = soup.select_one('.basic-new .bs_jz h4.title a')
            if title_tag and 'title' in title_tag.attrs:
                defaults['name'] = re.sub(r'\(.*?\)$', '', title_tag['title']).strip()

            # 2. 最新净值 + 日涨跌幅
            net_tag = soup.select_one('.basic-new .bs_jz .col-right .row1 b')
            if net_tag:
                text = net_tag.text.strip()
                parts = re.split(r'\s*\((.*?)\)\s*', text, 1)
                if len(parts) >= 3:
                    defaults['net_value'] = parts[0].strip()
                    defaults['daily_growth'] = f'({parts[1]})'
                else:
                    defaults['net_value'] = parts[0].strip()

            # --- 关键信息提取 ---
            info_table = soup.select_one('table.info.w790')
            if info_table:
                # 基金类型
                type_th = info_table.find('th', string=re.compile(r'基金类型'))
                if type_th:
                    type_td = type_th.find_next_sibling('td')
                    if type_td:
                        defaults['type'] = type_td.text.strip()

                # 资产规模（清理）
                size_th = info_table.find('th', string=re.compile(r'资产规模'))
                if size_th:
                    size_td = size_th.find_next_sibling('td')
                    if size_td:
                        raw_size = size_td.text.strip().replace('\n', ' ').replace('\t', '')
                        match = re.search(r'([\d.]+)\s*([亿万]元)', raw_size)
                        if match:
                            defaults['size'] = match.group(0)
                        else:
                             defaults['size'] = raw_size.split('（')[0].strip() 

                # 基金费率（管理费率、托管费率、销售服务费率）
                
                # 管理费率 (rate)
                rate_th = info_table.find('th', string=re.compile(r'管理费率'))
                if rate_th:
                    rate_td = rate_th.find_next_sibling('td')
                    if rate_td:
                        defaults['rate'] = rate_td.text.strip()
                
                # 托管费率 (custody_rate)
                custody_th = info_table.find('th', string=re.compile(r'托管费率'))
                if custody_th:
                    custody_td = custody_th.find_next_sibling('td')
                    if custody_td:
                        defaults['custody_rate'] = custody_td.text.strip()
                        
                # 销售服务费率 (sales_rate)
                sales_th = info_table.find('th', string=re.compile(r'销售服务费率'))
                if sales_th:
                    sales_td = sales_th.find_next_sibling('td')
                    if sales_td:
                        defaults['sales_rate'] = sales_td.text.strip()

            # 成功获取并返回
            FUND_INFO_CACHE[fund_code] = defaults
            return defaults

        except requests.exceptions.RequestException as e:
            print(f"[{fund_code}] Request failed: {e}. Retrying...")
            if attempt == max_retries - 1:
                 print(f"[{fund_code}] Max retries reached. Using default N/A values.")
        except Exception as e:
            print(f"[{fund_code}] Parsing failed: {e}. Retrying...")
            if attempt == max_retries - 1:
                 print(f"[{fund_code}] Max retries reached. Using default N/A values.")

    FUND_INFO_CACHE[fund_code] = defaults
    return defaults


# --- 主函数 ---
def main():
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found.")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not files:
        print(f"Error: No CSV files in '{DATA_DIR}'.")
        return

    # 阶段 1: 确定共同期
    print("--- Phase 1/3: Determining Common Period ---")
    earliest_start = pd.to_datetime('1900-01-01')
    latest_end = pd.to_datetime('2200-01-01')

    all_data_frames = {} 
    
    for f in files:
        path = os.path.join(DATA_DIR, f)
        code = f.replace('.csv', '')
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='gbk')

        df.columns = df.columns.str.strip().str.lower()
        date_col = next((c for c in df.columns if '日期' in c or 'date' in c), None)
        nav_col = next((c for c in df.columns if '累计净值' in c or 'cumulative_net_value' in c), None)

        if date_col and nav_col:
            df = df.rename(columns={nav_col: 'cumulative_net_value', date_col: 'date'})
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date', 'cumulative_net_value'])
            
            if not df.empty:
                earliest_start = max(earliest_start, df['date'].min())
                latest_end = min(latest_end, df['date'].max())
                all_data_frames[code] = df
        else:
            print(f"Warning: Missing date or NAV column in {f}. Skipping.")


    if latest_end <= earliest_start:
        print("Error: No valid common period.")
        return

    print(f"Common Period: {earliest_start.strftime('%Y-%m-%d')} to {latest_end.strftime('%Y-%m-%d')}")

    # 阶段 2: 计算指标
    print("\n--- Phase 2/3: Calculating Metrics ---")
    results = []
    codes_to_fetch = []

    for code, df in all_data_frames.items():
        metrics = calculate_metrics(df, earliest_start, latest_end)
        if metrics:
            results.append({
                '基金代码': code,
                '起始日期': earliest_start.strftime('%Y-%m-%d'),
                '结束日期': latest_end.strftime('%Y-%m-%d'),
                **metrics
            })
            codes_to_fetch.append(code)
        else:
            print(f"Warning: Could not calculate valid metrics for {code} in common period. Skipping.")

    if not results:
        print("Error: No valid metrics calculated.")
        return

    summary_df = pd.DataFrame(results)

    # 阶段 3: 爬取基本信息 
    print(f"\n--- Phase 3/3: Fetching Info for {len(codes_to_fetch)} Funds using {MAX_THREADS} threads ---")
    
    unique_codes = list(set(codes_to_fetch)) 
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        list(executor.map(fetch_fund_info, unique_codes))


    # 整合信息
    info_list = [FUND_INFO_CACHE.get(code, {}) for code in summary_df['基金代码']]
    info_df = pd.DataFrame(info_list).rename(columns={
        'code': '基金代码_info', 
        'name': '基金简称', 
        'size': '资产规模',
        'type': '基金类型', 
        'daily_growth': '最新日涨跌幅',
        'net_value': '最新净值', 
        'rate': '管理费率', # 保持原有管理费率
        'sales_rate': '销售服务费率', # 新增
        'custody_rate': '托管费率' # 新增
    })
    
    info_df['基金代码'] = info_df['基金代码_info'].fillna(summary_df['基金代码'])
    info_df = info_df.drop(columns=['基金代码_info'])

    summary_df.index = info_df.index
    # 最终结果 DataFrame
    final_df = pd.concat([info_df, summary_df.drop(columns=['基金代码'])], axis=1)

    # 格式化
    final_df['共同期夏普比率_Num'] = pd.to_numeric(final_df['共同期夏普比率'], errors='coerce')
    for col in final_df.columns:
        if '收益率' in col or '标准差' in col or '回撤' in col:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').apply(
                lambda x: f"{x:.2%}" if pd.notna(x) and not np.isinf(x) else 'N/A')
        elif '夏普比率' in col and '_Num' not in col:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').apply(
                lambda x: f"{x:.3f}" if pd.notna(x) and not np.isinf(x) else 'N/A')

    # 排序
    final_output = final_df.sort_values(by='共同期夏普比率_Num', ascending=False)
    final_output = final_output.drop(columns=['共同期夏普比率_Num']).reset_index(drop=True)

    # 列顺序 (在 '管理费率' 后插入 '销售服务费率' 和 '托管费率')
    target_cols = [
        '基金代码', '基金简称', '资产规模', '基金类型', '最新日涨跌幅', '最新净值', 
        '管理费率', '销售服务费率', '托管费率', # 新增费率字段
        '起始日期', '结束日期', '共同期年化收益率', '共同期年化标准差',
        '共同期最大回撤(MDD)', '共同期夏普比率'
    ]
    # 确保只保留存在的列
    final_output = final_output[[c for c in target_cols if c in final_output.columns]]

    # 输出
    final_output.to_csv(OUTPUT_FILE, index=False, encoding='utf_8_sig')
    print(f"\n--- Success ---\nResults saved to: {os.path.abspath(OUTPUT_FILE)}")
    print("\nTop Funds by Sharpe Ratio:")
    # 打印前 10 行基金数据
    print(final_output.head(10).to_string(index=False))


if __name__ == '__main__':
    main()