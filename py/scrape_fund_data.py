import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import glob
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 配置 ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
BASE_URL = "http://fundf10.eastmoney.com/jjjl_{code}.html"
FUND_CODES_FILE = "C类.txt"
SHA_TZ = timezone(timedelta(hours=8), name='Asia/Shanghai')
MAX_WORKERS = 10 
# 历史数据文件模式：递归查找仓库中所有总表文件
HISTORY_FILE_PATTERN = "**/*总表_*.csv" 

def get_shanghai_time():
    """获取当前的上海时区时间"""
    return datetime.now(SHA_TZ)

def load_latest_history():
    """读取最新的总表文件作为历史数据"""
    try:
        # 递归查找所有匹配的文件
        all_files = glob.glob(HISTORY_FILE_PATTERN, recursive=True)
        # 排除目录，只保留文件
        all_files = [f for f in all_files if os.path.isfile(f)]
        
        if not all_files:
            print("未找到历史数据文件，将执行完整抓取。")
            return pd.DataFrame(), {}
        
        # 按修改时间或创建时间排序，取最新的文件
        latest_file = max(all_files, key=os.path.getmtime)
        print(f"-> 发现最新历史文件: {latest_file}")
        
        # 读取历史数据，使用 'utf-8-sig' 处理中文
        history_df = pd.read_csv(latest_file, encoding='utf-8-sig', errors='ignore')
        
        # 标准化基金代码为6位字符串，补领先零
        if '基金代码' in history_df.columns:
            history_df['基金代码'] = history_df['基金代码'].astype(str).str.strip().str.zfill(6)
        
        # 检查关键列是否存在，如果不存在则返回空
        if history_df.empty or '基金代码' not in history_df.columns:
             print("历史文件为空或缺少 '基金代码' 列，将执行完整抓取。")
             return pd.DataFrame(), {}
        
        # 为快速查找创建历史基本信息字典
        history_basic_info = {}
        for code, group in history_df.groupby('基金代码'):
            code_str = str(code).strip()
            if not code_str: continue

            # 取该基金在历史表中任意一条记录的基本信息作为对比基准
            first_record = group.iloc[0]
            history_basic_info[code_str] = {
                '成立日期': str(first_record.get('成立日期', '')).strip(),
                '基金经理(现任)': str(first_record.get('基金经理(现任)', '')).strip(),
                '资产规模': str(first_record.get('资产规模', '')).strip(),
                '截止至': str(first_record.get('截止至', '')).strip()
            }
        
        return history_df, history_basic_info
        
    except Exception as e:
        print(f"读取历史数据时发生错误: {e}，将执行完整抓取。")
        return pd.DataFrame(), {}

def compare_basic_info(code, new_info, history_basic_info):
    """对比基本信息，判断是否需要更新"""
    code_str = str(code).strip()
    if code_str not in history_basic_info:
        print(f"   [新基金] {code_str}：无历史记录，需完整抓取。")
        return True # 新基金，必须抓取
        
    # 定义需要对比的关键字段
    keys_to_compare = ['成立日期', '基金经理(现任)', '资产规模', '截止至']
    historical = history_basic_info[code_str]
    
    for key in keys_to_compare:
        new_value = str(new_info.get(key, '')).strip()
        old_value = str(historical.get(key, '')).strip()
        
        # 只要有一个关键字段不匹配，就认为有更新
        if new_value != old_value:
            print(f"   [更新] {code_str}：'{key}' 已更新 ('{old_value}' -> '{new_value}')，需完整抓取。")
            return True
            
    print(f"   [跳过] {code_str}：关键信息无变化，使用历史数据。")
    return False

def extract_basic_info(soup, fund_code):
    """从页面提取基金基本信息"""
    basic_info = {'基金代码': fund_code, '截止至': '', '基金经理(现任)': ''}
    bs_gl_div = soup.find('div', class_='bs_gl')
    if bs_gl_div:
        labels = bs_gl_div.find_all('label')
        for label in labels:
            text = label.get_text(strip=True)
            if '：' in text:
                match = re.search(r'(.+?)：\s*(.+)', text)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    
                    if key == '资产规模':
                        scale_match = re.search(r'(.+?)（截止至：(.+?)）', value)
                        if scale_match:
                            basic_info['资产规模'] = scale_match.group(1).strip()
                            basic_info['截止至'] = scale_match.group(2).strip()
                        else:
                            basic_info['资产规模'] = value
                    elif key == '基金经理':
                         basic_info['基金经理(现任)'] = value
                    elif key:
                         basic_info[key] = value
                         
    # 格式化用于对比的基础数据
    base_data_extracted = {
        '基金代码': fund_code, # <<< FIX: 确保基金代码被包含在返回的字典中
        '成立日期': basic_info.get('成立日期', ''),
        '基金经理(现任)': basic_info.get('基金经理(现任)', ''),
        '类型': basic_info.get('类型', ''),
        '管理人': basic_info.get('管理人', ''),
        '资产规模': basic_info.get('资产规模', ''),
        '截止至': basic_info.get('截止至', '')
    }
    return base_data_extracted

def extract_manager_changes(soup, base_data):
    """从页面提取基金经理变动一览"""
    manager_changes_list = []
    manager_table_container = soup.find('div', class_='boxitem w790')
    if manager_table_container:
        manager_table = manager_table_container.find('table', class_='comm jloff')
        if manager_table:
            # 直接查找所有tr，跳过表头（假设第一行是表头）
            rows = manager_table.find_all('tr')
            if rows:
                for row in rows[1:]:  # 跳过第一行表头
                    cols = row.find_all(['td', 'th'])
                    if len(cols) >= 5:
                        change_entry = {
                            '起始期': cols[0].get_text(strip=True),
                            '截止期': cols[1].get_text(strip=True),
                            '基金经理(变动)': cols[2].get_text(strip=True),
                            '任职期间': cols[3].get_text(strip=True),
                            '任职回报': cols[4].get_text(strip=True)
                        }
                        # 合并基本信息到每一条变动记录中
                        record = {} # 创建一个空字典
                        record.update(base_data) # 包含所有基本信息（包括基金代码）
                        record.update(change_entry)
                        manager_changes_list.append(record)

    # 如果没有变动记录，仍返回一条包含基本信息的记录
    if not manager_changes_list:
        record = {}
        record.update(base_data)
        manager_changes_list.append(record)

    return manager_changes_list

def scrape_fund_data(fund_code, history_df, history_basic_info):
    """抓取单个基金的详细数据，并根据历史数据决定是否跳过"""
    fund_code_str = str(fund_code).zfill(6)
    url = BASE_URL.format(code=fund_code_str)
    print(f"\n-> 准备处理基金代码: {fund_code_str}")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.encoding = 'utf-8' 
        if response.status_code != 200:
            print(f"   ❌ 请求失败，状态码: {response.status_code} - {fund_code_str}")
            return fund_code_str, None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. 抓取基本资料 (必须抓取用于对比)
        base_data_new = extract_basic_info(soup, fund_code_str)

        # 2. 对比历史数据，决定是否抓取变动一览
        needs_full_update = compare_basic_info(fund_code_str, base_data_new, history_basic_info)

        if not needs_full_update and fund_code_str in history_basic_info:
            # 关键信息无变化，直接从历史数据中提取记录
            historical_records = history_df[history_df['基金代码'].astype(str) == fund_code_str].to_dict('records')
            
            # 使用最新的基本信息覆盖历史记录中的基本信息
            updated_historical_records = []
            for record in historical_records:
                # 提取历史记录中的变动信息
                change_info = {
                    '起始期': record.get('起始期', ''), 
                    '截止期': record.get('截止期', ''),
                    '基金经理(变动)': record.get('基金经理(变动)', ''), 
                    '任职期间': record.get('任职期间', ''),
                    '任职回报': record.get('任职回报', '')
                }
                
                # 创建新的记录字典，并合并最新的基本信息和历史变动信息
                new_record = {}
                new_record.update(base_data_new)
                new_record.update(change_info)
                updated_historical_records.append(new_record)

            return fund_code_str, updated_historical_records
        
        # 3. 执行完整抓取 (包括基金经理变动一览)
        fund_records = extract_manager_changes(soup, base_data_new)
        
        print(f"   ✅ 成功完成 {fund_code_str} 的完整抓取。")
        return fund_code_str, fund_records
        
    except Exception as e:
        print(f"   ❌ 抓取基金代码 {fund_code_str} 时发生致命错误: {e}")
        return fund_code_str, None

def main():
    """主函数，读取基金代码并生成总表"""
    if not os.path.exists(FUND_CODES_FILE):
        print(f"错误: 找不到基金代码文件 {FUND_CODES_FILE}。请创建此文件并填入基金代码。")
        return

    # 1. 读取所有需要追踪的代码 (C类.txt)
    with open(FUND_CODES_FILE, 'r', encoding='utf-8') as f:
        # 过滤空行、注释行、非数字行，并转换为字符串
        new_fund_codes = [line.strip() for line in f if line.strip() and not line.startswith('#') and line.strip().isdigit()]
    
    # 标准化为6位代码
    new_fund_codes = [str(code).zfill(6) for code in new_fund_codes]
    
    # 2. 读取历史数据
    history_df, history_basic_info = load_latest_history()
    
    # 3. 合并和去重基金代码列表：C类.txt + 历史文件中的代码
    history_codes = []
    # 只有在历史DataFrame非空且包含'基金代码'列时才提取历史代码
    if not history_df.empty and '基金代码' in history_df.columns:
        history_codes = history_df['基金代码'].astype(str).str.strip().unique().tolist()
    
    all_codes_to_track = sorted(list(set(new_fund_codes) | set(history_codes)))
    
    if not all_codes_to_track:
        print(f"没有基金代码需要追踪，程序退出。")
        return

    print(f"\n待追踪基金总数: {len(all_codes_to_track)}")
    
    all_fund_data = []
    
    # 4. 使用 ThreadPoolExecutor 实现并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有抓取任务
        future_to_code = {
            executor.submit(scrape_fund_data, code, history_df, history_basic_info): code 
            for code in all_codes_to_track
        }
        
        # 处理完成的任务结果
        for future in as_completed(future_to_code):
            code = future_to_code[future]
            try:
                _, data = future.result() 
                if data:
                    all_fund_data.extend(data)
            except Exception as exc:
                print(f"基金 {code} 的处理生成了一个异常: {exc}")

    if not all_fund_data:
        print("\n未抓取到任何基金数据，无法生成总表。")
        return

    # 5. 生成CSV总表
    df = pd.DataFrame(all_fund_data)
    
    final_columns = [
        '基金代码', '成立日期', '基金经理(现任)', '类型', '管理人', '资产规模', '截止至',
        '起始期', '截止期', '基金经理(变动)', '任职期间', '任职回报'
    ]
    
    # 确保dataframe只有需要的列，缺失值用空字符串填充
    df = df.reindex(columns=final_columns, fill_value='')

    # 6. 保存到指定目录
    now_sha = get_shanghai_time()
    timestamp = now_sha.strftime("%Y%m%d_%H%M%S")
    output_dir = now_sha.strftime("%Y/%m")
    output_filename = f"总表_{timestamp}.csv"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    # encoding='utf-8-sig' 确保 Excel 中文不乱码
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n==========================================")
    print(f"✅ 数据抓取完成，总表已保存到: {output_path}")
    print(f"待追踪基金总数: {len(all_codes_to_track)}, 成功记录总数: {len(all_fund_data)}")
    print(f"==========================================")

if __name__ == "__main__":
    main()
