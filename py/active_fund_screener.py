# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置区（直接编码阈值）===================
THRESHOLDS = {
    'min_tenure': 5,                # 经理任职年限 ≥ 5年
    'min_annual_return': 0.15,      # 近5年年化 ≥ 15%
    'max_drawdown': -0.30,          # 最大回撤 ≤ -30%
    'min_sharpe': 1.0,              # 夏普比率 ≥ 1.0
    'min_calmar': 0.5,              # 卡玛比率 ≥ 0.5
    'min_return_drawdown_ratio': 0.6,  # 收益回撤比 ≥ 0.6
    'min_size': 5,                  # 规模 ≥ 5亿
    'max_size': 50,                 # 规模 ≤ 50亿
    'top_rank_percent': 0.2         # 同类前20%
}

SETTINGS = {
    'max_funds_to_scan': 200,       # 最多扫描基金数量（防超时）
    'enable_cache': True            # 启用缓存（推荐）
}

TH = THRESHOLDS
CACHE_DIR = 'cache'  # 缓存目录（不会 commit）
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================== 工具函数：缓存读写 ====================
def cache_get(key, default=None):
    if not SETTINGS['enable_cache']:
        return default
    path = f"{CACHE_DIR}/{key}.pkl"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return default

def cache_set(key, value):
    if SETTINGS['enable_cache']:
        path = f"{CACHE_DIR}/{key}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(value, value)

# ==================== 数据获取：经理年限 ====================
def get_manager_tenure(fund_code):
    key = f"tenure_{fund_code}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        df = ak.fund_manager_info_em(symbol=fund_code)
        if df.empty:
            tenure = 0
        else:
            start_date = df.iloc[0]['任职日期']
            tenure = (datetime.now() - pd.to_datetime(start_date)).days / 365.25
            tenure = round(tenure, 2)
        cache_set(key, tenure)
        time.sleep(0.3)
        return tenure
    except:
        cache_set(key, 0)
        return 0

# ==================== 数据获取：同类排名 ====================
def get_peer_rank_percent(fund_code):
    key = f"rank_{fund_code}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        df = ak.fund_em_open_fund_info(fund=fund_code)
        if df.empty or '同类排名' not in df.columns:
            percent = 0.5
        else:
            rank_str = str(df['同类排名'].iloc[0])
            if '/' in rank_str:
                rank, total = map(int, rank_str.split('/'))
                percent = rank / total
            else:
                percent = 0.5
        cache_set(key, percent)
        time.sleep(0.3)
        return percent
    except:
        cache_set(key, 0.5)
        return 0.5

# ==================== 高级指标计算 ====================
def calc_calmar(annual_return, max_dd):
    return annual_return / abs(max_dd) if max_dd < 0 else np.nan

def calc_return_drawdown_ratio(annual_return, max_dd):
    return annual_return / abs(max_dd) if max_dd < 0 else np.nan

# ==================== 主筛选函数（已添加强制列检查） ====================
def screen_funds(fund_codes, max_funds=SETTINGS['max_funds_to_scan']):
    print(f"正在处理 {min(len(fund_codes), max_funds)} 只基金...")
    results = []
    
    # 定义所有期望的列及其安全的默认值（用于缺失时填充）
    required_cols = {
        '基金代码': '', '基金名称': '',
        '基金类型': '', '经理任职年限': 0.0,
        '近5年年化回报': 0.0, '最大回撤': 0.0,
        '夏普比率': 0.0, '年换手率': 0.0,
        '基金规模': 0.0, '同类排名百分位': 1.0, 
        '卡玛比率': 0.0, '收益回撤比': 0.0
    }

    for i, code in enumerate(fund_codes[:max_funds]):
        if i % 20 == 0 and i > 0:
            print(f"  已处理 {i} 只...")
        try:
            info = ak.fund_em_open_fund_info(fund=code)
            if info.empty: 
                continue
            
            row = info.iloc[0].to_dict()
            name = row.get('基金简称', '未知')
            
            # 使用 .get() 确保提取时不会发生 KeyError，并设置默认值
            fund_type = row.get('基金类型', '')
            annual_return = pd.to_numeric(row.get('近5年年化收益率', 0), errors='coerce') or 0
            max_dd = pd.to_numeric(row.get('最大回撤', 0), errors='coerce') or 0
            sharpe = pd.to_numeric(row.get('夏普比率', 0), errors='coerce') or 0
            turnover = pd.to_numeric(row.get('换手率', 0), errors='coerce') or 0
            size = pd.to_numeric(row.get('基金规模(亿元)', 0), errors='coerce') or 0
            
            # 获取并计算高级指标
            tenure = get_manager_tenure(code)
            rank_percent = get_peer_rank_percent(code)
            calmar = calc_calmar(annual_return, max_dd)
            rdr = calc_return_drawdown_ratio(annual_return, max_dd)

            results.append({
                '基金代码': code,
                '基金名称': name,
                '基金类型': fund_type,
                '经理任职年限': tenure,
                '近5年年化回报': annual_return,
                '最大回撤': max_dd,
                '夏普比率': sharpe,
                '年换手率': turnover,
                '基金规模': size,
                '同类排名百分位': rank_percent,
                '卡玛比率': calmar,
                '收益回撤比': rdr
            })
        except Exception as e:
             # 捕获其他运行时错误并跳过当前基金
             continue
             
    df = pd.DataFrame(results)
    
    # ⬇️ 最终健壮性修复：强制确保所有用于筛选的列都存在且类型正确
    for col, default_val in required_cols.items():
        if col not in df.columns:
            # 如果列不存在，则添加该列并填充默认值
            df[col] = default_val
            
    # 强制将数值列转换为float类型，以防出现字符串导致比较失败
    for col in ['经理任职年限', '近5年年化回报', '最大回撤', '夏普比率', '卡玛比率', 
                '收益回撤比', '年换手率', '基金规模', '同类排名百分位']:
        if col in df.columns:
            # 使用 errors='coerce' 将不能转换为数字的值设为 NaN，然后用默认值填充
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(required_cols[col])


    # 最终筛选逻辑 (现在是安全的)
    mask = (
        df['基金类型'].str.contains('偏股|灵活配置', na=False) &
        (df['经理任职年限'] >= TH['min_tenure']) &
        (df['近5年年化回报'] >= TH['min_annual_return']) &
        (df['最大回撤'] >= TH['max_drawdown']) &
        (df['夏普比率'] >= TH['min_sharpe']) &
        (df['卡玛比率'] >= TH['min_calmar']) &
        (df['收益回撤比'] >= TH['min_return_drawdown_ratio']) &
        (df['年换手率'].between(1.0, 3.0)) &
        (df['基金规模'].between(TH['min_size'], TH['max_size'])) &
        (df['同类排名百分位'] <= TH['top_rank_percent'])
    )
    result = df[mask].sort_values('近5年年化回报', ascending=False).reset_index(drop=True)
    return result

# ==================== 输出函数：年月目录 + 时间戳 ====================
def export_and_plot(df):
    if df.empty:
        print("未找到符合条件的基金，建议放宽阈值")
        return
    now = datetime.now()
    year = now.strftime('%Y')
    month = now.strftime('%m')
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    output_dir = f"output/{year}/{month}"
    os.makedirs(output_dir, exist_ok=True)
    # Excel
    excel_file = f"{output_dir}/优秀基金筛选结果_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='入选基金', index=False)
        summary = pd.DataFrame({
            '指标': ['入选数量', '平均年化', '平均回撤', '平均夏普', '平均卡玛'],
            '数值': [
                len(df),
                f"{df['近5年年化回报'].mean():.1%}",
                f"{df['最大回撤'].mean():.1%}",
                f"{df['夏普比率'].mean():.2f}",
                f"{df['卡玛比率'].mean():.2f}"
            ]
        })
        summary.to_excel(writer, sheet_name='总结', index=False)
    print(f"已导出 Excel：{excel_file}")
    # 图表
    top10 = df.head(10)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(top10['基金名称'], top10['近5年年化回报']*100, color='#4CAF50')
    plt.xlabel('近5年年化回报率 (%)')
    plt.title('Top 10 长跑健将基金')
    plt.gca().invert_yaxis()
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', fontsize=9)
    plt.tight_layout()
    plot_file = f"{output_dir}/top10_{timestamp}.png"
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
    plt.close()  # 关闭图表，避免内存泄漏
    print(f"图表已保存：{plot_file}")
    # 日志（全局追加）
    log_file = "output/筛选日志.txt"
    os.makedirs('output', exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"筛选时间：{now.strftime('%Y-%m-%d %H:%M:%S')} (Asia/Shanghai)\n")
        f.write(f"入选基金：{len(df)} 只\n")
        f.write(f"Top 3：{', '.join(df['基金名称'].head(3).tolist())}\n")
        f.write(f"平均年化：{df['近5年年化回报'].mean():.1%} | 平均回撤：{df['最大回撤'].mean():.1%}\n")
        f.write(f"输出目录：{output_dir}\n")
    print(f"日志已记录：{log_file}")

# ==================== 主程序（保持不变） ====================
def main():
    print("启动 主动型基金筛选系统 v4.0（GitHub Actions 版）")
    print("正在读取 C类.txt 中的基金代码...")
    try:
        with open('C类.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 跳过第一行（假设是 'code' 标题），读取后续非空行作为基金代码
            codes = [line.strip() for line in lines[1:] if line.strip()] 
    except FileNotFoundError:
        print("错误: 未找到 C类.txt 文件！请确保文件已提交到仓库根目录。")
        return
    
    if not codes:
        print("警告: C类.txt 中未找到有效的基金代码。筛选中止。")
        return

    print(f"共读取 {len(codes)} 只基金代码，开始筛选...")
    result_df = screen_funds(codes)
    export_and_plot(result_df)
    print("筛选完成！结果已保存至 output/ 年月目录")

if __name__ == "__main__":
    main()
