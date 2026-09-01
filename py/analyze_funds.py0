import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# --- 配置 ---
DATA_DIR = 'fund_data'
OUTPUT_FILE = 'fund_analysis_summary.csv'
RISK_FREE_RATE = 0.02 # 假设无风险利率为 2.0%
TRADING_DAYS_PER_YEAR = 250
# --- 配置结束 ---

# 定义滚动分析周期（以交易日近似）
# 1周: 5天, 1月: 20天, 1季度: 60天, 半年: 125天, 1年: 250天
ROLLING_PERIODS = {
    '1周': 5,
    '1月': 20,
    '1季度': 60,
    '半年': 125,
    '1年': 250
}

def calculate_rolling_returns(cumulative_net_value, period_days):
    """计算指定周期（交易日）的平均滚动年化收益率"""
    
    # 计算周期回报率 (Rolling Return)
    rolling_returns = (cumulative_net_value.pct_change(periods=period_days) + 1).pow(TRADING_DAYS_PER_YEAR / period_days) - 1
    
    # 返回所有滚动回报率的平均值（平均滚动年化收益率）
    return rolling_returns.mean()

def calculate_metrics(df, start_date, end_date):
    """计算基金的关键指标：年化收益、年化标准差、最大回撤、夏普比率和滚动收益"""
    
    # 1. 筛选共同分析期数据
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].sort_values(by='date')
    
    if df.empty or len(df) < 2:
        return None

    # 确保累计净值是数值类型，并去除0值和缺失值
    cumulative_net_value = pd.to_numeric(df['cumulative_net_value'], errors='coerce').replace(0, np.nan).dropna()
    
    if len(cumulative_net_value) < 2:
          return None

    # 2. 计算日收益率
    returns = cumulative_net_value.pct_change().dropna()
    
    # 3. 长期指标
    total_days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    total_return = (cumulative_net_value.iloc[-1] / cumulative_net_value.iloc[0]) - 1
    annual_return = (1 + total_return) ** (365 / total_days) - 1
    
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # 避免除以零
    sharpe_ratio = (annual_return - RISK_FREE_RATE) / annual_volatility if annual_volatility != 0 else np.nan
    
    peak = cumulative_net_value.expanding(min_periods=1).max()
    drawdown = (cumulative_net_value / peak) - 1
    max_drawdown = drawdown.min()

    # 4. 短期滚动收益指标
    rolling_results = {}
    for name, days in ROLLING_PERIODS.items():
        # 确保数据长度足够进行滚动计算
        if len(cumulative_net_value) >= days:
            rolling_return = calculate_rolling_returns(cumulative_net_value, days)
            rolling_results[f'平均滚动年化收益率({name})'] = rolling_return
        else:
             rolling_results[f'平均滚动年化收益率({name})'] = np.nan
    
    # 5. 整合结果
    metrics = {
        '年化收益率(总)': annual_return,
        '年化标准差(总)': annual_volatility,
        '最大回撤(MDD)': max_drawdown,
        '夏普比率(总)': sharpe_ratio,
        **rolling_results
    }
    
    return metrics

def main():
    earliest_start_date = pd.to_datetime('1900-01-01')
    latest_end_date = pd.to_datetime('2200-01-01')
    
    file_list = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    

    
    # 简化：仅为展示和测试，此处保留核心逻辑
    if not file_list:
        print(f"错误：{DATA_DIR} 目录中没有找到 CSV 文件。")
        return
        
    for filename in file_list:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.lower()
            if 'date' in df.columns and 'cumulative_net_value' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                valid_dates = df.dropna(subset=['cumulative_net_value'])['date']
                if not valid_dates.empty:
                    earliest_start_date = max(earliest_start_date, valid_dates.min())
                    latest_end_date = min(latest_end_date, valid_dates.max())
            else:
                 # 保留您的警告，以便您知道哪个文件有问题
                 print(f"警告：文件 {filename} 缺少必要的 'date' 或 'cumulative_net_value' 列，已跳过。")
        except Exception as e:
            print(f"读取文件 {filename} 时发生错误: {e}")
            
    if latest_end_date <= earliest_start_date:
        print("错误：无法找到有效的共同分析期。请检查文件日期范围。")
        return

    print(f"确定共同分析期：{earliest_start_date.strftime('%Y-%m-%d')} 至 {latest_end_date.strftime('%Y-%m-%d')}")
    
    # 第二步：计算所有基金的指标 (增加滚动收益)
    results = []
    for filename in file_list:
        fund_code = filename.replace('.csv', '')
        filepath = os.path.join(DATA_DIR, filename)
        
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.lower()
            
            if 'date' in df.columns and 'cumulative_net_value' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                metrics = calculate_metrics(df, earliest_start_date, latest_end_date)
            else:
                metrics = None
            
            if metrics:
                results.append({
                    '基金代码': fund_code,
                    '起始日期': earliest_start_date.strftime('%Y-%m-%d'),
                    '结束日期': latest_end_date.strftime('%Y-%m-%d'),
                    **metrics
                })
        except Exception as e:
            print(f"计算文件 {filename} 的指标时发生错误: {e}")
            
    # 第三步：生成统计表和分析总结 (列名汉化，增加短期收益列)
    if not results:
        print("没有成功计算出任何基金的指标。")
        return
        
    summary_df = pd.DataFrame(results)
    
    # 用于排序的数值列
    summary_df['夏普比率(总)_Num'] = pd.to_numeric(summary_df['夏普比率(总)'], errors='coerce') 
    
    # 格式化百分比和夏普比率
    for col in summary_df.columns:
        if '收益率' in col or '标准差' in col or '回撤' in col:
            # 应用百分比格式
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce').apply(lambda x: f"{x:.2%}" if pd.notna(x) else 'NaN')
        elif '夏普比率' in col and '_Num' not in col:
            # 应用夏普比率格式
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce').apply(lambda x: f"{x:.3f}" if pd.notna(x) else 'NaN')

    # 按夏普比率降序排序
    summary_df = summary_df.sort_values(by='夏普比率(总)_Num', ascending=False)
    summary_df = summary_df.drop(columns=['夏普比率(总)_Num'])
    
    # 将结果保存到 CSV
    summary_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\n--- 分析完成 ---\n结果已保存到 {OUTPUT_FILE}")
    print("\n按夏普比率排名的分析摘要：")
    
    # 使用 to_string 代替 to_markdown 避免依赖问题
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()