import pandas as pd
import glob
import os
import sys

def load_stock_categories(category_path):
    """
    遍历指定目录，加载所有 .xlsx 格式的股票分类表。
    
    Args:
        category_path (str): 包含分类表的目录路径。
        
    Returns:
        dict: 一个字典，键为股票代码，值为其所属的分类。
    """
    all_categories = {}
    xlsx_files = glob.glob(os.path.join(category_path, "*.xlsx"))
    
    if not xlsx_files:
        print(f"未在 '{category_path}' 目录中找到任何 XLSX 文件。")
        return all_categories

    for f in xlsx_files:
        try:
            category_name = os.path.basename(f).split('.')[0].replace('分类表', '')
            
            df = pd.read_excel(f, header=0, engine='openpyxl')
            
            if '股票代码' not in df.columns or '股票名称' not in df.columns:
                print(f"文件 {f} 缺少关键列 '股票代码' 或 '股票名称'，跳过。")
                continue
            
            df['股票代码'] = df['股票代码'].astype(str).str.strip().str.zfill(6)
            
            for code in df['股票代码']:
                all_categories[code] = category_name
        except Exception as e:
            print(f"读取分类文件 {f} 时出错: {e}")
            continue
            
    return all_categories

def generate_fund_report(df, fund_code, report):
    """
    为单个基金生成详细分析报告。
    """
    report.append(f"## 基金代码: {fund_code} 持仓分析报告")
    report.append("---")
    
    unclassified_stocks = df[df['行业'].str.contains('未分类')]
    if not unclassified_stocks.empty:
        report.append("\n### 未能匹配到行业分类的股票列表")
        report.append("---")
        for quarter, group in unclassified_stocks.groupby('季度'):
            report.append(f"#### {quarter}")
            for index, row in group.iterrows():
                report.append(f"- **{row['股票名称']}** ({row['股票代码']}): 占净值比例 {row['占净值比例']:.2f}%")
        report.append("---\n")

    report.append("### 1. 重仓股变动")
    quarters = df['季度'].unique()
    if len(quarters) > 1:
        for i in range(len(quarters) - 1):
            current_q = quarters[i]
            next_q = quarters[i+1]
            
            current_holdings = df[df['季度'] == current_q][['股票代码', '股票名称', '占净值比例']].set_index('股票代码')
            next_holdings = df[df['季度'] == next_q][['股票代码', '股票名称', '占净值比例']].set_index('股票代码')
            
            new_additions = next_holdings.index.difference(current_holdings.index)
            removed = current_holdings.index.difference(next_holdings.index)
            
            report.append(f"#### 从 {current_q} 到 {next_q} 的变动")
            if not new_additions.empty:
                report.append("- **新增股票**：")
                for code in new_additions:
                    stock_name = next_holdings.loc[code, '股票名称']
                    ratio = next_holdings.loc[code, '占净值比例']
                    report.append(f"  - **{stock_name}** ({code}): 占净值比例 {ratio:.2f}%")
            if not removed.empty:
                report.append("- **移除股票**：")
                for code in removed:
                    stock_name = current_holdings.loc[code, '股票名称']
                    ratio = current_holdings.loc[code, '占净值比例']
                    report.append(f"  - **{stock_name}** ({code}): 占净值比例 {ratio:.2f}%")

            common_stocks = current_holdings.index.intersection(next_holdings.index)
            if not common_stocks.empty:
                report.append("- **持仓变动**：")
                for code in common_stocks:
                    name = current_holdings.loc[code, '股票名称']
                    current_ratio = current_holdings.loc[code, '占净值比例']
                    next_ratio = next_holdings.loc[code, '占净值比例']
                    diff = next_ratio - current_ratio
                    
                    if abs(diff) > 0.5:
                        action = "增持" if diff > 0 else "减持"
                        report.append(f"  - **{name}** ({code}): **{action}**，比例从 {current_ratio:.2f}% 变为 {next_ratio:.2f}% (变化 {diff:+.2f}%)")

    report.append("\n### 2. 行业偏好和持仓集中度")
    sector_summary = df.groupby(['季度', '行业'])['占净值比例'].sum().unstack(fill_value=0)
    
    sector_summary = sector_summary.loc[:, ~sector_summary.columns.str.contains('未分类')]
    sector_summary = sector_summary.loc[:, (sector_summary != 0).any(axis=0)]
    sector_summary = sector_summary.astype(float)
    
    report.append("#### 行业偏好（占净值比例之和）")
    if not sector_summary.empty:
        report.append("| 季度 | 行业 | 占比 | 进度条 |")
        report.append("|---|---|---|---|")
        for quarter, row in sector_summary.iterrows():
            sorted_sectors = row.sort_values(ascending=False)
            for sector, ratio in sorted_sectors.items():
                if ratio > 0:
                    progress_bar = '█' * int(ratio / 5)
                    report.append(f"| {quarter} | {sector} | {ratio:.2f}% | {progress_bar} |")
    else:
        report.append("无行业偏好数据可供分析。")

    concentration_summary = df.groupby('季度')['占净值比例'].sum()
    
    report.append("\n#### 前十大持仓集中度（占净值比例之和）")
    report.append("| 季度 | 占净值比例 | 进度条 |")
    report.append("|---|---|---|")
    for quarter, ratio in concentration_summary.items():
        progress_bar = '█' * int(ratio / 5)
        report.append(f"| {quarter} | {ratio:.2f}% | {progress_bar} |")

    report.append("\n### 3. 趋势总结和投资建议")
    report.append("> **免责声明**：本报告基于历史持仓数据进行分析，不构成任何投资建议。投资有风险，入市需谨慎。")
    report.append(f"\n基于对基金 **{fund_code}** 的历史持仓数据分析，本报告得出以下关键观察结果：")
    
    if len(concentration_summary) > 1:
        first_concentration = concentration_summary.iloc[0]
        last_concentration = concentration_summary.iloc[-1]
        concentration_diff = last_concentration - first_concentration
        
        if concentration_diff > 10:
            report.append("- **持仓集中度**：在分析期内，该基金的持仓集中度显著**上升**，表明基金经理正将资金集中到少数看好股票上。")
        elif concentration_diff < -10:
            report.append("- **持仓集中度**：在分析期内，该基金的持仓集中度显著**下降**，表明基金经理正在分散投资以降低风险。")
        else:
            report.append("- **持仓集中度**：该基金的持仓集中度在分析期内相对**稳定**，可能反映其投资风格稳健。")

    if not sector_summary.empty and len(sector_summary.index) > 1:
        first_quarter_summary = sector_summary.iloc[0]
        last_quarter_summary = sector_summary.iloc[-1]
        
        try:
            first_dominant_sector = first_quarter_summary.idxmax()
            last_dominant_sector = last_quarter_summary.idxmax()
            
            if first_dominant_sector != last_dominant_sector:
                report.append(f"- **行业偏好**：基金的投资偏好发生了明显变化，从**{first_dominant_sector}**转向了**{last_dominant_sector}**。")
            else:
                report.append(f"- **行业偏好**：该基金在分析期内主要偏向于**{first_dominant_sector}**行业。")
        except ValueError:
            report.append("- **行业偏好**：由于数据不足，无法分析行业偏好变化。")
    
    report.append("\n**总结与建议：**")
    report.append("  在考虑投资该基金时，建议将上述分析结果与其他因素结合考量，例如基金的过往业绩、基金经理的管理经验、基金规模以及费率等。")

def analyze_holdings():
    """
    遍历 fund_data 目录，对所有基金的持仓数据进行合并和分析，
    并生成一份总览报告和多份单基金报告。
    """
    base_path = 'fund_data'
    category_path = '分类表'
    all_files = glob.glob(os.path.join(base_path, "*.csv"))

    if not all_files:
        print("未在 'fund_data' 目录中找到任何 CSV 文件。")
        return

    stock_categories = load_stock_categories(category_path)
    if not stock_categories:
        print("未加载到任何股票分类数据，将使用默认板块分析。")
        sector_mapping = {
            '688': '科创板', '300': '创业板', '002': '中小板',
            '000': '主板', '600': '主板', '601': '主板',
            '603': '主板', '605': '主板', '005': '主板', '006': '主板',
        }
        use_detailed_categories = False
    else:
        use_detailed_categories = True

    all_funds_df_list = []
    fund_files = {}
    for f in all_files:
        try:
            fund_code = os.path.basename(f).split('_')[1]
            if fund_code not in fund_files:
                fund_files[fund_code] = []
            fund_files[fund_code].append(f)
        except IndexError:
            print(f"文件名格式不正确，跳过：{f}")
            continue

    if not fund_files:
        print("未找到任何有效基金文件。")
        return

    for fund_code, files in fund_files.items():
        df_list = []
        for f in files:
            try:
                df = pd.read_csv(f, engine='python')
                
                column_mapping = {
                    '占净值 比例': '占净值比例', 
                    '占净值比例': '占净值比例',
                    '持仓市值 （万元）': '持仓市值',
                    '持仓市值': '持仓市值',
                    '市值': '持仓市值',
                    '持仓市值 （万元人民币）': '持仓市值',
                    '股票名称': '股票名称',
                    '股票代码': '股票代码',
                    '季度': '季度'
                }
                
                df.columns = [column_mapping.get(col, col) for col in df.columns]

                required_cols = ['股票代码', '股票名称', '占净值比例', '持仓市值', '季度']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise KeyError(f"缺少关键列 {missing_cols}")

                df['占净值比例'] = df['占净值比例'].astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False)
                df['占净值比例'] = pd.to_numeric(df['占净值比例'], errors='coerce')
                
                df['持仓市值'] = df['持仓市值'].astype(str).str.replace(',', '', regex=False)
                df['持仓市值'] = pd.to_numeric(df['持仓市值'], errors='coerce')
                
                df['股票代码'] = df['股票代码'].astype(str).str.strip().str.zfill(6)
                
                if use_detailed_categories:
                    df['行业'] = df['股票代码'].map(stock_categories).fillna('未分类')
                else:
                    df['行业'] = df['股票代码'].astype(str).str[:3].map(sector_mapping).fillna('未分类')
                
                df['基金代码'] = fund_code
                df_list.append(df)
            except KeyError as e:
                print(f"读取文件 {f} 时出错：缺少关键列 {e}")
                continue
            except Exception as e:
                print(f"读取文件 {f} 时出错：{e}")
                continue
        
        if df_list:
            combined_df = pd.concat(df_list, ignore_index=True)
            combined_df['季度'] = combined_df['季度'].str.replace('年', '-Q')
            combined_df['年份'] = combined_df['季度'].str.split('-').str[0].astype(int)
            combined_df['季度编号'] = combined_df['季度'].str.split('-').str[1].str.replace('季度', '')
            combined_df.sort_values(by=['年份', '季度编号'], inplace=True)
            all_funds_df_list.append(combined_df)

    if not all_funds_df_list:
        print("所有基金文件都因错误而跳过，无法生成报告。")
        return

    all_funds_combined_df = pd.concat(all_funds_df_list, ignore_index=True)
    report = []

    # 总览报告部分
    report.append("# 所有基金持仓总览")
    report.append("---")
    report.append("### 整体行业偏好")
    
    # 逻辑优化：按季度、行业、基金分组汇总持仓市值，然后按季度和行业汇总
    overall_sector_fund_summary = all_funds_combined_df.groupby(['季度', '基金代码', '行业'])['持仓市值'].sum().reset_index()

    # 按季度和行业汇总总市值，用于排序
    overall_sector_total_summary = overall_sector_fund_summary.groupby(['季度', '行业'])['持仓市值'].sum().sort_values(ascending=False).reset_index()

    unique_quarters = all_funds_combined_df['季度'].unique()

    for quarter in unique_quarters:
        report.append(f"\n#### {quarter} 行业持仓总览")
        
        # 获取当前季度市值排名前5的行业
        top_sectors = overall_sector_total_summary[overall_sector_total_summary['季度'] == quarter].head(5)
        
        for index, row in top_sectors.iterrows():
            sector = row['行业']
            total_market_value = row['持仓市值']
            
            report.append(f"\n- **{sector}**：总持仓市值 **{total_market_value:.2f} 万元**")
            
            # 找到所有持有该行业的基金
            fund_contributions = overall_sector_fund_summary[
                (overall_sector_fund_summary['季度'] == quarter) & 
                (overall_sector_fund_summary['行业'] == sector)
            ].sort_values(by='持仓市值', ascending=False)
            
            for _, fund_row in fund_contributions.iterrows():
                fund_code = fund_row['基金代码']
                fund_market_value = fund_row['持仓市值']
                contribution_ratio = (fund_market_value / total_market_value) * 100 if total_market_value > 0 else 0
                report.append(f"  - 基金代码 {fund_code}：持仓市值 {fund_market_value:.2f} 万元 ({contribution_ratio:.2f}%)")
    
    # 汇总未分类股票
    unclassified_overall = all_funds_combined_df[all_funds_combined_df['行业'].str.contains('未分类')]
    if not unclassified_overall.empty:
        report.append("\n### 未分类股票列表（按总市值汇总）")
        report.append("---")
        unclassified_summary = unclassified_overall.groupby(['股票代码', '股票名称'])['持仓市值'].sum().sort_values(ascending=False).reset_index()
        report.append(unclassified_summary.to_markdown(index=False))
    
    report.append("\n---")

    # 单基金详细报告部分
    for fund_code in all_funds_combined_df['基金代码'].unique():
        fund_df = all_funds_combined_df[all_funds_combined_df['基金代码'] == fund_code].copy()
        generate_fund_report(fund_df, fund_code, report)
    
    with open('analysis_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print("分析报告已生成：analysis_report.md")

if __name__ == "__main__":
    analyze_holdings()
