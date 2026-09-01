import pandas as pd
import requests
from datetime import datetime
import os
import time
from io import StringIO
from typing import Optional, List
import logging
from pathlib import Path
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FundHoldingsFetcher:
    """基金持仓数据抓取器"""
    
    def __init__(self, base_url: str = "http://fundf10.eastmoney.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def fetch_fund_holdings(self, fund_code: str, year: int) -> Optional[pd.DataFrame]:
        """
        从东方财富网获取特定年份的所有基金持仓信息（包含所有季度）
        
        Args:
            fund_code: 基金代码
            year: 年份
            
        Returns:
            合并后的持仓数据DataFrame或None
        """
        url = f"{self.base_url}/FundArchivesDatas.aspx?type=jjcc&code={fund_code}&topline=10&year={year}"
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # 使用 StringIO 包装字符串，避免FutureWarning
            tables = pd.read_html(StringIO(response.text), encoding='utf-8')
            
            if not tables:
                logger.warning(f"⚠️ 基金 {fund_code} 在 {year} 年没有表格数据")
                return None

            full_year_df = pd.DataFrame()
            
            for i, table in enumerate(tables):
                # 从表格上方的文本中提取季度信息
                quarter_match = re.search(r'(\d{4}年\d季度)', response.text.split('<table')[i])
                quarter_info = quarter_match.group(1) if quarter_match else f"Q{i+1}"
                
                # 数据清洗
                cleaned_df = self._clean_holdings_data(table)
                
                if not cleaned_df.empty:
                    cleaned_df['季度'] = quarter_info
                    full_year_df = pd.concat([full_year_df, cleaned_df], ignore_index=True)
            
            if not full_year_df.empty:
                logger.info(f"✅ 成功获取基金 {fund_code} 在 {year} 年的全部季度持仓数据，总记录数：{len(full_year_df)}")
                return full_year_df
            else:
                logger.warning(f"⚠️ 基金 {fund_code} 在 {year} 年没有有效的持仓数据")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 网络请求失败 - 基金 {fund_code}, 年份 {year}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ 解析HTML表格或处理数据失败 - 基金 {fund_code}, 年份 {year}: {e}")
            return None
    
    def _clean_holdings_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗持仓数据"""
        if df.empty:
            return df
            
        # 移除空行
        df = df.dropna(how='all')
        
        # 标准化列名
        if not df.columns.empty:
            df.columns = df.columns.str.strip()
        
        # 转换数值列
        numeric_cols = ['占净值比例', '持股数（万股）', '持仓市值（万元）']
        for col in numeric_cols:
            if col in df.columns:
                # 移除可能的逗号或百分号
                df[col] = df[col].astype(str).str.replace(',', '').str.replace('%', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 丢弃无效行
        df = df.dropna(subset=['股票代码', '股票名称'])
        
        return df
    
    def batch_fetch(self, fund_codes: List[str], years: List[int], 
                    input_file: str, output_dir: str = 'fund_data') -> dict:
        """
        批量抓取基金持仓数据
        
        Args:
            fund_codes: 基金代码列表
            years: 要抓取的年份列表
            input_file: 输入CSV文件路径
            output_dir: 输出目录
            
        Returns:
            抓取结果统计字典
        """
        results = {'success': 0, 'failed': 0, 'total': len(fund_codes) * len(years)}
        
        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)
        
        # 读取输入文件
        try:
            input_df = pd.read_csv(input_file)
            logger.info(f"📊 读取输入文件：{input_file}，包含 {len(input_df)} 条记录")
        except Exception as e:
            logger.error(f"❌ 无法读取输入文件 {input_file}: {e}")
            return results
        
        # 确保基金代码格式正确
        fund_codes = [str(code).zfill(6) for code in fund_codes]
        
        for i, code in enumerate(fund_codes, 1):
            for year in years:
                logger.info(f"[{i}/{len(fund_codes)}] 🔍 处理基金 {code} - {year}年")
                
                holdings_df = self.fetch_fund_holdings(code, year)
                
                if holdings_df is not None and not holdings_df.empty:
                    # 保存数据
                    filename = f'持仓_{code}_{year}.csv'
                    output_path = Path(output_dir) / filename
                    
                    holdings_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    logger.info(f"💾 数据已保存: {output_path}")
                    results['success'] += 1
                else:
                    results['failed'] += 1
                
                # 延时避免被封
                time.sleep(2)
        
        logger.info(f"🎉 批量抓取完成！成功: {results['success']}, 失败: {results['failed']}")
        return results

    def analyze_holdings_changes(self, fund_code: str, years: List[int], output_dir: str = 'fund_data', 
                                 analysis_dir: str = 'fund_analysis') -> dict:
        """
        分析基金持仓变化
        
        Args:
            fund_code: 基金代码
            years: 年份列表
            output_dir: 持仓数据目录
            analysis_dir: 分析输出目录
            
        Returns:
            分析结果统计
        """
        Path(analysis_dir).mkdir(exist_ok=True)
        results = {'analyzed_pairs': 0, 'total_pairs': len(years) - 1}
        
        data_dict = {}
        for year in years:
            file_path = Path(output_dir) / f'持仓_{fund_code}_{year}.csv'
            if file_path.exists():
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                data_dict[year] = df # 读取文件后直接使用
            else:
                logger.warning(f"⚠️ 缺少 {fund_code} {year}年的持仓数据文件")
        
        if len(data_dict) < 2:
            logger.warning(f"⚠️ 基金 {fund_code} 可用年份不足，无法分析变化")
            return results
        
        sorted_years = sorted(data_dict.keys())
        for i in range(len(sorted_years) - 1):
            year1 = sorted_years[i]
            year2 = sorted_years[i+1]
            
            df1 = data_dict[year1]
            df2 = data_dict[year2]
            
            # 合并数据
            merged = pd.merge(
                df1[['股票代码', '股票名称', '占净值比例']],
                df2[['股票代码', '股票名称', '占净值比例']],
                on=['股票代码', '股票名称'],
                how='outer',
                suffixes=(f'_{year1}', f'_{year2}')
            )
            
            # 计算变化
            prop_col1 = f'占净值比例_{year1}'
            prop_col2 = f'占净值比例_{year2}'
            
            merged[prop_col1] = merged[prop_col1].fillna(0)
            merged[prop_col2] = merged[prop_col2].fillna(0)
            
            merged['比例变化'] = merged[prop_col2] - merged[prop_col1]
            merged['变化类型'] = merged.apply(
                lambda row: '新买入' if row[prop_col1] == 0 else 
                            '卖出' if row[prop_col2] == 0 else 
                            '增加' if row['比例变化'] > 0 else 
                            '减少' if row['比例变化'] < 0 else '不变',
                axis=1
            )
            
            # 排序按变化绝对值
            merged = merged.sort_values(by='比例变化', key=abs, ascending=False)
            
            # 保存分析结果
            filename = f'变化_{fund_code}_{year1}_{year2}.csv'
            output_path = Path(analysis_dir) / filename
            merged.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"📈 持仓变化分析已保存: {output_path}")
            
            results['analyzed_pairs'] += 1
        
        return results

    def batch_analyze(self, fund_codes: List[str], years: List[int], 
                      output_dir: str = 'fund_data', analysis_dir: str = 'fund_analysis') -> dict:
        """
        批量分析持仓变化
        
        Args:
            fund_codes: 基金代码列表
            years: 年份列表
            output_dir: 持仓数据目录
            analysis_dir: 分析输出目录
            
        Returns:
            批量分析统计
        """
        batch_results = {'success': 0, 'failed': 0, 'total': len(fund_codes)}
        
        for i, code in enumerate(fund_codes, 1):
            logger.info(f"[{i}/{len(fund_codes)}] 📊 分析基金 {code} 持仓变化")
            results = self.analyze_holdings_changes(code, years, output_dir, analysis_dir)
            if results['analyzed_pairs'] > 0:
                batch_results['success'] += 1
            else:
                batch_results['failed'] += 1
        
        logger.info(f"🎉 批量分析完成！成功: {batch_results['success']}, 失败: {batch_results['failed']}")
        return batch_results

def main():
    """主函数"""
    # 当前日期
    today_date = datetime.now().strftime('%Y%m%d')
    input_csv_path = f'data/买入信号基金_{today_date}.csv'
    
    logger.info(f"🚀 开始执行基金持仓数据抓取任务")
    logger.info(f"📅 当前日期: {today_date}")
    
    # 检查输入文件
    if not Path(input_csv_path).exists():
        logger.error(f"❌ 输入文件不存在: {input_csv_path}")
        logger.info("💡 请确保文件路径正确，或者手动创建示例文件")
        return
    
    # 读取基金代码
    try:
        df = pd.read_csv(input_csv_path)
        fund_codes = df['fund_code'].unique().tolist()
        logger.info(f"📋 找到 {len(fund_codes)} 个唯一基金代码")
    except Exception as e:
        logger.error(f"❌ 读取基金代码失败: {e}")
        return
    
    # 配置抓取参数
    years_to_fetch = [2023, 2024, 2025]
    output_dir = 'fund_data'
    analysis_dir = 'fund_analysis'
    
    # 创建抓取器实例
    fetcher = FundHoldingsFetcher()
    
    # 执行批量抓取
    fetch_results = fetcher.batch_fetch(
        fund_codes=fund_codes,
        years=years_to_fetch,
        input_file=input_csv_path,
        output_dir=output_dir
    )
    
    # 执行批量分析
    analyze_results = fetcher.batch_analyze(
        fund_codes=fund_codes,
        years=years_to_fetch,
        output_dir=output_dir,
        analysis_dir=analysis_dir
    )
    
    # 输出总结
    logger.info("=" * 50)
    logger.info("📊 任务总结")
    logger.info(f"抓取总任务数: {fetch_results['total']}")
    logger.info(f"抓取成功: {fetch_results['success']}")
    logger.info(f"抓取失败: {fetch_results['failed']}")
    logger.info(f"分析成功基金: {analyze_results['success']}")
    logger.info(f"分析失败基金: {analyze_results['failed']}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
