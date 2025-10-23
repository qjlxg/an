import pandas as pd
import requests
from datetime import datetime
import os
import time
from io import StringIO
from typing import Optional, List
import logging
from pathlib import Path

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
        从东方财富网获取特定年份的基金持仓信息
        
        Args:
            fund_code: 基金代码
            year: 年份
            
        Returns:
            持仓数据DataFrame或None
        """
        url = f"{self.base_url}/FundArchivesDatas.aspx?type=jjcc&code={fund_code}&topline=10&year={year}"
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # 使用 StringIO 包装字符串，避免FutureWarning
            tables = pd.read_html(StringIO(response.text), encoding='utf-8')
            
            if tables and len(tables) > 0:
                holdings_table = tables[0]
                # 数据清洗
                holdings_table = self._clean_holdings_data(holdings_table)
                logger.info(f"✅ 成功获取基金 {fund_code} 在 {year} 年的持仓数据，记录数：{len(holdings_table)}")
                return holdings_table
            else:
                logger.warning(f"⚠️ 基金 {fund_code} 在 {year} 年没有表格数据")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 网络请求失败 - 基金 {fund_code}, 年份 {year}: {e}")
            return None
        except pd.errors.EmptyDataError as e:
            logger.error(f"❌ 解析HTML表格失败 - 基金 {fund_code}, 年份 {year}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ 未知错误 - 基金 {fund_code}, 年份 {year}: {e}")
            return None
    
    def _clean_holdings_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗持仓数据"""
        if df.empty:
            return df
            
        # 移除空行和重复列
        df = df.dropna(how='all')
        
        # 标准化列名（如果需要）
        if not df.columns.empty:
            df.columns = df.columns.str.strip()
        
        # 转换数值列（如果包含百分比或金额）
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            # 尝试转换百分比格式
            if '%' in str(df[col].iloc[0] if len(df) > 0 else ''):
                df[col] = pd.to_numeric(df[col].str.replace('%', ''), errors='coerce')
        
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
    
    # 创建抓取器实例
    fetcher = FundHoldingsFetcher()
    
    # 执行批量抓取
    results = fetcher.batch_fetch(
        fund_codes=fund_codes,
        years=years_to_fetch,
        input_file=input_csv_path,
        output_dir=output_dir
    )
    
    # 输出总结
    logger.info("=" * 50)
    logger.info("📊 任务总结")
    logger.info(f"总任务数: {results['total']}")
    logger.info(f"成功抓取: {results['success']}")
    logger.info(f"抓取失败: {results['failed']}")
    logger.info(f"成功率: {(results['success']/results['total']*100):.1f}%")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
