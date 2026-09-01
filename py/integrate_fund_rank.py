import pandas as pd
import os
import subprocess
import sys
import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('fund_rank_integrate.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def run_script(script_name, args=None):
    """运行脚本并捕获错误"""
    cmd = ['python', script_name] + (args or [])
    logger.info(f"运行脚本: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"{script_name} 运行成功: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"{script_name} 运行失败: {e.stderr}")
        sys.exit(1)

def main(start_date=None, end_date=None):
    """运行 fund-rank.py 并将其输出转换为 recommended_cn_funds.csv"""
    # 默认日期：过去30天
    today = datetime.date.today()
    end_date = end_date or today.strftime('%Y-%m-%d')
    start_date = start_date or (today - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
    logger.info(f"分析日期范围: {start_date} 到 {end_date}")

    # 运行 fund-rank.py
    run_script('fund-rank.py', [start_date, end_date])

    # 读取 fund-rank.py 输出
    result_file = f'result_{start_date}_{end_date}_C类.txt'
    if not os.path.exists(result_file):
        logger.error(f"错误: {result_file} 未生成！检查 fund-rank.py 是否运行成功。")
        sys.exit(1)

    # 转换为 recommended_cn_funds.csv
    try:
        df = pd.read_csv(result_file, sep='\t', encoding='utf-8')

        # 1. 选取 Top 80 基金
        df_top = df.head(80).copy()  # 取 Top 80 基金

        # 2. 定义排除关键字列表
        exclude_keywords = ['持有', '债', '币', '美元']
        
        # 3. 排除名称中含有关键字的基金
        original_count = len(df_top)
        
        # 使用正则表达式构建排除条件，'|' 表示“或”
        pattern = '|'.join(exclude_keywords)
        df_filtered = df_top[~df_top['名称'].str.contains(pattern, na=False)].copy()
        
        excluded_count = original_count - len(df_filtered)
        
        logger.info(f"从 {result_file} 提取 Top {original_count} 只基金")
        if excluded_count > 0:
            logger.info(f"已排除 {excluded_count} 只名称中含有 {exclude_keywords} 关键字的基金。剩余 {len(df_filtered)} 只。")

        # 4. 整理列名和数据
        df_filtered = df_filtered[['编码', '名称']]
        df_filtered.columns = ['代码', '名称']  # 适配后续脚本
        
        df_filtered.to_csv('recommended_cn_funds.csv', index=False, encoding='utf-8')
        logger.info("已生成 recommended_cn_funds.csv")
    except Exception as e:
        logger.error(f"转换 {result_file} 失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_date = sys.argv[1] if len(sys.argv) > 1 else None
    end_date = sys.argv[2] if len(sys.argv) > 2 else None
    main(start_date, end_date)
