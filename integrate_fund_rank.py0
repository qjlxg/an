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
        df = df[['编码', '名称']].head(50)  # 取 Top 50 基金
        df.columns = ['代码', '名称']  # 适配后续脚本
        logger.info(f"从 {result_file} 提取 {len(df)} 只基金")
        df.to_csv('recommended_cn_funds.csv', index=False, encoding='utf-8-sig')
        logger.info("已生成 recommended_cn_funds.csv")
    except Exception as e:
        logger.error(f"转换 {result_file} 失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_date = sys.argv[1] if len(sys.argv) > 1 else None
    end_date = sys.argv[2] if len(sys.argv) > 2 else None
    main(start_date, end_date)
