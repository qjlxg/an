import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

# 检查并导入所需的库
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"❌ 缺少必要的Python库：{e}")
    print("请使用以下命令安装：pip install pandas numpy")
    exit()

class MarketMonitorParser:
    def __init__(self):
        """
        初始化解析器。这个类只负责解析本地文件，不进行网络请求。
        """
        pass

    def parse_signals_from_md(self, md_file='market_monitor_report.md'):
        """
        从 Markdown 表格解析买入信号基金。
        它会读取指定文件，查找并解析包含“行动信号”的表格，
        然后返回所有“弱买入”或“强买入”基金的代码。
        """
        print("🔍 正在检查 'market_monitor_report.md' 文件...")
        if not os.path.exists(md_file):
            print(f"❌ 未找到 {md_file} 文件")
            print("请确保该文件与脚本在同一个目录下。")
            return []

        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        print("📖 解析 Markdown 表格...")
        
        # 匹配包含"基金代码"和"行动信号"的表格
        table_pattern = r'(?s).*?\|.*?基金代码.*?\|.*?\|.*?\|.*?\|.*?\|.*?\|.*?行动信号.*?\|.*?(?=\n\n|\Z)'
        table_match = re.search(table_pattern, content)

        if not table_match:
            print("❌ 正则匹配失败，尝试逐行解析...")
            lines = content.split('\n')
            in_table = False
            table_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith('|') and '基金代码' in line and '行动信号' in line:
                    in_table = True
                    table_lines = [line]
                    continue
                if in_table:
                    if line.startswith('|') and len(line.split('|')) >= 8:
                        table_lines.append(line)
                    elif not line.strip() and len(table_lines) > 1:
                        in_table = False
            
            if table_lines:
                print(f"✅ 找到 {len(table_lines) - 2} 条表格数据")
                return self._parse_table_lines(table_lines)
            else:
                print("❌ 未找到包含买入信号的表格")
                return []

        table_content = table_match.group(0)
        lines = [line.strip() for line in table_content.split('\n') if line.strip()]
        
        header_line_index = -1
        for i, line in enumerate(lines):
            if line.startswith('|') and '基金代码' in line and '行动信号' in line:
                header_line_index = i
                break

        if header_line_index == -1:
            print("❌ 未找到表头行")
            return []
            
        print(f"✅ 找到 {len(lines) - 2} 条表格数据")
        return self._parse_table_lines(lines[header_line_index:])

    def _parse_table_lines(self, table_lines):
        """
        内部方法：从表格行中解析出基金代码和行动信号。
        """
        buy_signals = []
        data_start = 2 if len(table_lines) > 2 and '|---' in table_lines[1] else 1
        
        for i, line in enumerate(table_lines[data_start:], data_start):
            if not line.startswith('|'):
                continue
            
            parts = [part.strip() for part in line.split('|')]
            cells = [part for part in parts[1:-1]]
            
            if len(cells) < 8:
                continue
            
            fund_code = cells[0].strip()
            action_signal = cells[-1].strip()
            
            if re.match(r'^\d{6}$', fund_code) and '买入' in action_signal:
                buy_signals.append({
                    'fund_code': fund_code,
                    '最新净值': cells[1].strip(),
                    'RSI': cells[2].strip(),
                    '净值/MA50': cells[3].strip(),
                    'MACD信号': cells[4].strip(),
                    '布林带位置': cells[5].strip(),
                    '投资建议': cells[6].strip(),
                    '行动信号': action_signal
                })
                
        print(f"📊 最终结果: {len(buy_signals)} 只买入信号基金")
        return buy_signals

def main():
    """
    主函数：读取报告，解析买入信号并保存为 CSV。
    """
    print("🚀 基金买入信号解析器")
    print("=" * 50)
    
    parser = MarketMonitorParser()
    signals = parser.parse_signals_from_md()
    
    if not signals:
        print("\n💡 建议检查:")
        print("1. 'market_monitor_report.md' 文件是否在同一目录下")
        print("2. 文件中表格是否包含 '基金代码' 和 '行动信号' 列")
        print("3. '行动信号' 列是否包含 '买入' 关键词")
        return
    
    df = pd.DataFrame(signals)
    
    os.makedirs('data', exist_ok=True)
    today_date = datetime.now().strftime('%Y%m%d')
    filename = f"data/买入信号基金_{today_date}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 完成！总 {len(df)} 条记录")
    print(f"💾 汇总文件已保存至: {filename}")

if __name__ == "__main__":
    main()
