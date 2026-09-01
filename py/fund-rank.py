#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import datetime
import glob
import urllib.request
import json
import sys
import re
import threading
import queue
import os
import math

# --- 常量定义 ---
# 假设年化交易日为252天
ANNUALIZATION_FACTOR = 252
# 假设年化无风险收益率为3% (可根据实际情况调整)
RISK_FREE_RATE = 0.03
# 均线周期定义
SHORT_MA_PERIOD = 20 # 短期均线 (约一个月交易日)
LONG_MA_PERIOD = 60  # 长期均线 (约一个季度交易日)

# --- 原始使用方法 ---
def usage():
    print('fund-rank.py usage:')
    print('\tpython fund.py start-date end-date fund-code=none\n')
    print('\tdate format ****-**-**')
    print('\t\tstart-date must before end-date')
    print('\tfund-code default none')
    print('\t\tif not input, get top 20 funds from all funds in C类.txt')
    print('\t\telse get that fund\'s rate of rise and risk metrics\n')
    print('\teg:\tpython fund-rank.py 2017-03-01 2017-03-25')
    print('\teg:\tpython fund-rank.py 2017-03-01 2017-03-25 377240')

# --- 原始函数：获取某一基金在某一日的累计净值数据 (为保留原功能而保留) ---
def get_jingzhi(strfundcode, strdate):
    try:
        url = 'http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code=' + \
              strfundcode + '&page=1&per=20&sdate=' + strdate + '&edate=' + strdate
        response = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        return '-1'
    except Exception as e:
        return '-1'

    json_fund_value = response.read().decode('utf-8')
    tr_re = re.compile(r'<tr>(.*?)</tr>')
    item_re = re.compile(r'''<td>(\d{4}-\d{2}-\d{2})</td><td.*?>(.*?)</td><td.*?>(.*?)</td><td.*?>(.*?)</td><td.*?>(.*?)</td><td.*?>(.*?)</td><td.*?></td>''', re.X)

    jingzhi = '-1'
    for line in tr_re.findall(json_fund_value):
        match = item_re.match(line)
        if match:
            entry = match.groups()
            jingzhi1 = entry[1] # 单位净值
            jingzhi2 = entry[2] # 累计净值
            
            if jingzhi2.strip() == '':
                jingzhi = '-1'
            elif jingzhi2.find('%') > -1:
                jingzhi = '-1'
            elif float(jingzhi1) > float(jingzhi2):
                jingzhi = entry[1]
            else:
                jingzhi = entry[2]

    return jingzhi
    
# --- 新增函数：从本地文件加载历史净值数据 (已适应 CSV 格式) ---
def load_local_data(strfundcode, strsdate, stredate):
    """
    从fund_data目录加载基金历史净值数据，适应 CSV 格式：
    读取：第1列(日期) 和 第3列(累计净值)
    """
    # 优先尝试 .txt，再尝试 .csv
    data_file = os.path.join('fund_data', f'{strfundcode}.txt')
    if not os.path.exists(data_file):
        data_file = os.path.join('fund_data', f'{strfundcode}.csv')
        if not os.path.exists(data_file):
            return None

    net_values_map = {} # {date: net_value}

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]: # 跳过第一行表头
                parts = line.strip().split(',')
                # 检查至少有3列：日期,单位净值,累计净值
                if len(parts) >= 3:
                    date_str = parts[0].strip()
                    try:
                        # 从索引 2 读取累计净值
                        net_value = float(parts[2].strip()) 
                        net_values_map[date_str] = net_value
                    except ValueError:
                        continue
    except Exception as e:
        return None
    
    # 筛选出在指定日期范围内的数据，并按日期排序
    sorted_dates = sorted([d for d in net_values_map.keys() if strsdate <= d <= stredate])
    
    if not sorted_dates:
        return None

    # 返回所有在范围内的日期和净值序列
    return sorted_dates, [net_values_map[d] for d in sorted_dates]

# --- 新增函数：计算简单移动平均 (SMA) ---
def calculate_moving_average(net_values, period):
    """
    计算净值序列末尾的简单移动平均。
    如果数据点数量不足 period，则返回 None。
    """
    if len(net_values) < period:
        return None
    
    # 截取序列末尾 period 个数据点
    ma_values = net_values[-period:]
    return sum(ma_values) / period

# --- 新增函数：计算最大回撤 ---
def calculate_mdd(net_values):
    """计算最大回撤（Maximum Drawdown）"""
    if not net_values:
        return 0.0

    peak_value = net_values[0]
    max_drawdown = 0.0

    for current_value in net_values:
        if current_value > peak_value:
            peak_value = current_value
        
        drawdown = (peak_value - current_value) / peak_value
        
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return round(max_drawdown * 100, 2) # 返回百分比

# --- 新增函数：计算夏普比率 ---
def calculate_sharpe_ratio(net_values):
    """
    计算夏普比率（Sharpe Ratio）。
    返回: (sharpe_ratio, warning_type)
    """
    if len(net_values) < 2:
        return 0.0, "NODATA"

    daily_returns = []
    for i in range(1, len(net_values)):
        ret = (net_values[i] / net_values[i-1]) - 1
        daily_returns.append(ret)

    num_trading_days = len(daily_returns)
    
    if num_trading_days < 10: 
        return 0.0, "INSUFFICIENT_DATA"

    avg_daily_return = sum(daily_returns) / num_trading_days
    
    variance = sum([(r - avg_daily_return) ** 2 for r in daily_returns]) / num_trading_days
    std_dev_daily_return = math.sqrt(variance)

    if std_dev_daily_return == 0:
        return 0.0, "ZERO_VOLATILITY"

    sharpe_ratio = (avg_daily_return * ANNUALIZATION_FACTOR - RISK_FREE_RATE) / \
                   (std_dev_daily_return * math.sqrt(ANNUALIZATION_FACTOR))
    
    return round(sharpe_ratio, 4), "OK"
    
# --- 新增函数：计算索提诺比率 ---
def calculate_sortino_ratio(net_values):
    """
    计算索提诺比率（Sortino Ratio）。
    返回: (sortino_ratio, warning_type)
    """
    if len(net_values) < 2:
        return 0.0, "NODATA"

    daily_returns = []
    for i in range(1, len(net_values)):
        ret = (net_values[i] / net_values[i-1]) - 1
        daily_returns.append(ret)

    num_trading_days = len(daily_returns)
    
    if num_trading_days < 10: 
        return 0.0, "INSUFFICIENT_DATA"
    
    # 目标回报率 (Target Return) 设为无风险利率的日化值
    daily_risk_free_rate = RISK_FREE_RATE / ANNUALIZATION_FACTOR
    
    # 计算平均超额回报
    avg_daily_return = sum(daily_returns) / num_trading_days
    avg_daily_excess_return = avg_daily_return - daily_risk_free_rate

    # 计算下行标准差 (Downside Deviation)
    downside_returns = [min(0, r - daily_risk_free_rate) for r in daily_returns]
    
    if not downside_returns or all(d == 0 for d in downside_returns):
        # 无下行风险或数据不足，索提诺比率高得不可信，设为 0.0 或一个极大值。
        # 这里设为 0.0，避免误导。
        return 0.0, "ZERO_DOWNSIDE_RISK"

    # 下行方差：仅计算低于目标回报率（Rf）的超额回报的平方和
    downside_variance = sum([d ** 2 for d in downside_returns]) / num_trading_days
    downside_std_dev = math.sqrt(downside_variance)
    
    # 索提诺比率公式
    # 注意：分子使用年化平均超额回报，分母使用年化下行标准差
    sortino_ratio = (avg_daily_excess_return * ANNUALIZATION_FACTOR) / \
                    (downside_std_dev * math.sqrt(ANNUALIZATION_FACTOR))

    return round(sortino_ratio, 4), "OK"


# --- 线程工作函数 (已加入索提诺比率) ---
def worker(q, strsdate, stredate, result_queue):
    while not q.empty():
        fund = q.get()
        strfundcode = fund[0]
        
        local_data = load_local_data(strfundcode, strsdate, stredate)

        jingzhimin = '0'
        jingzhimax = '0'
        jingzhidif = 0.0
        jingzhirise = 0.0
        max_drawdown = 0.0
        sharpe_ratio = 0.0
        sortino_ratio = 0.0  # 新增
        ma_trend = 'N/A' 

        if local_data:
            sorted_dates, net_values = local_data
            
            if len(net_values) > 1:
                jingzhimin = '%.4f' % net_values[0]
                jingzhimax = '%.4f' % net_values[-1]
                
                jingzhidif = float('%.4f' % (net_values[-1] - net_values[0]))
                
                if float(jingzhimin) != 0:
                    jingzhirise = float('%.2f' % (jingzhidif * 100 / float(jingzhimin)))
                
                max_drawdown = calculate_mdd(net_values)
                sharpe_ratio, sharpe_warning = calculate_sharpe_ratio(net_values)
                sortino_ratio, sortino_warning = calculate_sortino_ratio(net_values) # 计算索提诺比率
                
                # --- 均线计算和趋势判断 ---
                sma20 = calculate_moving_average(net_values, SHORT_MA_PERIOD)
                sma60 = calculate_moving_average(net_values, LONG_MA_PERIOD)
                
                if sma20 is not None and sma60 is not None:
                    if sma20 > sma60:
                        ma_trend = '↑'
                    elif sma20 < sma60:
                        ma_trend = '↓'
                    else:
                        ma_trend = '—'
                else:
                    ma_trend = 'N/A' 
                # --- 均线计算结束 ---

                # 输出警告（保留原有功能）
                if sharpe_warning != "OK":
                    print(f"Warning: Fund {strfundcode} ({fund[2]}) Sharpe: {sharpe_warning}.")
                if sortino_warning != "OK":
                    print(f"Warning: Fund {strfundcode} ({fund[2]}) Sortino: {sortino_warning}.")
            else:
                 print(f"Warning: Fund {strfundcode} ({fund[2]}) has insufficient data points (less than 2) in the period.")
        else:
             print(f"Warning: Fund {strfundcode} ({fund[2]}) local data not found or incomplete.")


        # fund: [0:代码, 1:简写, 2:名称, 3:类型, 4:状态, 5:净值min, 6:净值max, 7:净增长, 8:增长率, 9:最大回撤, 10:夏普比率, 11:MA趋势, 12:索提诺比率]
        fund.append(jingzhimin)     # 5
        fund.append(jingzhimax)     # 6
        fund.append(jingzhidif)     # 7
        fund.append(jingzhirise)    # 8
        fund.append(max_drawdown)   # 9 
        fund.append(sharpe_ratio)   # 10
        fund.append(ma_trend)       # 11 
        fund.append(sortino_ratio)  # 12 - 新增
        
        result_queue.put(fund)
        print('process fund:\t' + fund[0] + '\t' + fund[2])
        q.task_done()

# --- 主函数 (已更新排序逻辑和输出格式) ---
def main(argv):
    gettopnum = 50
    
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        usage()
        sys.exit(1)
    
    strsdate = sys.argv[1]
    stredate = sys.argv[2]
    
    strtoday = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
    tdatetime = datetime.datetime.strptime(strtoday, '%Y-%m-%d')
    
    sdatetime = datetime.datetime.strptime(strsdate, '%Y-%m-%d')
    if sdatetime.isoweekday() in [6, 7]:
        sdatetime += datetime.timedelta(days=- (sdatetime.isoweekday() - 5))
    strsdate = datetime.datetime.strftime(sdatetime, '%Y-%m-%d')

    edatetime = datetime.datetime.strptime(stredate, '%Y-%m-%d')
    if edatetime.isoweekday() in [6, 7]:
        edatetime += datetime.timedelta(days=- (edatetime.isoweekday() - 5))
    stredate = datetime.datetime.strftime(edatetime, '%Y-%m-%d')

    if edatetime <= sdatetime or tdatetime <= sdatetime or tdatetime <= edatetime:
        print('date input error!\n')
        usage()
        sys.exit(1)

    # --- 处理单个基金查询 (已加入索提诺比率输出) ---
    if len(sys.argv) == 4:
        strfundcode = sys.argv[3]
        
        local_data = load_local_data(strfundcode, strsdate, stredate)
        
        if not local_data:
            print(f'Cannot find local data for {strfundcode} or data is incomplete/missing!\n')
            usage()
            sys.exit(1)

        sorted_dates, net_values = local_data
        
        if len(net_values) < 2:
             print(f'Local data for {strfundcode} has fewer than 2 entries in the period!\n')
             usage()
             sys.exit(1)

        jingzhimin = '%.4f' % net_values[0]
        jingzhimax = '%.4f' % net_values[-1]
        
        jingzhidif = float('%.4f' % (net_values[-1] - net_values[0]))
        jingzhirise = float('%.2f' % (jingzhidif * 100 / float(jingzhimin)))
        
        max_drawdown = calculate_mdd(net_values)
        sharpe_ratio, sharpe_warning = calculate_sharpe_ratio(net_values)
        sortino_ratio, sortino_warning = calculate_sortino_ratio(net_values) # 计算索提诺比率
        
        # --- 单个基金的均线计算 ---
        sma20 = calculate_moving_average(net_values, SHORT_MA_PERIOD)
        sma60 = calculate_moving_average(net_values, LONG_MA_PERIOD)
        ma_trend = 'N/A'
        
        if sma20 is not None and sma60 is not None:
            if sma20 > sma60:
                ma_trend = '↑' 
            elif sma20 < sma60:
                ma_trend = '↓' 
            else:
                ma_trend = '—'
        # --- 均线计算结束 ---

        if sharpe_warning != "OK":
            print(f"Warning: Fund {strfundcode} Sharpe: {sharpe_warning}.")
        if sortino_warning != "OK":
            print(f"Warning: Fund {strfundcode} Sortino: {sortino_warning}.")
                 
        print('fund:' + strfundcode + '\n')
        
        # 单基金输出格式 (已加入索提诺比率)
        print(f'索提诺比率\t夏普比率\tMA趋势\t增长率\t净增长\t最大回撤\t{strsdate}\t{stredate}')
        print(f'{str(sortino_ratio)}\t\t{str(sharpe_ratio)}\t\t{ma_trend}\t\t{str(jingzhirise)}%\t{str(jingzhidif)}\t{str(max_drawdown)}%\t{jingzhimin}\t{jingzhimax}')
        sys.exit(0)
        
    # --- 基金列表获取 (从 C类.txt 获取代码) ---
    
    c_funds_list = []
    c_list_file = 'C类.txt'
    if not os.path.exists(c_list_file):
        print(f'Error: C类.txt file not found in current directory!')
        sys.exit(1)
        
    print(f'从 {c_list_file} 读取基金代码...')
    try:
        with open(c_list_file, 'r', encoding='utf-8') as f:
            for line in f:
                code = line.strip()
                if code and code != 'code':
                    c_funds_list.append([code, 'N/A', 'N/A', 'C类', 'N/A'])
    except Exception as e:
        print(f'Error reading {c_list_file}: {e}')
        sys.exit(1)
        
    all_funds_list = c_funds_list
    print('已读取 C 类基金数量：' + str(len(all_funds_list)))
      
    print('start:')
    print(datetime.datetime.now())
    print('funds sum:' + str(len(all_funds_list)))
    
    # --- 并行处理部分开始 (保持不变) ---
    task_queue = queue.Queue()
    result_queue = queue.Queue()

    for fund in all_funds_list:
        task_queue.put(fund)

    threads = []
    num_threads = 10 
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(task_queue, strsdate, stredate, result_queue))
        t.daemon = True
        t.start()
        threads.append(t)

    task_queue.join()

    all_funds_list = []
    while not result_queue.empty():
        all_funds_list.append(result_queue.get())
    # --- 并行处理部分结束 ---

    # 注意：结果文件名中加入了 SortinoRank 标识
    fileobject = open('result_' + strsdate + '_' + stredate + '_C类_Local_Analysis_SortinoRank.txt', 'w')
    
    # *** 核心修改：按索提诺比率 (fund[12]) 降序排列 ***
    print("\n--- 报告排序：按索提诺比率 (Sortino Ratio) 降序排列，寻找下行风险控制最好的基金 ---")
    all_funds_list.sort(key=lambda fund: fund[12], reverse=True) 
    
    # *** 报告头部：调整列顺序，将索提诺比率、夏普比率、MA趋势、增长率提前 ***
    strhead = '排序\t' + '编码\t\t' + '名称\t\t' + '类型\t\t' + \
              '索提诺比率\t' + '夏普比率\t' + 'MA趋势\t\t' + '增长率\t' + \
              '净增长\t' + '最大回撤\t' + strsdate + '\t' + stredate + '\n'
    print(strhead)
    fileobject.write(strhead)
    
    # fund: [0:代码, 1:简写, 2:名称, 3:类型, 4:状态, 5:净值min, 6:净值max, 7:净增长, 8:增长率, 9:最大回撤, 10:夏普比率, 11:MA趋势, 12:索提诺比率]
    for index in range(len(all_funds_list)):
        fund_data = all_funds_list[index]
        
        # *** 进阶筛选策略应用：高夏普比率 (> 1.5) 且 低最大回撤 (< 15%) ***
        # 满足条件的基金在打印时用星号 (*) 标记，以突出显示
        is_premium_fund = fund_data[10] >= 1.5 and fund_data[9] < 15.0
        
        # *** 报告内容：按新的顺序输出 ***
        strcontent = f"{index+1}{'*' if is_premium_fund else ''}\t" \
                     f"{fund_data[0]}\t" \
                     f"{fund_data[2]}\t\t" \
                     f"{fund_data[3]}\t\t" \
                     f"{str(fund_data[12])}\t\t" \
                     f"{str(fund_data[10])}\t\t" \
                     f"{fund_data[11]}\t\t\t" \
                     f"{str(fund_data[8])}%\t\t" \
                     f"{str(fund_data[7])}\t" \
                     f"{str(fund_data[9])}%\t\t" \
                     f"{fund_data[5]}\t\t" \
                     f"{fund_data[6]}\n"
                     
        print(strcontent)
        fileobject.write(strcontent)
        
        if index >= gettopnum:
            break
            
    # 打印筛选提示
    print("\n* 排序后的列表中，带星号 (*) 的基金同时满足：夏普比率 >= 1.5 且 最大回撤 < 15%。")
        
    fileobject.close()
    
    print('end:')
    print(datetime.datetime.now())
    
    sys.exit(0)
    
if __name__ == "__main__":
    main(sys.argv)
