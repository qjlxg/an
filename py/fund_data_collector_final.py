# encoding:utf-8
# FileName: craw_project
# Author:    wzg
# email:     1010490079@qq.com
# Date:      2021/04/30 19:50
# Description: 爬取天天基金网指定基金代码的详细信息和持仓数据
import json
import random
import re
import sys
import time
from collections import OrderedDict

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException # 引入requests异常

# 显示所有列
from craw_tools.get_ua import get_ua

pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


def resolve_rank_info(data):
    """
    解析每一页的所有基金的收益数据 (此函数在文件读取模式下不再使用)
    @param data:
    @return:
    """
    # 保持原样，但在 main 函数中不再调用
    rank_pages_data = []
    data = data.replace("[", "").replace("]", "").replace("\"", "")
    for data_row in data.split(","):
        # 生成一个有序字典，保存排序结果
        rank_info = OrderedDict()
        row_arr = data_row.split("|")

        if len(row_arr)>16:
            # 获取每个星级的评分
            rank_info['基金代码'] = 'd'+row_arr[0]
            rank_info['基金名称'] = row_arr[1]
            rank_info['截止日期'] = row_arr[3]
            rank_info['单位净值'] = row_arr[4]
            rank_info['日增长率'] = row_arr[5]
            rank_info['近1周'] = row_arr[6]
            rank_info['近1月'] = row_arr[7]
            rank_info['近3月'] = row_arr[8]
            rank_info['近6月'] = row_arr[9]
            rank_info['近1年'] = row_arr[10]
            rank_info['近2年'] = row_arr[11]
            rank_info['近3年'] = row_arr[12]
            rank_info['今年来'] = row_arr[13]
            rank_info['成立来'] = row_arr[14]
            rank_info['起购金额'] = row_arr[-5]
            rank_info['原手续费'] = row_arr[-4]
            rank_info['现手续费'] = row_arr[-3]

            # 保存当前rank信息
            rank_pages_data.append(rank_info)

    return rank_pages_data


def get_rank_data(url, page_index, max_page, fund_type):
    """
    根据起始页码获取当前页面的所有基金情况 (此函数在文件读取模式下不再使用)
    :return:
    """
    # 保持原样，但在 main 函数中不再调用
    try_cnt = 0
    rank_data = []
    # 若当前页其实页码小于总页数 或者 超时3次 则退出
    while page_index<max_page and try_cnt<3:
        # 根据每页数据条数确定起始下标
        new_url = url + '?ft=' + fund_type + '&sc=1n&st=desc&pi=' + str(page_index) + '&pn=100&fl=0&isab=1'
        print('正在爬取第 {0} 页数据：{1}'.format(page_index, new_url))
        # 爬取当前页码的数据
        response = requests.get(url=new_url, headers={'User-Agent': get_ua()}, timeout=10)
        if len(response.text) > 100:
            # 匹配数据并解析
            res_data = re.findall("\[{1}\S+\]{1}", response.text)[0]
            # 解析单页数据
            rank_pages_data = resolve_rank_info(res_data)
            rank_data.extend(rank_page_data for rank_page_data in rank_pages_data)
        else:
            try_cnt += 1
        page_index += 1

        # 随机休眠3-5 秒
        time.sleep(random.randint(3, 5))

    df_rank_data = pd.DataFrame(rank_data)
    return df_rank_data


def resolve_rank_detail_info(fund_code, response):
    """
    解析基金的详细数据
    @param fund_code: 
    @param response:
    @return: 
    """
    rank_detail_info = OrderedDict()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    try:
        rank_detail_info['基金代码'] = 'd'+fund_code
        # 增加 find_all 的索引检查，防止页面结构变化导致 IndexError
        soup_div = soup.find_all('div', class_='bs_gl')
        if not soup_div:
             # 如果找不到关键信息块，返回空，让 try_craw_info 触发重试
            return {} 

        soup_div = soup_div[0]
        rank_detail_info['成立日期'] = soup_div.find_all('label')[0].find_all('span')[0].get_text()
        rank_detail_info['基金经理'] = soup_div.find_all('label')[1].find_all('a')[0].get_text()
        rank_detail_info['类型'] = soup_div.find_all('label')[2].find_all('span')[0].get_text()
        rank_detail_info['管理人'] = soup_div.find_all('label')[3].find_all('a')[0].get_text()
        rank_detail_info['资产规模'] = soup_div.find_all('label')[4].find_all('span')[0].get_text().replace("\r\n", "").replace(" ", "")
    except IndexError:
        print(f"解析 {fund_code} 详细数据时发生索引错误，可能页面结构已变化。")
        return {} # 返回空字典
    except Exception as e:
        print(f"解析 {fund_code} 详细数据时发生未知错误: {e}")
        return {}
    
    return rank_detail_info


def resolve_position_info(fund_code, text):
    """
    解析基金的持仓数据
    @param fund_code:
    @param text:
    @return:
    """
    fund_positions_data = []
    try:
        res_data = re.findall(r'\"(.*)\"', text)[0]
        soup = BeautifulSoup(res_data, 'html.parser')

        update_date = soup.find_all('label', class_='right lab2 xq505')[0].find_all('font')[0].get_text()
        soup_tbody = soup.find_all('table', class_='w782 comm tzxq')
        if not soup_tbody:
            print(f"基金 {fund_code} 无持仓数据或页面结构异常。")
            return [] # 返回空列表

        soup_tbody = soup_tbody[0].find_all('tbody')[0]
        
        for soup_tr in soup_tbody.find_all('tr'):
            postion_info = OrderedDict()
            postion_info['基金代码'] = 'd'+fund_code
            # 使用列表索引获取可能报错，但这里为保持代码一致性，暂不修改
            postion_info['基金截止日期'] = soup.find_all('font', class_='px12')[0].get_text() 

            postion_info['持仓排序'] = soup_tr.find_all('td')[0].get_text()
            postion_info['持仓股票代码'] = 'd'+soup_tr.find_all('td')[1].find_all('a')[0].get_text()
            postion_info['持仓股票名称'] = soup_tr.find_all('td')[2].find_all('a')[0].get_text()
            postion_info['持仓股票最新价'] = soup_tr.find_all('td')[3].find_all('span')[0].get_text()
            postion_info['持仓股票涨跌幅'] = soup_tr.find_all('td')[4].find_all('span')[0].get_text()
            postion_info['持仓股票占比'] = soup_tr.find_all('td')[6].get_text()
            postion_info['持仓股票持股数'] = soup_tr.find_all('td')[7].get_text()
            postion_info['持仓股票持股市值'] = soup_tr.find_all('td')[8].get_text().replace(",", "")
            postion_info['更新日期'] = update_date

            fund_positions_data.append(postion_info)
    
    except IndexError as e:
        print(f"解析 {fund_code} 持仓数据时发生索引错误，可能数据格式异常或页面结构变化: {e}")
        return [] # 返回空列表
    except Exception as e:
        print(f"解析 {fund_code} 持仓数据时发生未知错误: {e}")
        return []

    return fund_positions_data


def try_craw_info(fund_code, try_cnt):
    """
    基金详细数据和持仓数据的重试爬取函数。
    @param fund_code: 带有 'd' 前缀的基金代码字符串
    @return: (rank_detail_info, fund_positions_data) 或 (None, None)
    """
    if try_cnt > 5:
        print(f"基金 {fund_code[1:]} 最终爬取失败，达到最大重试次数。")
        return None, None
        
    try:
        # 详细数据爬取
        fund_pure_code = fund_code[1:] # 去掉 'd'
        position_title_url = f"http://fundf10.eastmoney.com/ccmx_{fund_pure_code}.html"
        print(f'第 {try_cnt} 次尝试，正在爬取基金 {fund_pure_code} 的详细数据中...')
        response_title = requests.get(url=position_title_url, headers={'User-Agent': get_ua()}, timeout=10)
        response_title.raise_for_status() # 检查HTTP错误
        rank_detail_info = resolve_rank_detail_info(fund_pure_code, response_title)
        
        # 如果解析失败（返回空字典），则视为失败，触发重试
        if not rank_detail_info:
            raise ValueError("详细数据解析失败或返回空")

        # 持仓数据爬取
        position_data_url = f"http://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=jjcc&code={fund_pure_code}&topline=10&year=&month=&rt={random.uniform(0, 1)}"
        print(f'第 {try_cnt} 次尝试，正在爬取基金 {fund_pure_code} 的持仓情况中...')
        response_data = requests.get(url=position_data_url, headers={'User-Agent': get_ua()}, timeout=10)
        response_data.raise_for_status() # 检查HTTP错误
        fund_positions_data = resolve_position_info(fund_pure_code, response_data.text)
        
        # 持仓数据可以为空（没有持仓），因此不检查是否为空
        
        time.sleep(random.randint(2, 4))
        return rank_detail_info, fund_positions_data
        
    except (RequestException, ValueError, IndexError, Exception) as e:
        # 捕获网络错误、解析错误及其他未预料的错误
        print(f"❌ 基金 {fund_code[1:]} 数据爬取失败: {type(e).__name__} - {e}，将在 {2*try_cnt} 到 {4*try_cnt} 秒后重试。")
        time.sleep(random.randint(2*try_cnt, 4*try_cnt))
        # 递归调用重试
        return try_craw_info(fund_code, try_cnt + 1)


def get_position_data(data, rank):
    """
    根据给定的 DataFrame (或本例中的基金代码列表) 获取基金详细信息和持仓数据
    NOTE: 在读取文件模式下，'data' 参数实际为 fund_codes 列表。
    """
    # -----------------------------------------------------
    # 注释掉原有的排名筛选逻辑，因为现在使用文件代码列表
    # """筛选Top数据"""
    # data = data.replace('', np.NaN, regex=True)
    # data_notna = data.dropna(subset=['近2年'])
    # data_notna['近2年'] = data_notna['近2年'].astype(float)
    # data_sort = data_notna.sort_values(by='近2年', ascending=False)
    # data_sort.reset_index(inplace=True)
    # data_rank = data_sort.loc[0:rank-1, :]
    # -----------------------------------------------------
    
    # 传入的 data 是基金代码列表 (已在 main 函数中处理为带 'd' 的列表)
    fund_codes_to_craw = data 

    # 爬取每个基金的数据
    rank_detail_data = []
    position_data = []
    error_funds_list = []
    
    for row_index, fund_code in enumerate(fund_codes_to_craw):
        # fund_code 已经是 'd'+code 格式
        fund_pure_code = fund_code[1:] 
        
        try:
            '''爬取页面，获得该基金的详细数据'''
            position_title_url = f"http://fundf10.eastmoney.com/ccmx_{fund_pure_code}.html"
            print('正在爬取第 {0}/{1} 个基金 {2} 的详细数据中...'.format(row_index+1, len(fund_codes_to_craw), fund_pure_code))
            response_title = requests.get(url=position_title_url, headers={'User-Agent': get_ua()}, timeout=10)
            response_title.raise_for_status()
            # 解析基金的详细数据
            rank_detail_info = resolve_rank_detail_info(fund_pure_code, response_title)
            
            # 如果解析失败，抛出错误，进入 except 块
            if not rank_detail_info:
                raise ValueError("详细数据解析失败")
                
            # 保存数据
            rank_detail_data.append(rank_detail_info)

            """爬取页面，获取该基金的持仓数据"""
            position_data_url = f"http://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=jjcc&code={fund_pure_code}&topline=10&year=&month=&rt={random.uniform(0, 1)}"
            print('正在爬取第 {0}/{1} 个基金 {2} 的持仓情况中...'.format(row_index + 1, len(fund_codes_to_craw), fund_pure_code))
            # 解析基金的持仓情况
            response_data = requests.get(url=position_data_url, headers={'User-Agent': get_ua()}, timeout=10)
            response_data.raise_for_status()
            fund_positions_data = resolve_position_info(fund_pure_code, response_data.text)

            # 保存数据
            position_data.extend(fund_positions_data)
        
        except (RequestException, ValueError, IndexError, Exception) as e:
            error_funds_list.append(fund_code)
            print("{0} 数据爬取失败: {1}，稍后会进行重试，请注意！".format(fund_pure_code, type(e).__name__))
            
        # 随机休眠2-4 秒
        time.sleep(random.randint(2, 4))

    """爬取失败的进行重试"""
    if error_funds_list:
        print("\n--- 开始重试爬取失败的基金数据 ---")
    for fund_code in error_funds_list:
        rank_detail_data_try, position_data_try = try_craw_info(fund_code, 1)
        
        # 检查重试是否成功（返回的不是 None 且详细信息不是空字典）
        if rank_detail_data_try is not None and position_detail_data_try:
            # 保存重试成功的数据
            rank_detail_data.append(rank_detail_data_try)
            position_data.extend(position_data_try)
        else:
            print(f"⚠️ 基金 {fund_code[1:]} 重试后仍然失败，已跳过。")


    df_rank_detail_data = pd.DataFrame(rank_detail_data)
    df_position_data = pd.DataFrame(position_data)

    return df_rank_detail_data, df_position_data


def load_fund_codes_from_file(file_path):
    """
    读取文件中的基金代码，并处理为爬虫所需的格式。
    """
    try:
        # [cite_start]读取文件内容 [cite: 1]
        with open(file_path, 'r', encoding='utf-8') as f:
            # [cite_start]过滤掉空行和空白，并去除每行首尾空白 [cite: 1]
            codes = [line.strip() for line in f if line.strip()] 
            # 添加 'd' 前缀以符合原代码逻辑
            return ['d' + code for code in codes]
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        return []

if __name__ == '__main__':
    # 1. 定义目标文件路径
    file_path = 'C类.txt' 
    
    # 2. 读取基金代码列表
    fund_codes_list = load_fund_codes_from_file(file_path)
    
    if not fund_codes_list:
        print("基金代码列表为空，程序退出。")
        sys.exit(1)

    print(f"成功读取 {len(fund_codes_list)} 个基金代码。")
    
    # --------------------------------------------------------------------------------
    # 3. **跳过原有的排名爬取逻辑**
    # url = 'https://fundapi.eastmoney.com/fundtradenew.aspx'
    # page_index = 1
    # max_page = 100
    # fund_type = 'gp'
    # df_rank_data = get_rank_data(url, page_index, max_page, fund_type)
    # max_rank = 300 if int(len(df_rank_data)/100) > 3 else 100
    # --------------------------------------------------------------------------------

    # 4. **直接调用 get_position_data**
    # 传入 fund_codes_list 代替原有的 df_rank_data
    # 基金代码文件没有收益排名数据，因此 df_rank_data 不存在，我们将跳过收益数据的处理。
    print("开始爬取指定基金代码的详细信息和持仓数据...")
    df_rank_detail_data, df_position_data = get_position_data(fund_codes_list, 0) # rank 参数不再重要，设为 0

    # 5. **构建最终的基金收益详情数据（仅包含详细信息）**
    # 由于没有爬取收益排名，我们不能进行 merge。我们只保存爬取到的详细信息和持仓信息。
    
    # 构造一个空的df_rank_data来表示没有收益数据
    # 为了简化，我们只使用 df_rank_detail_data 进行保存，并命名文件以区分。
    df_rank_detail_data = df_rank_detail_data.fillna('--')
    
    """保存数据"""
    output_filename_detail = 'file/C类基金_详情数据.csv'
    output_filename_position = 'file/C类基金_持仓数据.csv'
    
    if not df_rank_detail_data.empty:
        df_rank_detail_data.to_csv(output_filename_detail, encoding='gbk', index=False)
        print(f"✅ 基金详情数据已保存到 {output_filename_detail}")
    else:
        print("⚠️ 基金详情数据为空，未生成详情CSV文件。")

    if not df_position_data.empty:
        df_position_data.to_csv(output_filename_position, encoding='gbk', index=False)
        print(f"✅ 基金持仓数据已保存到 {output_filename_position}")
    else:
        print("⚠️ 基金持仓数据为空，未生成持仓CSV文件。")
