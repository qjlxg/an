
# !/usr/bin/python
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

# 使用方法
def usage():
    print('fund-rank.py usage:')
    print('\tpython fund.py start-date end-date fund-code=none\n')
    print('\tdate format ****-**-**')
    print('\t\tstart-date must before end-date')
    print('\tfund-code default none')
    print('\t\tif not input, get top 20 funds from all more than 6400 funds')
    print('\t\telse get that fund\'s rate of rise\n')
    print('\teg:\tpython fund-rank.py 2017-03-01 2017-03-25')
    print('\teg:\tpython fund-rank.py 2017-03-01 2017-03-25 377240')

# 获取某一基金在某一日的累计净值数据
def get_jingzhi(strfundcode, strdate):
    try:
        url = 'http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code=' + \
              strfundcode + '&page=1&per=20&sdate=' + strdate + '&edate=' + strdate
        response = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        print(e)
        return '-1'
    except Exception as e:
        print(e)
        return '-1'

    json_fund_value = response.read().decode('utf-8')
    tr_re = re.compile(r'<tr>(.*?)</tr>')
    item_re = re.compile(r'''<td>(\d{4}-\d{2}-\d{2})</td><td.*?>(.*?)</td><td.*?>(.*?)</td><td.*?>(.*?)</td><td.*?>(.*?)</td><td.*?>(.*?)</td><td.*?></td>''', re.X)

    jingzhi = '-1'
    for line in tr_re.findall(json_fund_value):
        match = item_re.match(line)
        if match:
            entry = match.groups()
            jingzhi1 = entry[1]
            jingzhi2 = entry[2]
            
            if jingzhi2.strip() == '':
                jingzhi = '-1'
            elif jingzhi2.find('%') > -1:
                jingzhi = '-1'
            elif float(jingzhi1) > float(jingzhi2):
                jingzhi = entry[1]
            else:
                jingzhi = entry[2]

    return jingzhi
    
# --- 新增的线程工作函数 ---
def worker(q, strsdate, stredate, result_queue):
    while not q.empty():
        fund = q.get()
        strfundcode = fund[0]
        
        # 获取净值
        jingzhimin = get_jingzhi(strfundcode, strsdate)
        jingzhimax = get_jingzhi(strfundcode, stredate)

        if jingzhimin == '-1' or jingzhimax == '-1' or jingzhimin.strip() == '' or jingzhimax.strip() == '':
            jingzhimin = '0'
            jingzhimax = '0'
            jingzhidif = 0
            jingzhirise = 0
        elif jingzhimin.find('%') > -1 or jingzhimax.find('%') > -1:
            jingzhidif = 0
            jingzhirise = 0
        else:
            jingzhidif = float('%.4f' %(float(jingzhimax) - float(jingzhimin)))
            jingzhirise = float('%.2f' %(jingzhidif * 100 / float(jingzhimin)))
        
        fund.append(jingzhimin)
        fund.append(jingzhimax)
        fund.append(jingzhidif)
        fund.append(jingzhirise)
        
        result_queue.put(fund) # 将处理好的基金数据放入结果队列
        print('process fund:\t' + fund[0] + '\t' + fund[2])
        q.task_done()

# --- 主函数 ---
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

    if len(sys.argv) == 4:
        strfundcode = sys.argv[3]
        jingzhimin = get_jingzhi(strfundcode, strsdate)
        jingzhimax = get_jingzhi(strfundcode, stredate)
        
        if jingzhimin == '-1' or jingzhimax == '-1' or jingzhimin.strip() == '' or jingzhimax.strip() == '':
            print('maybe date input error!\n')
            usage()
            sys.exit(1)
        
        jingzhidif = float(jingzhimax) - float(jingzhimin)
        jingzhirise = float('%.2f' %(jingzhidif * 100 / float(jingzhimin)))
    
        print('fund:' + strfundcode + '\n')
        print(strsdate + '\t' + stredate + '\t净增长' + '\t' + '增长率')
        print(jingzhimin + '\t\t' + jingzhimax + '\t\t' + str(jingzhidif) + '\t' + str(jingzhirise) + '%')
        sys.exit(0)
        
    fundlist_files = glob.glob('fundlist-*.txt')
    if (len(fundlist_files) > 0):
        file_object = open(fundlist_files[0], 'r')
        try:
            all_funds_txt = file_object.read()
        finally:
            file_object.close()
    else:
        response_all_funds = urllib.request.urlopen('http://fund.eastmoney.com/js/fundcode_search.js')
        all_funds_txt = response_all_funds.read().decode('utf-8')
        file_object = open('fundlist-' + strtoday + '.txt', 'w')
        try:
            file_object.write(all_funds_txt)
        finally:
            file_object.close()
    
    all_funds_txt = all_funds_txt[all_funds_txt.find('=')+2:all_funds_txt.rfind(';')]
    all_funds_list = json.loads(all_funds_txt)
    
    print('筛选中，只处理场外C类基金...')
    c_funds_list = []
    for fund in all_funds_list:
        if fund[0].endswith('C') or 'C' in fund[2] or ('C' in fund[3] and '场外' in fund[3]):
            c_funds_list.append(fund)
    
    all_funds_list = c_funds_list
    print('筛选后，场外C类基金数量：' + str(len(all_funds_list)))
     
    print('start:')
    print(datetime.datetime.now())
    print('funds sum:' + str(len(all_funds_list)))
    
    # --- 并行处理部分开始 ---
    task_queue = queue.Queue()
    result_queue = queue.Queue()

    # 将所有基金放入任务队列
    for fund in all_funds_list:
        task_queue.put(fund)

    # 创建并启动线程
    threads = []
    for i in range(10): # 使用10个线程，可以根据网络情况调整
        t = threading.Thread(target=worker, args=(task_queue, strsdate, stredate, result_queue))
        t.daemon = True # 设置为守护线程，主线程退出时自动结束
        t.start()
        threads.append(t)

    # 等待所有任务完成
    task_queue.join()

    # 收集所有结果
    all_funds_list = []
    while not result_queue.empty():
        all_funds_list.append(result_queue.get())
    # --- 并行处理部分结束 ---

    fileobject = open('result_' + strsdate + '_' + stredate + '_C类.txt', 'w')
    
    all_funds_list.sort(key=lambda fund: fund[8], reverse=True)
    strhead = '排序\t' + '编码\t\t' + '名称\t\t' + '类型\t\t' + \
    strsdate + '\t' + stredate + '\t' + '净增长' + '\t' + '增长率' + '\n'
    print(strhead)
    fileobject.write(strhead)
    
    for index in range(len(all_funds_list)):
        strcontent = str(index+1) + '\t' + all_funds_list[index][0] + '\t' + all_funds_list[index][2] + \
        '\t\t' + all_funds_list[index][3] + '\t\t' + all_funds_list[index][5] + '\t\t' + \
        all_funds_list[index][6] + '\t\t' + str(all_funds_list[index][7]) + '\t' + str(all_funds_list[index][8]) + '%\n'
        print(strcontent)
        fileobject.write(strcontent)
        
        if index >= gettopnum:
            break
        
    fileobject.close()
    
    print('end:')
    print(datetime.datetime.now())
    
    sys.exit(0)
    
if __name__ == "__main__":
    main(sys.argv)
