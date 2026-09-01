import requests
import pandas as pd
import re
from io import StringIO
import threading
from functools import wraps

import time

def new_thread(func):
    @wraps(func)
    def inner(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return inner

default_source_url = "https://fundf10.eastmoney.com/F10DataApi.aspx"

def update_fund_list():
    import json
    response = requests.get('https://fund.eastmoney.com/js/fundcode_search.js')
    if response.status_code == 200:
        response.encoding = 'utf-8'
        # 提取数组部分
        content = response.text
        start = content.find('[')
        end = content.rfind(']') + 1
        array_text = content[start:end]
        # 将 JS 数组转换为 Python 列表
        data = json.loads(array_text)
        df = pd.DataFrame(data, columns=['code', 'name1', 'name2', 'type', 'name3'])
        df.to_csv('./fund_list.csv')


class FundData:
    def __init__(self, code, sdata, edata, source=default_source_url):
        self.source = source
        self.params = {
            "type": "lsjz",
            "code": code,
            "sdate": sdata,
            "edate": edata,
            "per": 20,
        }
        fund_list = pd.read_csv('./fund_list.csv', dtype={'code':str})
        detail = fund_list[fund_list['code']==code]
        self._data_ = {'info': self.params, 'detail': detail}
        self._request_page_(1).join()
        self.data = self._get_data_(self._data_[1])

    @new_thread
    def _request_page_(self, page, num_retry=1):
        params = self.params.copy()
        params['page'] = page
        response = requests.get(self.source, params=params)
        if response.status_code == 200:
            self._data_[(page)] = response
        else:
            while num_retry <= 10:
                print(f"page: {page}. 请求失败，状态码：{response.status_code}")
                print(f'retry {num_retry}...')
                time.sleep(1)
                self._data_[(page)] = response
                num_retry += 1

    def _get_data_(self, response):
        # 使用正则表达式提取 content 字段的内容
        total_pages_match = re.search(r'pages:(\d+)', response.text)
        curpage_match = re.search(r'curpage:(\d+)', response.text)
        total_pages = int(total_pages_match.group(1))
        curpage = int(curpage_match.group(1))
        requests_list = []
        df_list = []
        while curpage <= total_pages:
            response = self._request_page_(curpage)
            requests_list.append(response)
            curpage += 1
        for i in range(len(requests_list)):
            print(f'{i+1}/{total_pages}\r', flush=True, end='')
            requests_list[i].join()
            match = re.search(r'content:"(.*?)"', self._data_[i+1].text, re.S)
            if match:
                content = match.group(1)
                # 将 content 中的 \r\n 替换为换行符
                content = content.replace("\\r\\n", "\n")
                # 使用 pandas 读取表格数据
                df = pd.read_html(StringIO(content))[0]
                # 打印数据框
                df_list.append(df)
            else:
                print("未找到 content 字段")
            curpage += 1
        print(f'{i+1}/{total_pages}')
        df = pd.concat(df_list, ignore_index=True)
        df['净值日期'] = pd.to_datetime(df['净值日期'])
        return df

if __name__ == '__main__':
    fund_data = FundData('019548', '2024-01-01', '2025-01-01')
    update_fund_list()