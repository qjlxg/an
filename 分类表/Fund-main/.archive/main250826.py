import numpy as np
import pandas as pd
from data_source import FundData


class invest_method:
    def __init__(self, data:pd.DataFrame):
        self.data = data.sort_index(ascending=False)
        self.current_cash = 100.0
        self.current_share = 0.0

    def input(self, cash, unit_net_value):
        if self.current_cash >= cash:
            self.current_cash -= cash
            self.current_share += cash/unit_net_value
        else:
            # print('Out of cash.)
            pass

    def output(self, share, unit_net_value):
        if self.current_share >= share:
            self.current_share -= share
            self.current_cash += share*unit_net_value
        else:
            # print('Out of share.')
            pass

    def run(self):
        # all in
        self.states = []
        self.input(self.current_cash, self.data.values[0][1])
        self.output(self.current_share, self.data.values[-1][1])
        self.states.append({'date':self.data.values[-1][0], 'cash': self.current_cash, 'share': self.current_share, 'total_value': self.current_cash + self.current_share*self.data.values[-1][1], 'unit_net_value': self.data.values[-1][1]})


class All_In(invest_method):
    pass


class Target_Smart(invest_method):
    def run(self, target_profit=0.05, day_input=1.0):
        cost = 0.0
        self.states = []
        for day in self.data.values:
            self.input(day_input, day[1])
            cost += day_input
            if self.current_share*day[1]/cost >= 1+target_profit:
                self.output(self.current_share, day[1])
                cost = 0.0
            self.states.append({'date':day[0], 'cash': self.current_cash, 'share': self.current_share, 'total_value': self.current_cash + self.current_share*day[1], 'unit_net_value': day[1]})


class Period(invest_method):
    def run(self, interval=5, day_input=2.0):
        cost = 0.0
        self.states = []
        for i,day in enumerate(self.data.values):
            if i%5 == 0:
                self.input(day_input, day[1])
                cost += day_input
            self.states.append({'date':day[0], 'cash': self.current_cash, 'share': self.current_share, 'total_value': self.current_cash + self.current_share*day[1], 'unit_net_value': day[1]})

fund_code = '006479'
fund_data = FundData(fund_code, '2021-01-01', '2025-01-01')
fund_data_set = fund_data.data
fund_detail = fund_data._data_['detail']
print(fund_detail)
start_time = np.datetime64('2021-01-01')
time_series = np.arange(start_time, start_time+np.timedelta64(365*4, 'D'), np.timedelta64(15, 'D'))
res = []
for start_time in time_series:
    interval = 1 # year
    end_time = start_time + np.timedelta64(365, 'D')
    fund_data = fund_data_set[(fund_data_set['净值日期']>=start_time) & (fund_data_set['净值日期']<=end_time)]
    method1 = Target_Smart(fund_data)
    method1.run()
    method2 = All_In(fund_data)
    method2.run()
    method3 = Period(fund_data)
    method3.run()
    res.append([method1.states[-1]['total_value'],method2.states[-1]['total_value'],method3.states[-1]['total_value']])

import matplotlib.pyplot as plt
plt.figure(dpi=300, figsize=(10,6))
plt.title(f'{fund_code} {fund_detail["name3"].values[0]}')
plt.plot(time_series, res, label=['Target Smart', 'All In', 'Period'])
plt.legend()
plt.savefig(f'./res_{fund_code}.png')