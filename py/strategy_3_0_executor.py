import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime

# --- 辅助函数：加载数据 ---
def load_config(config_path='holdings_config.yaml'):
    """加载配置文件并返回持仓数据。"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            holdings_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：{config_path} 文件未找到。")
        return {}
    
    return {k: v for k, v in holdings_config.items() if k != 'parameters'}

def load_fund_data(code, data_dir='fund_data/'):
    """加载基金净值数据。"""
    fund_file = os.path.join(data_dir, f"{code}.csv")
    if os.path.exists(fund_file):
        # 假设 fund_data 目录存在且文件包含 'date' 和 'net_value' 列
        fund_df = pd.read_csv(fund_file, parse_dates=['date'])
        fund_df = fund_df.sort_values('date').reset_index(drop=True)
        return fund_df
    else:
        # 使用模拟数据，用于演示目的
        print(f"警告: 基金数据文件 {fund_file} 未找到，使用模拟数据。")
        dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
        # 模拟净值从 V0 1.8158 开始下跌，确保 I 级二批可能被触发
        net_values = np.linspace(1.8158, 1.7000, 10) 
        return pd.DataFrame({'date': dates, 'net_value': net_values})

# --- 策略 3.0 核心状态和纪律 ---
def get_strategy_3_0_config():
    """定义策略 3.0 的所有数学参数和资金状态。"""
    
    # 策略 3.0 参数
    strategy_config = {
        'total_capital': 15000,
        'initial_nav_V0': 1.8158, # I 级首批买入净值 V0
        'target_code': '009645',
        
        # ⚠️ 关键假设：由于缺乏历史交易记录，我们假设：
        # 1. I 级首批 (2000元 @ 1.8158) 已完成。
        # 2. C_avg 和 C_current 此时都等于 V0。
        'C_avg': 1.8158,       # 初始固定锚定成本 (将会在 I 级二批后锁定)
        'C_current': 1.8158,   # 当前浮动平均成本
        'V_stop_profit_nav': 0.0, # I 档止盈时的净值 (0.0 表示 I 档未触发)
        
        # 资金配置状态
        'capital_status': {
            'I_level_spent': 2000,
            'current_total_invested': 2000,
            'remaining_I_IV': 4000, # 6000 - 2000
            'remaining_V_VII': 9000,
        },
        
        # 严格的加仓/止损/止盈触发点 (数值严谨版)
        'trigger_points': {
            # 加仓
            'I_2_add_nav': 1.8158 * (1 - 0.05), # 1.72501
            'II_add_nav_ratio': 0.05,
            'III_add_nav_ratio': 0.10,
            'IV_add_nav_ratio': 0.15,
            # 止盈
            'I_sell_ratio': 1.055, # C_current * 1.055
            'II_sell_ratio': 1.10, # V_止盈 * 1.10
            # 止损/防御
            'V_stop_loss_ratio': 0.20, # 暴跌止损线
            'V_defense_ratio': 0.30,
            'VI_defense_ratio': 0.50,
            'VII_defense_ratio': 0.80,
        },
        
        'add_capital_map': {
            'I_2': 1000,
            'II': 1000,
            'III': 1000,
            'IV': 1000,
            'V': 4000,
            'VI': 4000,
            'VII': 1000,
        }
    }
    
    return strategy_config

# --- 核心纪律执行函数 ---
def execute_strategy_3_0(fund_data, config):
    """
    严格执行策略 3.0 的加仓和止盈/止损纪律。
    """
    if fund_data.empty:
         return {'action': '数据不足', 'details': '无法获取净值数据', 'capital_change': 0}
         
    latest_nav = fund_data.iloc[-1]['net_value']
    
    # 从配置中获取关键参数
    C_avg = config['C_avg'] 
    C_current = config['C_current']
    V_stop_profit_nav = config['V_stop_profit_nav']
    
    triggers = config['trigger_points']
    capital_status = config['capital_status']
    add_capital_map = config['add_capital_map']
    
    decision = {'action': '等待', 'details': f'最新净值: {latest_nav:.4f}', 'capital_change': 0}
    
    # --- 1. 严格的止盈检查 (【脱】) (止盈优先级高于加仓) ---
    
    # 检查 I 档是否已触发过
    is_I_triggered = V_stop_profit_nav > 0.0 

    I_sell_nav = C_current * triggers['I_sell_ratio']
    
    # I 档止盈检查（零成本）
    if latest_nav >= I_sell_nav and not is_I_triggered: 
        decision['action'] = 'I 档止盈 (赎回本金份额)'
        decision['details'] = f'【零成本锁定】触发：净值 {latest_nav:.4f} >= C_current * 1.055 ({I_sell_nav:.4f})。操作：赎回投入的全部本金所对应的份额。'
        decision['capital_change'] = -capital_status['current_total_invested'] # 理论赎回金额
        # 此时应更新 V_stop_profit_nav = latest_nav，并进入 II 档监控
        return decision

    # II 档止盈检查（利润锁定）
    if is_I_triggered:
        II_sell_nav = V_stop_profit_nav * triggers['II_sell_ratio']
        if latest_nav >= II_sell_nav: 
            decision['action'] = 'II 档止盈 (卖出全部零成本仓位)'
            decision['details'] = f'【利润锁定】触发：净值 {latest_nav:.4f} >= V_止盈 * 1.10 ({II_sell_nav:.4f})。操作：卖出剩余所有零成本仓位。'
            decision['capital_change'] = -1 # 卖出所有剩余份额（假设为 1，实际应是份额数量）
            return decision

    # --- 2. 严格的加仓/黑天鹅防御检查 (【攻】和【守】) ---
    
    # 2.1 I 级二批加仓检查
    I_2_add_nav = triggers['I_2_add_nav']
    if latest_nav <= I_2_add_nav and capital_status['remaining_I_IV'] >= add_capital_map['I_2'] and capital_status['I_level_spent'] == 2000:
        decision['action'] = '加仓 (I 级二批)'
        decision['details'] = f'【I 级二批】触发：净值 {latest_nav:.4f} <= V0跌5% ({I_2_add_nav:.4f})。操作：投入 {add_capital_map["I_2"]} 元。'
        decision['capital_change'] = add_capital_map['I_2']
        return decision

    # 2.2 常规加仓 II, III, IV 级检查（基于 C_avg 累计跌幅）
    # 仅当 I 级 3000 元投入完成后（即 I_level_spent >= 3000）才启用 II-IV 级判断
    if capital_status['I_level_spent'] >= 3000: 
        current_decline_from_C_avg = (C_avg - latest_nav) / C_avg
        
        # II-IV 级按最高跌幅优先检查
        if current_decline_from_C_avg >= triggers['IV_add_nav_ratio'] and capital_status['remaining_I_IV'] >= add_capital_map['IV']:
            decision['action'] = '加仓 (IV 级)'
            decision['details'] = f'【IV 级】触发：C_avg累计跌幅 >= 15% ({current_decline_from_C_avg:.2%})。操作：投入 {add_capital_map["IV"]} 元。'
            decision['capital_change'] = add_capital_map['IV']
            return decision
        elif current_decline_from_C_avg >= triggers['III_add_nav_ratio'] and capital_status['remaining_I_IV'] >= add_capital_map['III']:
            decision['action'] = '加仓 (III 级)'
            decision['details'] = f'【III 级】触发：C_avg累计跌幅 >= 10% ({current_decline_from_C_avg:.2%})。操作：投入 {add_capital_map["III"]} 元。'
            decision['capital_change'] = add_capital_map['III']
            return decision
        elif current_decline_from_C_avg >= triggers['II_add_nav_ratio'] and capital_status['remaining_I_IV'] >= add_capital_map['II']:
            decision['action'] = '加仓 (II 级)'
            decision['details'] = f'【II 级】触发：C_avg累计跌幅 >= 5% ({current_decline_from_C_avg:.2%})。操作：投入 {add_capital_map["II"]} 元。'
            decision['capital_change'] = add_capital_map['II']
            return decision
        
        # 2.3 黑天鹅防御 V, VI, VII 级检查
        if current_decline_from_C_avg >= triggers['V_stop_loss_ratio']: # 满足暴跌止损线 > 20%
            if current_decline_from_C_avg >= triggers['VII_defense_ratio'] and capital_status['remaining_V_VII'] >= add_capital_map['VII']:
                decision['action'] = '黑天鹅防御买入 (VII 级)'
                decision['details'] = f'【VII 级】触发：C_avg累计跌幅 >= 80% ({current_decline_from_C_avg:.2%})。操作：投入 {add_capital_map["VII"]} 元。'
                decision['capital_change'] = add_capital_map['VII']
                return decision
            elif current_decline_from_C_avg >= triggers['VI_defense_ratio'] and capital_status['remaining_V_VII'] >= add_capital_map['VI']:
                decision['action'] = '黑天鹅防御买入 (VI 级)'
                decision['details'] = f'【VI 级】触发：C_avg累计跌幅 >= 50% ({current_decline_from_C_avg:.2%})。操作：投入 {add_capital_map["VI"]} 元。'
                decision['capital_change'] = add_capital_map['VI']
                return decision
            elif current_decline_from_C_avg >= triggers['V_defense_ratio'] and capital_status['remaining_V_VII'] >= add_capital_map['V']:
                decision['action'] = '黑天鹅防御买入 (V 级)'
                decision['details'] = f'【V 级】触发：C_avg累计跌幅 >= 30% ({current_decline_from_C_avg:.2%})。操作：投入 {add_capital_map["V"]} 元。'
                decision['capital_change'] = add_capital_map['V']
                return decision
            elif current_decline_from_C_avg >= triggers['V_stop_loss_ratio']:
                 decision['action'] = '暴跌止损线 (暂停买入)'
                 decision['details'] = f'【暴跌止损线】触发：C_avg累计跌幅 >= 20% ({current_decline_from_C_avg:.2%})。操作：暂停常规买入，启用防御池监控。'
                 return decision

    # 3. 默认：等待
    return decision

# --- 主执行逻辑 ---
if __name__ == '__main__':
    # 1. 定义策略 3.0 的配置
    strategy_config = get_strategy_3_0_config()
    code = strategy_config['target_code']

    # 2. 加载基金数据 (假设 fund_data 目录下有 009645.csv)
    fund_data = load_fund_data(code)
    
    if fund_data.empty or len(fund_data) < 1:
        print(f"--- 🚀 策略 3.0 极简·纯净值纪律执行失败 ---")
        print(f"错误：无法获取 {code} 净值数据。")
    else:
        # 3. 执行策略 3.0
        decision_3_0 = execute_strategy_3_0(fund_data, strategy_config)
        
        # 4. 输出结果 (打印到 stdout，将被工作流捕获并存档为 TXT)
        print("--- 🚀 策略 3.0 极简·纯净值纪律（数学严谨版）执行结果 ---")
        print(f"**分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"基金代码: {code}")
        print(f"最新净值: {fund_data.iloc[-1]['net_value']:.4f}")
        print(f"当前锚定成本 C_avg (固定): {strategy_config['C_avg']:.4f}")
        print(f"当前平均成本 C_current (浮动): {strategy_config['C_current']:.4f}")
        print("-" * 30)
        print(f"**操作行动**: {decision_3_0['action']}")
        print(f"**详情**: {decision_3_0['details']}")
        print(f"**涉及资金**: {decision_3_0['capital_change']} 元")
        print("---")
        if decision_3_0['action'] == '等待':
            print("结论：根据策略 3.0 的纯净值纪律，继续耐心等待，不采取行动。")
