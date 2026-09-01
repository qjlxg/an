# sell_decision.py - 5.0 策略卖出决策模块 (完整版)
# ==============================================================================



# 导入必要的库
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 策略参数设置 ---
# 卖出规则的最低盈利门槛
MIN_PROFIT_FOR_T1 = 5.5  # 达到此盈利百分比触发T1止盈
# 最短持有天数（避免超短期投机）
MIN_HOLDING_DAYS = 7
# 移动止盈回撤百分比
TRAILING_STOP_PERCENT = 0.08  # 从最高点回撤 8% 触发移动止盈
# --------------------

def decide_sell(code, holding, full_fund_data, big_market_latest, big_market_data):
    """
    基于复合指标和动态风控的卖出决策函数。

    参数:
    code (str): 基金代码
    holding (dict): 持仓信息，包含 'cost_nav' (成本净值), 'holding_days', 'max_nav' (持仓期最高净值)
    full_fund_data (pd.DataFrame): 完整的基金历史数据
    big_market_latest (dict): 大盘最新数据（用于环境判断）
    big_market_data (pd.DataFrame): 大盘历史数据（用于计算均线）

    返回:
    dict: 包含 'code', 'decision', 'target_nav', 'market_mood' 等信息的决策结果
    """
    
    # 确保数据充足
    if full_fund_data.shape[0] < 50:
        return {'code': code, 'decision': '持仓', 'target_nav': 0, 'market_mood': '数据不足'}

    # 提取最新数据和指标
    fund_latest = full_fund_data.iloc[-1].to_dict()
    latest_net_value = fund_latest['net_value']
    
    # 成本与盈亏计算
    cost_nav = holding['cost_nav']
    profit_rate = (latest_net_value / cost_nav - 1) * 100
    
    # 指标提取
    rsi = fund_latest.get('rsi', 50)
    macd_signal = fund_latest.get('macd_signal', '观察') # '金叉', '死叉', '观察'
    bb_pos = fund_latest.get('bb_pos', '中轨') # '上轨上方', '上轨附近', '中轨', '下轨附近', '下轨下方'
    fund_ma50 = fund_latest.get('ma50', 0)
    
    sell_reasons = []
    
    # -----------------------------------------------------
    # --- ！！！优化点 1：市场环境判断 (前置过滤器) ！！！---
    # 采用大盘MA50作为市场水温判断的基准
    big_nav = big_market_latest.get('net_value', 1.0)
    big_ma50 = big_market_latest.get('ma50', 1.0) 
    
    market_mood = '震荡市'
    # 简易判断：大盘高于MA50且RSI偏强（>50）视为强势
    if big_nav > big_ma50 and big_market_latest.get('rsi', 50) > 50:
        market_mood = '牛市/强势市'
    # 大盘低于MA50且RSI偏弱（<50）视为弱势
    elif big_nav < big_ma50 and big_market_latest.get('rsi', 50) < 50:
        market_mood = '熊市/弱势市'
    # -----------------------------------------------------
    
    # -----------------------------------------------------
    # --- 1. 绝对止损与分级止损 (最高优先级：风控) ---

    # A. 灾难性止损（绝对清仓）
    if profit_rate < -20:
        decision = '因亏损>20% 强制清仓 100%'
        sell_reasons.append('绝对止损（亏损>20%）')
        return { 'code': code, 'decision': decision, 'target_nav': latest_net_value, 'market_mood': market_mood }

    # B. 强制减仓 50%
    elif profit_rate < -15:
        decision = '因亏损>15% 强制减仓 50%'
        sell_reasons.append('分级止损（亏损>15%）')
        return { 'code': code, 'decision': decision, 'target_nav': latest_net_value, 'market_mood': market_mood }

    # C. ！！！修正和优化点 2：分级止损 (强制减仓 30% + 暂停定投) ！！！
    elif profit_rate < -10:
        # 即使刚买入，达到此线也必须执行
        decision = '因亏损>10% 强制减仓 30% + 暂停定投'
        sell_reasons.append('分级止损（亏损>10%）')
        return { 'code': code, 'decision': decision, 'target_nav': latest_net_value, 'market_mood': market_mood }
    
    # -----------------------------------------------------
    # --- 2. 持有天数限制 (仅限制短期微亏或盈利卖出，不限制重大止损) ---
    if holding['holding_days'] < MIN_HOLDING_DAYS and profit_rate < MIN_PROFIT_FOR_T1:
        # 如果未满足T1止盈，且持有天数不足，则强制继续持有（风控线已在上一步检查）
        return {'code': code, 'decision': '持仓 - 持有天数不足', 'target_nav': 0, 'market_mood': market_mood}

    # -----------------------------------------------------
    # --- 3. ！！！优化点 2：技术性止损（MA50跌破）！！！---
    # 在轻度亏损或微盈状态下，如果跌破MA50，则进行技术风控减仓
    # MA50是长期趋势的强支撑，跌破意味着长期持有逻辑可能被打破
    if latest_net_value < fund_ma50 and profit_rate > -5.0: # 亏损在 5% 以内，且跌破MA50
        # 实际操作应判断“有效跌破”，此处简化为最新净值低于MA50
        decision = '因跌破MA50支撑减仓 50%'
        sell_reasons.append('技术性止损：跌破MA50支撑线触发')
        return { 'code': code, 'decision': decision, 'target_nav': latest_net_value, 'market_mood': market_mood }

    # -----------------------------------------------------
    # --- 4. ！！！优化点 3：动态 T1 止盈 ---
    if profit_rate >= MIN_PROFIT_FOR_T1 and bb_pos in ['中轨', '上轨附近', '上轨上方']:
        sell_pct = 0.30
        
        # 动态调整减仓比例（解决卖飞）
        if market_mood == '牛市/强势市':
            sell_pct = 0.20 # 牛市少卖，让利润奔跑
            decision_text = '【T1止盈】牛市氛围，达到T1目标，减仓 20%'
        elif market_mood == '熊市/弱势市':
            sell_pct = 0.50 # 熊市多卖，快速落袋为安
            decision_text = '【T1止盈】熊市氛围，达到T1目标，减仓 50%'
        else:
            decision_text = f'【T1止盈】震荡市，达到T1目标，减仓 {int(sell_pct*100)}%'
            
        sell_reasons.append(f'T1止盈触发，盈利>={MIN_PROFIT_FOR_T1}%，锁定部分利润 ({sell_pct*100}%)')
        return { 'code': code, 'decision': decision_text, 'target_nav': latest_net_value, 'market_mood': market_mood }

    # -----------------------------------------------------
    # --- 5. 移动止盈 (解决卖飞 - 锁定利润，让利润奔跑) ---
    max_nav = holding.get('max_nav', latest_net_value)
    if latest_net_value < max_nav * (1 - TRAILING_STOP_PERCENT) and profit_rate > 0:
        # 从最高点回撤超过 TRAILING_STOP_PERCENT 且为正收益时触发
        decision = f'移动止盈：从最高点回撤>{int(TRAILING_STOP_PERCENT*100)}% 清仓 100%'
        sell_reasons.append(decision)
        return { 'code': code, 'decision': decision, 'target_nav': latest_net_value, 'market_mood': market_mood }

    # -----------------------------------------------------
    # --- 6. 技术过热卖出 (较低优先级) ---
    
    # MACD 顶背离：价格创新高，MACD指标未创新高（此处未完全实现，仅以MACD死叉作为信号）
    if macd_signal == '死叉':
         # 牛市中死叉可能是洗盘，熊市中死叉应警惕
        if market_mood == '熊市/弱势市' or profit_rate > 5:
            decision = '技术性卖出 - MACD死叉'
            sell_reasons.append('MACD死叉')
            return { 'code': code, 'decision': decision, 'target_nav': latest_net_value * 0.98, 'market_mood': market_mood }
            
    # RSI 极度超买卖出
    if rsi > 85:
        # 牛市中容忍更高RSI（钝化），熊市中RSI>85是强卖出信号
        if market_mood == '熊市/弱势市' or profit_rate > 10:
             decision = '技术性卖出 - RSI极度超买(>85)'
             sell_reasons.append('RSI极度超买')
             return { 'code': code, 'decision': decision, 'target_nav': latest_net_value * 0.98, 'market_mood': market_mood }
             
    # -----------------------------------------------------
    # --- 7. 默认：继续持有 ---

    return {
        'code': code,
        'decision': '持仓',
        'target_nav': latest_net_value,
        'market_mood': market_mood
    }
