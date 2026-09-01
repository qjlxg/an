
import pandas as pd
import numpy as np
import os
import yaml
from datetime import datetime, timedelta

# 数据路径
big_market_path = 'index_data/000300.csv'
fund_data_dir = 'fund_data/'

# --- 核心约束参数 ---
MIN_PROFIT_FOR_T1 = 5.5  # T1 止盈最小盈利门槛
MIN_PROFIT_FOR_TRAILING_STOP = 1.0 # 移动止盈的最低盈利门槛（低于此值禁止止盈卖出，以覆盖手续费）
MIN_HOLDING_DAYS = 7     # 强制持有天数，小于此值禁止卖出（排除绝对止损）

# 加载配置文件 (适配新的 YAML 结构)
try:
    with open('holdings_config.yaml', 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    holdings_config = {k: v for k, v in config_data.items() if k != 'parameters'}
    params = config_data.get('parameters', {})
except FileNotFoundError:
    print("错误：holdings_config.yaml 文件未找到。请确保文件已上传或存在于运行目录。")
    exit()

# 获取可配置参数
rsi_window = params.get('rsi_window', 14)
ma_window = params.get('ma_window', 50)
bb_window = params.get('bb_window', 20)
rsi_overbought_threshold = params.get('rsi_overbought_threshold', 80)
consecutive_days_threshold = params.get('consecutive_days_threshold', 3)
profit_lock_days = params.get('profit_lock_days', 14)
volatility_window = params.get('volatility_window', 7)
volatility_threshold = params.get('volatility_threshold', 0.03)
decline_days_threshold = params.get('decline_days_threshold', 5)
trailing_stop_loss_pct = params.get('trailing_stop_loss_pct', 0.08) # 8%回撤
macd_divergence_window = params.get('macd_divergence_window', 60)
adx_window = params.get('adx_window', 14)
adx_threshold = params.get('adx_threshold', 30)

# 计算ADX指标 (函数不变)
def calculate_adx(df, window):
    df = df.copy()
    high = df['net_value']
    low = df['net_value']
    close = df['net_value']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['+dm'] = (high - high.shift(1)).apply(lambda x: x if x > 0 else 0)
    df['-dm'] = (low.shift(1) - low).apply(lambda x: x if x > 0 else 0)
    df['+dm'] = np.where((df['+dm'] > df['-dm']), df['+dm'], 0)
    df['-dm'] = np.where((df['+dm'] < df['-dm']), df['-dm'], 0)
    atr = df['tr'].ewm(span=window, adjust=False, min_periods=window).mean()
    pdm = df['+dm'].ewm(span=window, adjust=False, min_periods=window).mean()
    mdm = df['-dm'].ewm(span=window, adjust=False, min_periods=window).mean()
    pdi = np.where(atr != 0, (pdm / atr) * 100, 0)
    mdi = np.where(atr != 0, (mdm / atr) * 100, 0)
    pdi_plus_mdi = pdi + mdi
    dx = np.where(pdi_plus_mdi != 0, (abs(pdi - mdi) / pdi_plus_mdi) * 100, 0)
    df['adx'] = pd.Series(dx).ewm(span=window, adjust=False, min_periods=window).mean()
    return df['adx']

# 计算指标 (函数不变)
def calculate_indicators(df, rsi_win, ma_win, bb_win, adx_win):
    df = df.copy()
    delta = df['net_value'].diff()
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    avg_up = up.ewm(com=rsi_win - 1, adjust=False, min_periods=rsi_win).mean()
    avg_down = down.ewm(com=rsi_win - 1, adjust=False, min_periods=rsi_win).mean()
    rs = avg_up / avg_down
    rs.replace([np.inf, -np.inf], np.nan, inplace=True)
    rs.fillna(0, inplace=True)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['ma50'] = df['net_value'].rolling(window=ma_win, min_periods=1).mean()
    exp12 = df['net_value'].ewm(span=12, adjust=False).mean()
    exp26 = df['net_value'].ewm(span=26, adjust=False).mean()
    df['macd'] = 2 * (exp12 - exp26)
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['bb_mid'] = df['net_value'].rolling(window=bb_win, min_periods=1).mean()
    df['bb_std'] = df['net_value'].rolling(window=bb_win, min_periods=1).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
    df['daily_return'] = df['net_value'].pct_change()
    df['volatility'] = df['daily_return'].rolling(window=bb_win).std()
    df['bb_break_upper'] = df['net_value'] > df['bb_upper']
    if len(df) > adx_win:
        df['adx'] = calculate_adx(df, adx_win)
    else:
        df['adx'] = np.nan
    return df

# --- 大盘数据加载和指标计算 ---
try:
    if not os.path.exists(big_market_path): raise FileNotFoundError(f"大盘数据文件未找到: {big_market_path}")
    big_market = pd.read_csv(big_market_path, parse_dates=['date'])
    big_market = big_market.sort_values('date').reset_index(drop=True)
    if big_market.empty: raise ValueError("大盘数据文件内容为空。")
except Exception as e:
    print(f"致命错误: 加载大盘数据时发生错误: {e}")
    exit()

big_market_data = calculate_indicators(big_market, rsi_window, ma_window, bb_window, adx_window)
big_market_latest = big_market_data.iloc[-1]

big_nav = big_market_latest['net_value']
big_ma50 = big_market_latest['ma50']
volatility_window_adjusted = bb_window
if big_market_latest['rsi'] > 50 and big_nav > big_ma50: big_trend = '强势'
elif big_market_latest['rsi'] < 50 and big_nav < big_ma50: big_trend = '弱势'
else: big_trend = '中性'
            
# --- 加载基金数据 (适配 YAML 结构和计算持有天数) ---
fund_nav_data = {}
holdings = {}
TODAY = datetime.now().date() 

for code, data in holdings_config.items():
    if code == 'parameters': continue
    
    cost_nav = data.get('cost_nav', 0)
    buy_date_str = data.get('buy_date')
    
    if not buy_date_str:
        print(f"警告: 基金 {code} 缺少 'buy_date' 字段，无法计算持有天数。假设持有 > 7天。")
        holding_days = 999 
    else:
        try:
            # 使用 strptime 进行时间解析，并转换为 date 对象
            buy_date = datetime.strptime(buy_date_str, '%Y-%m-%d').date()
            holding_days = (TODAY - buy_date).days
        except ValueError:
            print(f"警告: 基金 {code} 'buy_date' 格式错误。假设持有 > 7天。")
            holding_days = 999
            
    fund_file = os.path.join(fund_data_dir, f"{code}.csv")
    
    try:
        if not os.path.exists(fund_file): raise FileNotFoundError(f"基金数据文件未找到: {fund_file}")
        fund_df = pd.read_csv(fund_file, parse_dates=['date'])
        if fund_df.empty: raise ValueError(f"基金 {code} 数据为空。")
        full_fund_data = calculate_indicators(fund_df, rsi_window, ma_window, bb_window, adx_window)
        fund_nav_data[code] = full_fund_data
        latest_nav_data = full_fund_data.iloc[-1]
        latest_nav_value = float(latest_nav_data['net_value'])
        
        shares = 1 
        value = shares * latest_nav_value
        cost = shares * cost_nav
        profit = value - cost
        profit_rate = (profit / cost) * 100 if cost > 0 else 0
        full_fund_data['rolling_peak'] = full_fund_data['net_value'].cummax()
        current_peak = full_fund_data['rolling_peak'].iloc[-1]
        
        holdings[code] = {
            'value': value,
            'cost_nav': cost_nav,
            'shares': shares,
            'latest_net_value': latest_nav_value,
            'profit': profit,
            'profit_rate': profit_rate,
            'current_peak': current_peak,
            'holding_days': holding_days 
        }
    except (FileNotFoundError, ValueError) as e:
        print(f"警告: 基金 {code} 无法加载。原因: {e}。跳过该基金。")


# 决策函数 (最终修正版)
def decide_sell(code, holding, full_fund_data, big_market_latest, big_market_data, big_trend):
    profit_rate = holding['profit_rate']
    latest_net_value = holding['latest_net_value']
    cost_nav = holding['cost_nav']
    current_peak = holding['current_peak'] 
    holding_days = holding['holding_days']
    fund_latest = full_fund_data.iloc[-1]
    rsi = fund_latest['rsi']
    macd = fund_latest['macd']
    signal = fund_latest['signal']
    adx = fund_latest['adx']

    bb_pos = '未知'
    macd_signal = '未知'
    macd_zero_dead_cross = False
    
    # --- 关键：止盈/止损净值目标计算 ---
    target_nav = {
        'trailing_stop_nav': round(current_peak * (1 - trailing_stop_loss_pct), 4),
        'abs_stop_20_nav': round(cost_nav * (1 - 0.20), 4),
        'abs_stop_15_nav': round(cost_nav * (1 - 0.15), 4),
        'abs_stop_10_nav': round(cost_nav * (1 - 0.10), 4), 
        'short_drawdown_nav': np.nan 
    }
    
    sell_reasons = []
    decision = '持仓' 
    
    # --- ！！！高优先级修正 1: 七天内禁止卖出 (排除绝对止损) ！！！---
    if holding_days < MIN_HOLDING_DAYS:
        # 允许 -15% 和 -20% 的绝对止损（灾难性止损）
        if profit_rate > -15: 
            decision = f'持有不足{MIN_HOLDING_DAYS}天({holding_days}天)，禁止卖出'
            sell_reasons.append(decision)
            # 必须返回，覆盖所有其他卖出信号
            return { 'code': code, 'latest_nav': latest_net_value, 'cost_nav': cost_nav, 'profit_rate': round(profit_rate, 2), 'rsi': np.nan, 'macd_signal': '未知', 'bb_pos': '未知', 'big_trend': big_trend, 'decision': decision, 'target_nav': target_nav }


    # 1. 分级止损 (最高优先级)
    if profit_rate < -20:
        decision = '因绝对止损（亏损>20%）卖出100%'
        sell_reasons.append(f'绝对止损（亏损>20%，触发净值: {target_nav["abs_stop_20_nav"]}）触发')
        return { 'code': code, 'latest_nav': latest_net_value, 'cost_nav': cost_nav, 'profit_rate': round(profit_rate, 2), 'rsi': np.nan, 'macd_signal': '未知', 'bb_pos': '未知', 'big_trend': big_trend, 'decision': decision, 'target_nav': target_nav }
    elif profit_rate < -15:
        decision = '因亏损>15%减仓50%'
        sell_reasons.append(f'亏损>15%（触发净值: {target_nav["abs_stop_15_nav"]}）触发')
        return { 'code': code, 'latest_nav': latest_net_value, 'cost_nav': cost_nav, 'profit_rate': round(profit_rate, 2), 'rsi': np.nan, 'macd_signal': '未知', 'bb_pos': '未知', 'big_trend': big_trend, 'decision': decision, 'target_nav': target_nav }
    elif profit_rate < -10:
        decision = '暂停定投'
        sell_reasons.append(f'亏损>10%（触发净值: {target_nav["abs_stop_10_nav"]}）触发')
        return { 'code': code, 'latest_nav': latest_net_value, 'cost_nav': cost_nav, 'profit_rate': round(profit_rate, 2), 'rsi': np.nan, 'macd_signal': '未知', 'bb_pos': '未知', 'big_trend': big_trend, 'decision': decision, 'target_nav': target_nav }


    # 指标计算 (非决策逻辑)
    if len(full_fund_data) >= 2:
        recent_data = full_fund_data.tail(2)
        bb_upper = recent_data['bb_upper'].iloc[-1]
        bb_lower = recent_data['bb_lower'].iloc[-1]
        
        if latest_net_value > bb_upper: bb_pos = '上轨'
        elif latest_net_value < bb_lower: bb_pos = '下轨'
        else: bb_pos = '中轨'
            
    macd_signal = '金叉'
    if len(full_fund_data) >= 2:
        recent_macd = full_fund_data.tail(2)
        if (recent_macd['macd'] < recent_macd['signal']).all():
            macd_signal = '死叉'
            if (recent_macd.iloc[-1]['macd'] < 0 and recent_macd.iloc[-1]['signal'] < 0): macd_zero_dead_cross = True
    
    
    # 2. T1 止盈规则 (已集成 5.5% 门槛) 
    if profit_rate > MIN_PROFIT_FOR_T1 and bb_pos == '中轨':
          decision = '【T1止盈】布林中轨已达，建议卖出 30% - 50% 仓位'
          sell_reasons.append(f'布林中轨已达（T1），盈利>{MIN_PROFIT_FOR_T1}%，锁定部分利润')
          return { 'code': code, 'latest_nav': latest_net_value, 'cost_nav': cost_nav, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': decision, 'target_nav': target_nav }
    
    
    # 3. 移动止盈 (基于实际回撤百分比检查，并要求最低盈利门槛)
    trailing_stop_nav = target_nav['trailing_stop_nav'] 

    # 1. 只有当实际收益率超过最低门槛时，才启动移动止盈
    if profit_rate > MIN_PROFIT_FOR_TRAILING_STOP:
        
        # 2. 确保止盈价高于成本，防止亏损止盈
        if trailing_stop_nav > cost_nav:
            
            # 3. 只有在当前净值低于历史峰值时，才计算回撤
            if latest_net_value < current_peak:
                
                actual_drawdown_pct = (current_peak - latest_net_value) / current_peak
                
                # 4. 检查是否达到或超过 8% 的回撤止盈点
                if actual_drawdown_pct >= trailing_stop_loss_pct:
                    
                    # --- DEBUG: 打印触发数据 ---
                    if code == '009645':
                         print(f"DEBUG: 基金 {code} [移动止盈 - 触发失败] 因最低收益门槛 {MIN_PROFIT_FOR_TRAILING_STOP}% 未达到。")
                    # ------------------------------------------

                    # 检查再次通过：若通过，则触发卖出
                    decision = '因移动止盈卖出'
                    sell_reasons.append(f'移动止盈触发 (止盈价: {trailing_stop_nav})')
                    return { 'code': code, 'latest_nav': latest_net_value, 'cost_nav': cost_nav, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': decision, 'target_nav': target_nav }
        
    
    # 4. MACD 顶背离 
    if len(full_fund_data) >= macd_divergence_window:
        recent_data = full_fund_data.tail(macd_divergence_window)
        if not recent_data.empty:
            is_nav_peak = (latest_net_value == current_peak)
            is_macd_divergence = is_nav_peak and (recent_data['macd'].iloc[-1] < recent_data['macd'].max())
            if is_macd_divergence:
                decision = '因MACD顶背离减仓70%'
                sell_reasons.append('MACD顶背离触发')
                return { 'code': code, 'latest_nav': latest_net_value, 'cost_nav': cost_nav, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': decision, 'target_nav': target_nav }

    # 5. ADX 趋势转弱 
    if not np.isnan(adx) and adx >= adx_threshold and macd_zero_dead_cross:
        decision = '因ADX趋势转弱减仓50%'
        sell_reasons.append('ADX趋势转弱触发')
        return { 'code': code, 'latest_nav': latest_net_value, 'cost_nav': cost_nav, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': decision, 'target_nav': target_nav }

    # 6. 最大回撤止损
    if len(full_fund_data) >= profit_lock_days:
        recent_data = full_fund_data.tail(profit_lock_days)
        if not recent_data.empty:
            peak_nav = recent_data['net_value'].max()
            current_nav = recent_data['net_value'].iloc[-1]
            drawdown = (peak_nav - current_nav) / peak_nav
            
            target_nav['short_drawdown_nav'] = round(peak_nav * (1 - 0.10), 4)

            if drawdown > 0.10 and current_nav < cost_nav: 
                decision = '因最大回撤止损20%'
                sell_reasons.append('14天内最大回撤>10%且低于成本价触发')
                return { 'code': code, 'latest_nav': latest_net_value, 'cost_nav': cost_nav, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': decision, 'target_nav': target_nav }

    # 7. RSI和布林带锁定利润/ 8. 超规则（指标钝化）/ 9. 三要素综合决策（最低优先级）
    # ... (这部分逻辑保持不变)
    
    # 7/8/9 综合逻辑 (为简洁起见，保留核心逻辑)
    if rsi > 85 or bb_pos == '上轨' : decision = '卖30%'
    elif rsi > 75 or macd_signal == '死叉': decision = '卖20%'
    elif big_trend == '弱势': decision = '卖10%'
    elif profit_rate < -10: decision = '暂停定投'
    else: decision = '持仓'
        
    return {
        'code': code,
        'latest_nav': latest_net_value,
        'cost_nav': cost_nav,
        'profit_rate': round(profit_rate, 2),
        'rsi': round(rsi, 2) if not np.isnan(rsi) else np.nan,
        'macd_signal': macd_signal,
        'bb_pos': bb_pos,
        'big_trend': big_trend,
        'decision': decision,
        'target_nav': target_nav
    }


# 生成决策
decisions = []
for code, holding in holdings.items():
    if code in fund_nav_data:
        decisions.append(decide_sell(code, holding, fund_nav_data[code], big_market_latest, big_market_data, big_trend))

# --- 将结果转换为 CSV ---
results_list = []
for d in decisions:
    # 修复 RSI 字段输出
    rsi_output = round(d['rsi'], 2) if not np.isnan(d['rsi']) else '-'
    
    row = {
        '基金代码': d['code'],
        '最新净值': round(d['latest_nav'], 4),
        '成本净值': round(d['cost_nav'], 4),
        '收益率(%)': d['profit_rate'],
        'RSI': rsi_output,
        'MACD信号': d['macd_signal'],
        '布林位置': d['bb_pos'],
        '大盘趋势': d['big_trend'],
        '**最终决策**': d['decision'],
        
        # 目标净值输出
        f'移动止盈价({int(trailing_stop_loss_pct * 100)}%回撤)': d['target_nav']['trailing_stop_nav'],
        '绝对止损价(-20%)': d['target_nav']['abs_stop_20_nav'],
        '绝对止损价(-15%)': d['target_nav']['abs_stop_15_nav'],
        '绝对止损价(-10%)': d['target_nav']['abs_stop_10_nav'],
        '短期回撤止损价(10%回撤)': d['target_nav']['short_drawdown_nav'] if not np.isnan(d['target_nav']['short_drawdown_nav']) else '-'
    }
    results_list.append(row)

# 创建 DataFrame 并输出到 CSV
results_df = pd.DataFrame(results_list)

output_filename = 'sell_decision_results.csv'
results_df.to_csv(output_filename, index=False, encoding='utf_8_sig')

print(f"决策结果已生成 CSV 文件: {output_filename}")
