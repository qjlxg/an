# bot.py
import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import telebot

bot_token = open("bot.token", "r", encoding="utf-8").read().strip()
print("Bot Token:", bot_token)
bot = telebot.TeleBot(bot_token)

# Subscription management

def load_subscriptions(path="subscriptions.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_subscriptions(subscriptions, path="subscriptions.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(subscriptions, f)

subscriptions = load_subscriptions()

# Handlers

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Welcome! I am your friendly stock bot. Use /help to see available commands.")

@bot.message_handler(commands=['help'])
def send_help(message):
    print(message.from_user)
    help_text = (
        "Available commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/subscribe <symbol> - Subscribe to updates for a stock symbol\n"
        "/unsubscribe <symbol> - Unsubscribe from updates for a stock symbol\n"
        "/price <symbol> - Get the current price of the stock symbol\n"
        "/id - Get your user ID\n"
        "You can also send any message and I will echo it back!"
    )
    bot.reply_to(message, help_text)

@bot.message_handler(commands=['price'])
def send_price(message):
    try:
        _, symbol = message.text.split()
        price = get_stock_price(symbol)
        bot.reply_to(message, price)
    except ValueError:
        bot.reply_to(message, "Please provide a stock symbol. Usage: /price <symbol>")

from Ashare import get_price
def get_stock_price(symbol):
    # Placeholder function to simulate fetching stock price
    # In a real implementation, this would fetch data from an API
    price = get_price(symbol, frequency='1m', count=1)['close'].values[0]
    return f"The current price of {symbol} is {price}"

@bot.message_handler(commands=['id'])
def send_user_id(message):
    user_id = message.from_user.id
    bot.reply_to(message, f"Your user ID is: {user_id}")

@bot.message_handler(commands=['subscribe'])
def subscribe_stock(message):
    user_id = str(message.from_user.id)
    try:
        _, symbol = message.text.split()
        # Placeholder for subscription logic
        if user_id not in subscriptions:
            bot.reply_to(message, 'Sorry, you are not authorized to subscribe.')
            return
        if symbol not in subscriptions[user_id]:
            subscriptions[user_id][symbol] = []
            save_subscriptions(subscriptions)
            bot.reply_to(message, f"Subscribed to {symbol}.")
        else:
            bot.reply_to(message, f"Already subscribed to {symbol}.")
    except ValueError:
        bot.reply_to(message, "Please provide a stock symbol. Usage: /subscribe <symbol>")
        if user_id in subscriptions:
            bot.reply_to(message, f"Your current subscriptions: {', '.join(subscriptions[user_id].keys())}")

@bot.message_handler(commands=['unsubscribe'])
def unsubscribe_stock(message):
    try:
        _, symbol = message.text.split()
        user_id = str(message.from_user.id)
        if user_id not in subscriptions or symbol not in subscriptions[user_id]:
            bot.reply_to(message, f"You are not subscribed to {symbol}.")
            return
        del subscriptions[user_id][symbol]
        save_subscriptions(subscriptions)
        bot.reply_to(message, f"Unsubscribed from {symbol}.")
    except ValueError:
        bot.reply_to(message, "Please provide a stock symbol. Usage: /unsubscribe <symbol>")
        if user_id in subscriptions:
            bot.reply_to(message, f"Your current subscriptions: {', '.join(subscriptions[user_id].keys())}")

@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    bot.reply_to(message, message.text)

# Check RSI and notify users (Placeholder implementation)

# 24小时去重窗口：同一用户-股票的RSI告警在该窗口内只发一次
ALERT_DEDUP_WINDOW = timedelta(hours=24)

# 内存级的去重记录：{(user_id, symbol): datetime_of_last_alert}
last_rsi_alert_at = {}

# ===== 新增：RSI(12) 计算（Wilder 平滑）=====
def rsi_wilder(close: pd.Series, period: int = 12) -> pd.Series:
    """
    计算Wilder平滑的RSI，返回与close同长度的Series。
    要求close为升序时间序列（最早->最新）。
    """
    close = close.dropna()
    if len(close) < period + 1:
        return pd.Series(index=close.index, dtype=float)

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # 初始均值
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Wilder递推
    # 从第 period+1 个点开始用递推，之前用rolling均值
    rs = pd.Series(index=close.index, dtype=float)
    rsi = pd.Series(index=close.index, dtype=float)

    # 找到第一个可用位置
    first = avg_gain.first_valid_index()
    if first is None:
        return rsi  # 全NaN

    rs.loc[first] = (avg_gain.loc[first] / avg_loss.loc[first]) if avg_loss.loc[first] != 0 else np.inf
    rsi.loc[first] = 100 - (100 / (1 + rs.loc[first]))

    # 递推
    for i in range(close.index.get_loc(first) + 1, len(close)):
        idx = close.index[i]
        g = gain.iloc[i]
        l = loss.iloc[i]

        prev_idx = close.index[i - 1]
        prev_avg_gain = avg_gain.loc[prev_idx] if not pd.isna(avg_gain.loc[prev_idx]) else None
        prev_avg_loss = avg_loss.loc[prev_idx] if not pd.isna(avg_loss.loc[prev_idx]) else None

        # 如果 rolling 平均还不可用，就跳过（直到有初值）
        if prev_avg_gain is None or prev_avg_loss is None:
            continue

        cur_avg_gain = (prev_avg_gain * (period - 1) + g) / period
        cur_avg_loss = (prev_avg_loss * (period - 1) + l) / period

        avg_gain.loc[idx] = cur_avg_gain
        avg_loss.loc[idx] = cur_avg_loss

        cur_rs = (cur_avg_gain / cur_avg_loss) if cur_avg_loss != 0 else np.inf
        rs.loc[idx] = cur_rs
        rsi.loc[idx] = 100 - (100 / (1 + cur_rs))

    return rsi

# ===== 新增：拉取数据并计算RSI最新值 =====
def get_latest_rsi12(symbol: str) -> float | None:
    """
    用日线数据计算RSI(12)的最新值。
    返回 float 或 None（数据不足/异常）。
    """
    try:
        # 取足够长度，避免滚动期不足
        df = get_price(symbol, frequency='1d', count=120)
        if df.index[-1] < pd.Timestamp(datetime.now().date()):
            df_newest = get_price(symbol, frequency='1m', count=1).dropna()
            df = pd.concat([df, df_newest]).drop_duplicates().sort_index()
        if df is None or df.empty or 'close' not in df.columns:
            return None
        close = df['close']
        rsi = rsi_wilder(close, period=12)
        if rsi.empty or pd.isna(rsi.iloc[-1]):
            return None
        return float(rsi.iloc[-1])
    except Exception as e:
        # 生产建议加日志
        print(f"[RSI] fetch/compute failed for {symbol}: {e}")
        return None

# ===== 新增：告警去重判断 =====
def should_alert(user_id: str, symbol: str) -> bool:
    key = (user_id, symbol)
    now = datetime.now(timezone.utc)
    last_time = last_rsi_alert_at.get(key)
    if last_time is None or now - last_time >= ALERT_DEDUP_WINDOW:
        last_rsi_alert_at[key] = now
        return True
    return False

# ===== 新增：RSI 检查任务 =====
def check_rsi_and_notify():
    """
    遍历 subscriptions，计算每个 symbol 的 RSI(12)。
    若 > 80，且未在去重窗口内提醒过，则发送消息给对应用户。
    """
    # 复制一份，避免遍历期间修改带来问题
    subs_snapshot = dict(subscriptions)

    for user_id, sym_dict in subs_snapshot.items():
        # 兼容：有些人会把 subscriptions[user_id] 写成 list 或 set，这里只接受 dict
        if not isinstance(sym_dict, dict):
            continue

        for symbol in list(sym_dict.keys()):
            rsi12 = get_latest_rsi12(symbol)
            print(f"[RSI] {symbol} RSI(12)={rsi12}")
            if rsi12 is None:
                continue

            if rsi12 > 30.0 or rsi12 < 70.0:
                if should_alert(user_id, symbol):
                    try:
                        if rsi12 < 30.0:
                            bot.send_message(
                                user_id,
                                f"【RSI提醒】{symbol} 当前 RSI(12) = {rsi12:.2f}（< 30），可能处于超卖区间，请留意风险。"
                            )
                        elif rsi12 > 70.0:
                            bot.send_message(
                                user_id,
                                f"【RSI提醒】{symbol} 当前 RSI(12) = {rsi12:.2f}（> 70），可能处于超买区间，请留意风险。"
                            )
                        # 如需持久化去重时间，可在此写回 subscriptions[user_id][symbol] 后 save_subscriptions(subscriptions)
                    except Exception as e:
                        print(f"[RSI] send_message failed user={user_id}, symbol={symbol}: {e}")

# ===== 新增：后台循环线程 =====
def rsi_background_worker(interval_seconds: int = 300):
    """
    每 interval_seconds 秒执行一次 RSI 检查。
    """
    while True:
        try:
            check_rsi_and_notify()
        except Exception as e:
            print(f"[RSI] background worker error: {e}")
        time.sleep(interval_seconds)

# Start polling

if __name__ == "__main__":
    try:
        # 启动RSI后台检查线程（每5分钟）
        t = threading.Thread(target=rsi_background_worker, args=(300,), daemon=True)
        t.start()

        bot.send_message('6812353037', 'Bot started successfully!')
        bot.infinity_polling()
    except Exception as e:
        bot.send_message('6812353037', f'Bot stopped unexpectedly! Error: {e}')
