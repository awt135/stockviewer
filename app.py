# file: app.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import akshare as ak
import pandas as pd
import numpy as np
import math
import re

app = FastAPI(title="A股K线API")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或者指定前端地址 ["http://127.0.0.1:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 工具函数 ----------
def clean_nan(obj):
    """递归替换 NaN / inf 为 None"""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

def calc_indicators(df):
    # MA
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    
    # 多重移动平均线 (14,21,35,50,100,200)
    df['ma14'] = df['close'].rolling(14).mean()
    df['ma21'] = df['close'].rolling(21).mean()
    df['ma35'] = df['close'].rolling(35).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma100'] = df['close'].rolling(100).mean()
    df['ma200'] = df['close'].rolling(200).mean()
    
    # BOLL (20,2)
    df['bb_mid'] = df['ma20']
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2*df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2*df['bb_std']
    
    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_diff'] = exp12 - exp26
    df['macd_signal'] = df['macd_diff'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_diff'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    df['rsi14'] = 100 - 100/(1 + roll_up/roll_down)
    
    # Stoch RSI (14,14,3,3)
    df['stoch_rsi_k'], df['stoch_rsi_d'] = calc_stoch_rsi(df, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3)
    
    # 超级趋势 Supertrend (10,3)
    df['supertrend'] = calc_supertrend(df, period=10, multiplier=3)
    
    # 计算SKDJ指标
    df['skdj_k'], df['skdj_d'] = calc_skdj(df)
    
    # 计算策略信号
    df['td9_signal'] = calc_td9_strategy(df)
    df['ema_signal'] = calc_ema_strategy(df)
    df['super_band_signal'] = calc_super_band_strategy(df)
    df['mk_resonance'] = calc_mk_resonance(df)
    
    # 计算OBV和MFI指标
    df['obv'] = calc_obv(df)
    df['mfi14'] = calc_mfi(df, period=14)
    
    # 计算CCI、ROC和Williams %R指标
    df['cci20'] = calc_cci(df, period=20)
    df['cci14'] = calc_cci(df, period=14)
    df['roc10'] = calc_roc(df, period=10)
    df['roc20'] = calc_roc(df, period=20)
    df['williams_r14'] = calc_williams_r(df, period=14)
    
    # 计算KDJ指标（标准：9,3,3）
    df['kdj_k'], df['kdj_d'], df['kdj_j'] = calc_kdj(df, n=9, m1=3, m2=3)
    
    # 计算TRIX指标（三重指数平滑）
    df['trix'], df['trix_signal'] = calc_trix(df, period=12)
    
    # 计算BBI指标（多空指数）
    df['bbi'] = calc_bbi(df)
    
    # 计算ZigZag指标（5%阈值）
    df['zigzag_5'] = calc_zigzag(df, threshold_percent=5.0)
    
    # 计算ZigZag指标（7%阈值）
    df['zigzag_7'] = calc_zigzag(df, threshold_percent=7.0)
    
    # 计算PIVOT枢轴点指标
    pivot_points = calc_pivot_points(df)
    df['pivot'] = [point['pivot'] for point in pivot_points]
    df['resistance1'] = [point['resistance1'] for point in pivot_points]
    df['resistance2'] = [point['resistance2'] for point in pivot_points]
    df['support1'] = [point['support1'] for point in pivot_points]
    df['support2'] = [point['support2'] for point in pivot_points]
    
    # 计算Donchian Channel唐奇安通道
    dc_high, dc_low = calc_donchian_channel(df, period=20)
    df['dc_high20'] = dc_high
    df['dc_low20'] = dc_low
    
    # 填充NaN值，但保留策略信号中的0值（0表示无信号）
    # 先保存策略信号列
    strategy_columns = ['td9_signal', 'ema_signal', 'super_band_signal', 'mk_resonance']
    strategy_data = {}
    for col in strategy_columns:
        strategy_data[col] = df[col].copy()

    # 填充其他列的NaN值 (pandas新版本写法)
    df = df.ffill().bfill()

    # 恢复策略信号列，确保0值不被填充为其他值
    for col in strategy_columns:
        df[col] = strategy_data[col]
        # 只填充策略信号中的None值，保留0值
        df[col] = df[col].fillna(0)
    
    return df

def calc_stoch_rsi(df, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    """计算Stoch RSI指标"""
    # 计算RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(rsi_period).mean()
    roll_down = down.rolling(rsi_period).mean()
    rsi = 100 - 100 / (1 + roll_up / roll_down)
    
    # 计算Stoch RSI
    stoch_rsi_k = []
    stoch_rsi_d = []
    
    for i in range(len(df)):
        if i < stoch_period - 1:
            stoch_rsi_k.append(None)
            stoch_rsi_d.append(None)
            continue
            
        # 获取当前窗口内的RSI值
        rsi_window = rsi.iloc[i-stoch_period+1:i+1]
        rsi_min = rsi_window.min()
        rsi_max = rsi_window.max()
        
        if rsi_max == rsi_min or pd.isna(rsi_max) or pd.isna(rsi_min):
            stoch_rsi_k.append(None)
        else:
            stoch_rsi_k.append(((rsi.iloc[i] - rsi_min) / (rsi_max - rsi_min)) * 100)  # 乘以100转换为百分比形式
    
    # 平滑K线
    stoch_rsi_k_smooth = []
    for i in range(len(stoch_rsi_k)):
        if i < k_smooth - 1:
            stoch_rsi_k_smooth.append(None)
        else:
            window = stoch_rsi_k[i-k_smooth+1:i+1]
            if any(x is None for x in window) or len(window) < k_smooth:
                stoch_rsi_k_smooth.append(None)
            else:
                stoch_rsi_k_smooth.append(sum(window) / k_smooth)
    
    # 计算D线
    for i in range(len(stoch_rsi_k_smooth)):
        if i < d_smooth - 1 or len(stoch_rsi_k_smooth) < d_smooth:
            stoch_rsi_d.append(None)
        else:
            window = stoch_rsi_k_smooth[i-d_smooth+1:i+1]
            if any(x is None for x in window) or len(window) < d_smooth:
                stoch_rsi_d.append(None)
            else:
                stoch_rsi_d.append(sum(window) / d_smooth)
    
    # 确保返回的数据长度与原始数据一致
    while len(stoch_rsi_k_smooth) < len(df):
        stoch_rsi_k_smooth.append(None)
    while len(stoch_rsi_d) < len(df):
        stoch_rsi_d.append(None)
    
    return stoch_rsi_k_smooth[:len(df)], stoch_rsi_d[:len(df)]

def calc_supertrend(df, period=10, multiplier=3):
    """计算超级趋势指标"""
    # 计算ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()  # 使用指数移动平均替代简单移动平均
    
    # 计算超级趋势
    hl2 = (df['high'] + df['low']) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = []
    prev_upper = None
    prev_lower = None
    prev_trend = None
    
    for i in range(len(df)):
        if i < period:
            supertrend.append(None)
            continue
            
        current_upper = upper_band.iloc[i]
        current_lower = lower_band.iloc[i]
        
        if prev_upper is None or prev_lower is None:
            supertrend_value = current_lower
            trend = 1
        else:
            # 调整上轨
            if current_upper < prev_upper or df['close'].iloc[i-1] > prev_upper:
                upper_final = current_upper
            else:
                upper_final = prev_upper
            
            # 调整下轨
            if current_lower > prev_lower or df['close'].iloc[i-1] < prev_lower:
                lower_final = current_lower
            else:
                lower_final = prev_lower
            
            # 确定趋势
            if prev_trend == 1 and df['close'].iloc[i] <= lower_final:
                trend = -1
                supertrend_value = upper_final
            elif prev_trend == -1 and df['close'].iloc[i] >= upper_final:
                trend = 1
                supertrend_value = lower_final
            else:
                trend = prev_trend
                supertrend_value = lower_final if trend == 1 else upper_final
        
        supertrend.append(supertrend_value)
        prev_upper = upper_final if 'upper_final' in locals() else current_upper
        prev_lower = lower_final if 'lower_final' in locals() else current_lower
        prev_trend = trend
    
    # 确保返回的数据长度与原始数据一致
    while len(supertrend) < len(df):
        supertrend.append(None)
    
    return supertrend[:len(df)]

def calc_skdj(df, n=9, m=3):
    """计算SKDJ指标"""
    # 计算RSV值
    rsv = []
    for i in range(len(df)):
        if i < n - 1:
            rsv.append(None)
        else:
            low_n = df['low'].iloc[i-n+1:i+1].min()
            high_n = df['high'].iloc[i-n+1:i+1].max()
            if high_n == low_n:
                rsv.append(50)
            else:
                rsv.append((df['close'].iloc[i] - low_n) / (high_n - low_n) * 100)
    
    # 计算K值和D值
    k_values = []
    d_values = []
    
    for i in range(len(rsv)):
        if rsv[i] is None:
            k_values.append(None)
            d_values.append(None)
        else:
            if i == 0 or k_values[i-1] is None:
                k_values.append(50)
            else:
                k_values.append(k_values[i-1] * 2/3 + rsv[i] * 1/3)
            
            if i < m - 1 or any(k is None for k in k_values[max(0, i-m+1):i+1]):
                d_values.append(None)
            else:
                d_values.append(sum(k_values[max(0, i-m+1):i+1]) / min(m, i+1))
    
    return k_values, d_values

def calc_td9_strategy(df):
    """计算TD9抄底做多战法策略"""
    td9_signals = []
    
    for i in range(len(df)):
        if i < 8:
            td9_signals.append(None)
            continue
        
        # TD9策略：连续9个交易日收盘价低于前4个交易日的收盘价
        is_td9_buy = True
        for j in range(9):
            if i - j < 4:
                is_td9_buy = False
                break
            if df['close'].iloc[i-j] >= df['close'].iloc[i-j-4]:
                is_td9_buy = False
                break
        
        td9_signals.append(1 if is_td9_buy else 0)
    
    return td9_signals

def calc_ema_strategy(df):
    """计算EMA交易策略"""
    # 计算EMA指标
    ema_short = df['close'].ewm(span=12, adjust=False).mean()
    ema_long = df['close'].ewm(span=26, adjust=False).mean()
    
    # 策略信号：EMA金叉死叉
    ema_signals = []
    for i in range(len(df)):
        if i == 0:
            ema_signals.append(0)
        else:
            if ema_short.iloc[i] > ema_long.iloc[i] and ema_short.iloc[i-1] <= ema_long.iloc[i-1]:
                ema_signals.append(1)  # 金叉买入信号
            elif ema_short.iloc[i] < ema_long.iloc[i] and ema_short.iloc[i-1] >= ema_long.iloc[i-1]:
                ema_signals.append(-1)  # 死叉卖出信号
            else:
                ema_signals.append(0)  # 无信号
    
    return ema_signals

def calc_super_band_strategy(df):
    """计算超级波段追踪多头策略"""
    # 使用布林带和RSI结合
    bb_upper = df['bb_upper']
    bb_lower = df['bb_lower']
    rsi = df['rsi14']
    
    super_band_signals = []
    
    for i in range(len(df)):
        if i == 0 or bb_upper.iloc[i] is None or bb_lower.iloc[i] is None or rsi.iloc[i] is None:
            super_band_signals.append(0)
            continue
        
        # 策略：价格突破布林带上轨且RSI<70为买入信号
        if df['close'].iloc[i] > bb_upper.iloc[i] and rsi.iloc[i] < 70:
            super_band_signals.append(1)
        # 策略：价格跌破布林带下轨且RSI>30为卖出信号
        elif df['close'].iloc[i] < bb_lower.iloc[i] and rsi.iloc[i] > 30:
            super_band_signals.append(-1)
        else:
            super_band_signals.append(0)
    
    return super_band_signals

def calc_mk_resonance(df):
    """计算MK共振指标"""
    mk_signals = []

    for i in range(len(df)):
        if i < 1 or any(pd.isna(df.iloc[i][col]) for col in ['macd_diff', 'macd_signal', 'rsi14', 'supertrend']):
            mk_signals.append(0)
            continue

        # MK共振条件：多个指标同时发出同向信号
        signals = []

        # MACD金叉
        if (df['macd_diff'].iloc[i] > df['macd_signal'].iloc[i] and
            df['macd_diff'].iloc[i-1] <= df['macd_signal'].iloc[i-1]):
            signals.append(1)
        # MACD死叉
        elif (df['macd_diff'].iloc[i] < df['macd_signal'].iloc[i] and
              df['macd_diff'].iloc[i-1] >= df['macd_signal'].iloc[i-1]):
            signals.append(-1)
        else:
            signals.append(0)

        # RSI超买超卖信号
        if df['rsi14'].iloc[i] > 80:
            signals.append(-1)  # 超买卖出
        elif df['rsi14'].iloc[i] < 20:
            signals.append(1)   # 超卖买入
        else:
            signals.append(0)

        # 超级趋势信号
        if df['close'].iloc[i] > df['supertrend'].iloc[i]:
            signals.append(1)   # 趋势向上
        elif df['close'].iloc[i] < df['supertrend'].iloc[i]:
            signals.append(-1)  # 趋势向下
        else:
            signals.append(0)

        # 计算共振强度：多个同向信号产生强信号
        positive_signals = sum(1 for s in signals if s > 0)
        negative_signals = sum(1 for s in signals if s < 0)

        if positive_signals >= 2:
            mk_signals.append(2)  # 强买入信号
        elif positive_signals == 1:
            mk_signals.append(1)  # 弱买入信号
        elif negative_signals >= 2:
            mk_signals.append(-2) # 强卖出信号
        elif negative_signals == 1:
            mk_signals.append(-1) # 弱卖出信号
        else:
            mk_signals.append(0)  # 无信号

    return mk_signals

def calc_obv(df):
    """计算OBV（能量潮）指标，并进行归一化处理到0~100范围"""
    obv_values = []
    
    for i in range(len(df)):
        if i == 0:
            # 第一天OBV等于当日成交量
            obv_values.append(df['volume'].iloc[i])
        else:
            # 比较当日收盘价与前日收盘价
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                # 上涨：OBV累加当日成交量
                obv_values.append(obv_values[i-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                # 下跌：OBV累减当日成交量
                obv_values.append(obv_values[i-1] - df['volume'].iloc[i])
            else:
                # 平盘：OBV保持不变
                obv_values.append(obv_values[i-1])
    
    # 对OBV数据进行归一化处理到0~100范围
    if obv_values:
        min_obv = min(obv_values)
        max_obv = max(obv_values)
        
        # 如果所有值都相同，则全部设为50
        if max_obv == min_obv:
            normalized_obv = [50.0] * len(obv_values)
        else:
            # 最小-最大归一化到0~100范围
            normalized_obv = [(value - min_obv) / (max_obv - min_obv) * 100 for value in obv_values]
        
        return normalized_obv
    
    return obv_values

def calc_mfi(df, period=14):
    """计算MFI（资金流量指数）指标"""
    # 计算典型价格
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # 计算原始资金流
    raw_money_flow = typical_price * df['volume']
    
    # 计算正负资金流
    positive_mf = []
    negative_mf = []
    
    for i in range(len(df)):
        if i == 0:
            positive_mf.append(0)
            negative_mf.append(0)
        else:
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_mf.append(raw_money_flow.iloc[i])
                negative_mf.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_mf.append(0)
                negative_mf.append(raw_money_flow.iloc[i])
            else:
                positive_mf.append(0)
                negative_mf.append(0)
    
    # 计算资金流比率
    mfi_values = []
    
    for i in range(len(df)):
        if i < period:
            mfi_values.append(None)
        else:
            # 计算周期内的正负资金流总和
            sum_positive = sum(positive_mf[i-period+1:i+1])
            sum_negative = sum(negative_mf[i-period+1:i+1])
            
            if sum_negative == 0 and sum_positive == 0:
                mfi_values.append(50)  # 中性值
            elif sum_negative == 0:
                mfi_values.append(100)
            else:
                money_ratio = sum_positive / sum_negative
                mfi = 100 - (100 / (1 + money_ratio))
                mfi_values.append(mfi)
    
    # 确保返回的数据长度与原始数据一致
    while len(mfi_values) < len(df):
        mfi_values.append(None)
    
    return mfi_values[:len(df)]

def calc_cci(df, period=20):
    """计算CCI（商品通道指标）"""
    cci_values = []
    
    for i in range(len(df)):
        if i < period - 1:
            cci_values.append(None)
        else:
            # 计算典型价格
            typical_price = (df['high'].iloc[i] + df['low'].iloc[i] + df['close'].iloc[i]) / 3
            
            # 计算周期内的典型价格移动平均
            typical_prices = []
            for j in range(period):
                idx = i - j
                if idx >= 0:  # 确保索引有效
                    typical_prices.append((df['high'].iloc[idx] + df['low'].iloc[idx] + df['close'].iloc[idx]) / 3)
            
            # 检查是否有足够的数据
            if len(typical_prices) < period:
                cci_values.append(None)
                continue
                
            sma_tp = sum(typical_prices) / period
            
            # 计算平均偏差
            mean_deviation = 0
            for tp in typical_prices:
                mean_deviation += abs(tp - sma_tp)
            mean_deviation /= period
            
            # 计算CCI
            if mean_deviation == 0:
                cci_values.append(0)
            else:
                cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
                cci_values.append(cci)
    
    # 确保返回的数据长度与原始数据一致
    while len(cci_values) < len(df):
        cci_values.append(None)
    
    return cci_values[:len(df)]

def calc_roc(df, period=10):
    """计算ROC（变动率指标）"""
    roc_values = []
    
    for i in range(len(df)):
        if i < period:
            roc_values.append(None)
        else:
            # 计算变动率
            current_price = df['close'].iloc[i]
            past_price = df['close'].iloc[i - period]
            
            # 检查价格是否有效
            if pd.isna(current_price) or pd.isna(past_price) or past_price == 0:
                roc_values.append(0)
            else:
                roc = ((current_price - past_price) / past_price) * 100
                roc_values.append(roc)
    
    # 确保返回的数据长度与原始数据一致
    while len(roc_values) < len(df):
        roc_values.append(None)
    
    return roc_values[:len(df)]

def calc_williams_r(df, period=14):
    """计算Williams %R（威廉指标）"""
    williams_r_values = []
    
    for i in range(len(df)):
        if i < period - 1:
            williams_r_values.append(None)
        else:
            # 获取周期内的最高价和最低价
            high_window = df['high'].iloc[i-period+1:i+1]
            low_window = df['low'].iloc[i-period+1:i+1]
            
            # 检查数据是否有效
            if high_window.isna().any() or low_window.isna().any():
                williams_r_values.append(None)
                continue
                
            highest_high = high_window.max()
            lowest_low = low_window.min()
            
            # 计算Williams %R
            if highest_high == lowest_low:
                williams_r_values.append(-50)  # 避免除零错误
            else:
                williams_r = ((highest_high - df['close'].iloc[i]) / (highest_high - lowest_low)) * -100
                williams_r_values.append(williams_r)
    
    # 确保返回的数据长度与原始数据一致
    while len(williams_r_values) < len(df):
        williams_r_values.append(None)
    
    return williams_r_values[:len(df)]

def calc_kdj(df, n=9, m1=3, m2=3):
    """计算KDJ指标（标准：9,3,3）"""
    # 计算RSV值
    rsv = []
    for i in range(len(df)):
        if i < n - 1:
            rsv.append(None)
        else:
            low_n = df['low'].iloc[i-n+1:i+1].min()
            high_n = df['high'].iloc[i-n+1:i+1].max()
            if high_n == low_n:
                rsv.append(50)
            else:
                rsv.append((df['close'].iloc[i] - low_n) / (high_n - low_n) * 100)
    
    # 计算K值、D值和J值
    k_values = []
    d_values = []
    j_values = []
    
    for i in range(len(rsv)):
        if rsv[i] is None:
            k_values.append(None)
            d_values.append(None)
            j_values.append(None)
        else:
            # 计算K值（快速随机值）
            if i == 0 or k_values[i-1] is None:
                k_values.append(50)  # 初始值设为50
            else:
                k_values.append(k_values[i-1] * 2/3 + rsv[i] * 1/3)
            
            # 计算D值（慢速随机值）
            if i < m1 - 1 or any(k is None for k in k_values[max(0, i-m1+1):i+1]):
                d_values.append(None)
            else:
                d_values.append(sum(k_values[max(0, i-m1+1):i+1]) / min(m1, i+1))
            
            # 计算J值
            if k_values[i] is None or d_values[i] is None:
                j_values.append(None)
            else:
                j_values.append(3 * k_values[i] - 2 * d_values[i])
    
    return k_values, d_values, j_values

def calc_trix(df, period=12):
    """计算TRIX指标（三重指数平滑移动平均）"""
    # 第一次指数平滑
    ema1 = df['close'].ewm(span=period, adjust=False).mean()
    
    # 第二次指数平滑
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    
    # 第三次指数平滑
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    
    # 计算TRIX值
    trix_values = []
    for i in range(len(df)):
        if i == 0 or pd.isna(ema3.iloc[i]) or pd.isna(ema3.iloc[i-1]):
            trix_values.append(None)
        else:
            # TRIX = (当日三重指数平滑值 - 前一日三重指数平滑值) / 前一日三重指数平滑值 * 100
            trix = (ema3.iloc[i] - ema3.iloc[i-1]) / ema3.iloc[i-1] * 100
            trix_values.append(trix)
    
    # 计算TRIX的移动平均线（信号线）
    trix_signal = []
    for i in range(len(trix_values)):
        if i < period - 1 or any(x is None for x in trix_values[max(0, i-period+1):i+1]):
            trix_signal.append(None)
        else:
            signal = sum(trix_values[max(0, i-period+1):i+1]) / min(period, i+1)
            trix_signal.append(signal)
    
    return trix_values, trix_signal

def calc_bbi(df):
    """计算BBI指标（多空指数） - 常用配置：BBI = (MA3 + MA6 + MA12 + MA24) / 4"""
    # 计算不同周期的移动平均线
    ma3 = df['close'].rolling(3).mean()
    ma6 = df['close'].rolling(6).mean()
    ma12 = df['close'].rolling(12).mean()
    ma24 = df['close'].rolling(24).mean()
    
    # 计算BBI值
    bbi_values = []
    for i in range(len(df)):
        if pd.isna(ma3.iloc[i]) or pd.isna(ma6.iloc[i]) or pd.isna(ma12.iloc[i]) or pd.isna(ma24.iloc[i]):
            bbi_values.append(None)
        else:
            bbi = (ma3.iloc[i] + ma6.iloc[i] + ma12.iloc[i] + ma24.iloc[i]) / 4
            bbi_values.append(bbi)
    
    return bbi_values

def calc_zigzag(df, threshold_percent=5.0):
    """计算ZigZag指标，用于识别波段拐点（参数：5%或7%）"""
    if len(df) < 3:
        return [None] * len(df)
    
    zigzag_values = [None] * len(df)
    
    # 寻找第一个有效的高低点
    start_idx = 0
    while start_idx < len(df) - 1:
        if df['high'].iloc[start_idx] is not None and df['low'].iloc[start_idx] is not None:
            break
        start_idx += 1
    
    if start_idx >= len(df) - 1:
        return zigzag_values
    
    # 初始化第一个点
    last_extreme_idx = start_idx
    last_extreme_value = df['high'].iloc[start_idx]
    last_extreme_type = 'high'  # 'high' 或 'low'
    
    zigzag_values[start_idx] = last_extreme_value
    
    # 添加回溯验证的缓冲区
    potential_highs = []
    potential_lows = []
    lookback_window = 5  # 回溯窗口大小
    
    for i in range(start_idx + 1, len(df)):
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        
        if current_high is None or current_low is None:
            continue
        
        # 记录潜在的高点和低点
        if i >= lookback_window:
            # 检查当前点是否是局部高点
            is_local_high = True
            for j in range(max(start_idx, i - lookback_window), i):
                if df['high'].iloc[j] > current_high:
                    is_local_high = False
                    break
            if is_local_high:
                potential_highs.append((i, current_high))
            
            # 检查当前点是否是局部低点
            is_local_low = True
            for j in range(max(start_idx, i - lookback_window), i):
                if df['low'].iloc[j] < current_low:
                    is_local_low = False
                    break
            if is_local_low:
                potential_lows.append((i, current_low))
        
        # 计算相对于上一个极点的变化百分比
        change_percent = abs((current_high - last_extreme_value) / last_extreme_value * 100)
        
        if last_extreme_type == 'high':
            # 寻找低点 - 使用回溯验证
            if potential_lows and potential_lows[-1][0] == i:
                potential_low_idx, potential_low = potential_lows[-1]
                low_change_percent = abs((last_extreme_value - potential_low) / last_extreme_value * 100)
                
                if low_change_percent >= threshold_percent:
                    # 验证这个低点是否有效（比后续几个点都低）
                    is_valid_low = True
                    for j in range(i + 1, min(len(df), i + 3)):  # 检查后续2个点
                        if df['low'].iloc[j] < potential_low:
                            is_valid_low = False
                            break
                    
                    if is_valid_low:
                        zigzag_values[last_extreme_idx] = None  # 移除上一个极值点
                        zigzag_values[i] = potential_low
                        last_extreme_idx = i
                        last_extreme_value = potential_low
                        last_extreme_type = 'low'
                        # 清空潜在低点列表
                        potential_lows = []
        else:
            # 寻找高点 - 使用回溯验证
            if potential_highs and potential_highs[-1][0] == i:
                potential_high_idx, potential_high = potential_highs[-1]
                high_change_percent = abs((potential_high - last_extreme_value) / last_extreme_value * 100)
                
                if high_change_percent >= threshold_percent:
                    # 验证这个高点是否有效（比后续几个点都高）
                    is_valid_high = True
                    for j in range(i + 1, min(len(df), i + 3)):  # 检查后续2个点
                        if df['high'].iloc[j] > potential_high:
                            is_valid_high = False
                            break
                    
                    if is_valid_high:
                        zigzag_values[last_extreme_idx] = None
                        zigzag_values[i] = potential_high
                        last_extreme_idx = i
                        last_extreme_value = potential_high
                        last_extreme_type = 'high'
                        # 清空潜在高点列表
                        potential_highs = []
    
    return zigzag_values

def calc_pivot_points(df):
    """计算PIVOT枢轴点（支撑阻力位）"""
    pivot_points = []
    
    for i in range(len(df)):
        if i == 0:
            pivot_points.append({
                'pivot': None,
                'resistance1': None,
                'resistance2': None,
                'support1': None,
                'support2': None
            })
            continue
        
        # 获取前一日的高、低、收盘价
        prev_high = df['high'].iloc[i-1]
        prev_low = df['low'].iloc[i-1]
        prev_close = df['close'].iloc[i-1]
        
        if pd.isna(prev_high) or pd.isna(prev_low) or pd.isna(prev_close):
            pivot_points.append({
                'pivot': None,
                'resistance1': None,
                'resistance2': None,
                'support1': None,
                'support2': None
            })
            continue
        
        # 计算枢轴点
        pivot = (prev_high + prev_low + prev_close) / 3
        
        # 计算阻力位
        resistance1 = 2 * pivot - prev_low
        resistance2 = pivot + (prev_high - prev_low)
        
        # 计算支撑位
        support1 = 2 * pivot - prev_high
        support2 = pivot - (prev_high - prev_low)
        
        pivot_points.append({
            'pivot': pivot,
            'resistance1': resistance1,
            'resistance2': resistance2,
            'support1': support1,
            'support2': support2
        })
    
    return pivot_points

def calc_donchian_channel(df, period=20):
    """计算Donchian Channel唐奇安通道"""
    dc_high = []
    dc_low = []
    
    for i in range(len(df)):
        if i < period - 1:
            dc_high.append(None)
            dc_low.append(None)
        else:
            # 计算周期内的最高价和最低价
            high_window = df['high'].iloc[i-period+1:i+1]
            low_window = df['low'].iloc[i-period+1:i+1]
            
            # 检查数据是否有效
            if high_window.isna().any() or low_window.isna().any():
                dc_high.append(None)
                dc_low.append(None)
            else:
                dc_high.append(high_window.max())
                dc_low.append(low_window.min())
    
    return dc_high, dc_low

# ---------- API ----------
@app.get("/api/stock_info")
async def stock_info(symbol: str):
    """获取股票基本信息，包括中文名称"""
    try:

        #symbol 要去掉sh sz
        #symbol = symbol.replace('sh', '').replace('sz', '')
        # 使用akshare获取股票基本信息 - 使用不同的函数
        # stock_info_df = ak.stock_individual_info_em(symbol=symbol) 东方财富在pc下不通
        #使用 雪球获取股票基本信息
        stock_info_df =ak.stock_individual_basic_info_xq(symbol)

        
               
        # 检查数据是否有效
        if stock_info_df is None or stock_info_df.empty:
            # 尝试使用其他方法获取股票名称
            try:
                # 使用股票代码查询实时行情，获取名称
                realtime_df = ak.stock_zh_a_spot_em()
                if realtime_df is not None and not realtime_df.empty:
                    stock_row = realtime_df[realtime_df['代码'] == symbol.replace('sh', '').replace('sz', '')]
                    if not stock_row.empty:
                        stock_name = stock_row['名称'].iloc[0]
                    else:
                        stock_name = symbol
                else:
                    stock_name = symbol
            except:
                stock_name = symbol
        else:
            # 从原始数据中提取股票名称
            try:
                # 检查数据格式，尝试不同的列名
                if 'item' in stock_info_df.columns and 'value' in stock_info_df.columns:
                    name_row = stock_info_df[stock_info_df['item'] == 'org_short_name_cn']
                    if not name_row.empty:
                        stock_name = name_row['value'].iloc[0]
                    else:
                        stock_name = symbol
                else:
                    # 尝试直接获取第一行的名称信息
                    stock_name = stock_info_df.iloc[0, 0] if len(stock_info_df.columns) > 0 else symbol
            except:
                stock_name = symbol
        
        return JSONResponse(content={
            "data": {
                "symbol": symbol,
                "name": stock_name
            }
        })
    
    except Exception as e:
        # 如果所有方法都失败，返回默认信息
        return JSONResponse(content={
            "data": {
                "symbol": symbol,
                "name": symbol  # 使用代码作为名称
            }
        })


@app.get("/api/kline")
async def kline(symbol: str, period: str = "daily", limit: int = 1000, include_realtime: bool = False):
    try:
        if period == "daily":
            df = ak.stock_zh_a_daily(symbol=symbol,adjust="qfq")
        elif period == "weekly":
            df = ak.stock_zh_a_weekly(symbol=symbol,adjust="qfq")
        else:
            return JSONResponse(content={"error": "period must be daily or weekly"}, status_code=400)

        df = df.tail(limit).copy()
        
        # 调试：打印原始数据长度和基本信息
        print(f"原始数据长度: {len(df)}")
        print(f"股票代码: {symbol}")
        print(f"包含实时数据: {include_realtime}")
        
        # 如果需要包含实时数据
        if include_realtime and period == "daily":
            try:
                import aiohttp
                import datetime
                
                # 获取当日实时数据
                clean_symbol = symbol.lower()
                url = f"http://qt.gtimg.cn/q={clean_symbol}"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            # 解析腾讯接口数据
                            data_match = re.search(r'v_[^=]+="([^"]+)"', content)
                            if data_match:
                                data_str = data_match.group(1)
                                data_parts = data_str.split('~')
                                
                                if len(data_parts) >= 40:
                                    # 获取当前日期
                                    today = datetime.datetime.now().strftime('%Y-%m-%d')
                                    
                                    # 解析实时数据
                                    current_price = float(data_parts[3]) if data_parts[3] else None
                                    open_price = float(data_parts[5]) if data_parts[5] else None
                                    high_price = float(data_parts[33]) if data_parts[33] else current_price
                                    low_price = float(data_parts[34]) if data_parts[34] else current_price
                                    volume = float(data_parts[6]) * 100 if data_parts[6] else 0  # 转换为股数
                                    
                                    # 检查是否有当日数据，避免重复
                                    has_today_data = False
                                    if 'date' in df.columns:
                                        last_date = df['date'].iloc[-1] if len(df) > 0 else None
                                        if last_date and str(last_date).startswith(today):
                                            has_today_data = True
                                    
                                    # 如果当前价格有效且没有当日数据，则添加实时数据
                                    if current_price and not has_today_data:
                                        # 创建当日数据行
                                        today_data = {
                                            'date': today,
                                            'open': open_price if open_price else current_price,
                                            'high': high_price if high_price else current_price,
                                            'low': low_price if low_price else current_price,
                                            'close': current_price,
                                            'volume': volume
                                        }
                                        
                                        # 添加到DataFrame
                                        today_df = pd.DataFrame([today_data])
                                        df = pd.concat([df, today_df], ignore_index=True)
                                        
                                        print(f"已添加当日实时数据: {today} - 价格: {current_price}")
                                    else:
                                        print(f"跳过添加当日数据 - 已有当日数据: {has_today_data}, 当前价格: {current_price}")
                            
            except Exception as e:
                print(f"获取实时数据失败: {e}")
                # 实时数据获取失败不影响历史数据返回
        
        df = df.reset_index()

        # 检查列名是否存在
        col_mapping = {}
        if '日期' in df.columns:
            col_mapping['日期'] = 'date'
        if '开盘' in df.columns:
            col_mapping['开盘'] = 'open'
        if '收盘' in df.columns:
            col_mapping['收盘'] = 'close'
        if '最高' in df.columns:
            col_mapping['最高'] = 'high'
        if '最低' in df.columns:
            col_mapping['最低'] = 'low'
        if '成交量' in df.columns:
            col_mapping['成交量'] = 'volume'
        if '换手率' in df.columns:
            col_mapping['换手率'] = 'turnover_rate'
        if 'turnover' in df.columns:
            col_mapping['turnover'] = 'turnover_rate'

        df.rename(columns=col_mapping, inplace=True)

        df = calc_indicators(df)
        
        # 调试：检查策略信号列
        td9_signals = df['td9_signal'].tolist()
        ema_signals = df['ema_signal'].tolist()
        super_band_signals = df['super_band_signal'].tolist()
        
        print(f"TD9信号非零数量: {sum(1 for x in td9_signals if x != 0)}")
        print(f"EMA信号非零数量: {sum(1 for x in ema_signals if x != 0)}")
        print(f"超级波段信号非零数量: {sum(1 for x in super_band_signals if x != 0)}")
        
        # 如果有非零信号，打印具体信息
        for i, (td9, ema, sb) in enumerate(zip(td9_signals, ema_signals, super_band_signals)):
            if td9 != 0 or ema != 0 or sb != 0:
                print(f"第{i}行: TD9={td9}, EMA={ema}, 超级波段={sb}")

        # 把日期转成字符串
        if 'date' in df.columns:
            df['date'] = df['date'].astype(str)

        # 计算成交额（收盘价 * 成交量）
        df['turnover'] = df['close'] * df['volume']
        
        # 处理换手率数据
        # 优先使用akshare接口提供的换手率数据
        if 'turnover_rate' not in df.columns:
            # 如果akshare接口没有提供换手率，使用估算方法作为备选
            # 方法1：使用平均成交量估算总股本
            avg_volume = df['volume'].mean()
            # 假设换手率在0.1%到10%之间，估算总股本
            estimated_capital = avg_volume / 0.01  # 假设平均换手率为1%
            
            # 方法2：根据股票代码估算（大盘股总股本较大，小盘股较小）
            # 6开头的是上证（大盘股较多），0、3开头的是深证（中小盘股较多）
            if symbol.startswith('sh6') or symbol.startswith('sz000'):
                # 大盘股，总股本较大（几十亿到几百亿）
                base_capital = 1000000000  # 10亿股
            else:
                # 中小盘股，总股本较小（几亿到几十亿）
                base_capital = 500000000   # 5亿股
            
            # 结合两种方法，取较小值避免换手率过高
            estimated_capital = min(estimated_capital, base_capital)
            
            # 确保总股本不为零
            if estimated_capital <= 0:
                estimated_capital = 100000000  # 默认1亿股
            
            # 计算估算换手率
            df['turnover_rate'] = (df['volume'] / estimated_capital * 100).round(2)
        else:
            # 如果akshare接口提供了换手率数据，确保格式正确
            # 有些接口的换手率可能是百分比格式（如0.23表示0.23%），需要转换为百分比数值
            if df['turnover_rate'].max() < 1:
                # 如果最大值小于1，可能是百分比格式，需要乘以100
                df['turnover_rate'] = (df['turnover_rate'] * 100).round(2)
            elif df['turnover_rate'].max() > 100:
                # 如果最大值大于100，可能是原始数值，需要转换为百分比
                df['turnover_rate'] = df['turnover_rate'].round(2)
        
        # 添加昨收价字段（前一天的收盘价）
        df['prev_close'] = df['close'].shift(1)
        # 第一天的昨收价用当天的开盘价代替
        df.loc[0, 'prev_close'] = df.loc[0, 'open']

        data = df[['date','open','high','low','close','prev_close','volume','turnover','turnover_rate','ma5','ma10','ma20','ma14','ma21','ma35','ma50','ma100','ma200',
                   'bb_upper','bb_mid','bb_lower','macd_diff','macd_signal','macd_hist','rsi14','stoch_rsi_k','stoch_rsi_d','supertrend',
                   'skdj_k','skdj_d','td9_signal','ema_signal','super_band_signal','mk_resonance','obv','mfi14',
                   'cci20','cci14','roc10','roc20','williams_r14','kdj_k','kdj_d','kdj_j','trix','trix_signal','bbi',
                   'zigzag_5','zigzag_7','pivot','resistance1','resistance2','support1','support2','dc_high20','dc_low20']].to_dict(orient='records')
        return JSONResponse(content={"data": clean_nan(data)})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/realtime_price")
async def realtime_price(symbol: str):
    """获取单个股票的实时价格数据"""
    try:
        import aiohttp
        
        # 处理股票代码格式（腾讯接口需要小写sh/sz前缀）
        clean_symbol = symbol.lower()
        
        # 构建腾讯接口URL
        url = f"http://qt.gtimg.cn/q={clean_symbol}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return JSONResponse(content={"error": "获取实时数据失败"}, status_code=500)
                
                content = await response.text()
                
                # 解析腾讯接口返回的数据
                # 格式示例：v_sz000858="51~五粮液液~000858~116.83~117.65~117.60~89756~33514~56224~116.83~21~116.82~47~116.81~100~116.80~201~116.79~82~116.84~31~116.85~44~116.86~35~116.87~53~116.88~26~~20251202135315~-0.82~-0.70~117.77~116.70~116.83/89756/1050761012~89756~105076~0.23~15.95~~117.77~116.70~0.91~4534.69~4534.88~3.18~129.42~105.89~1.16~262~117.07~15.81~14.24~~~0.77~105076.1012~0.0000~0~ ~GP-A~-13.00~-1.42~4.92~19.95~16.27~149.14~113.83~-2.93~-0.28~-9.34~3881444512~3881608005~40.94~-9.80~3881444512~~~-17.52~-0.04~~CNY~0~~116.75~354";
                
                # 提取数据部分
                data_match = re.search(r'v_[^=]+="([^"]+)"', content)
                if not data_match:
                    return JSONResponse(content={"error": "数据格式错误"}, status_code=500)
                
                data_str = data_match.group(1)
                data_parts = data_str.split('~')
                
                if len(data_parts) < 40:
                    return JSONResponse(content={"error": "数据不完整"}, status_code=500)
                
                # 解析腾讯接口数据字段
                # 字段说明：
                # 0: 未知
                # 1: 股票名称
                # 2: 股票代码
                # 3: 当前价格
                # 4: 昨收
                # 5: 今开
                # 6: 成交量（手）
                # 7: 外盘
                # 8: 内盘
                # 9: 买一价
                # 10: 买一量
                # ... 其他买卖盘数据
                # 32: 涨跌额
                # 33: 涨跌幅
                # 34: 最高
                # 35: 最低
                
                # 安全解析价格数据，处理可能的复合数据格式
                def safe_float_parse(value):
                    if not value:
                        return 0
                    # 如果包含斜杠，只取第一个部分
                    if '/' in value:
                        value = value.split('/')[0]
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return 0
                
                def safe_int_parse(value):
                    if not value:
                        return 0
                    # 如果包含斜杠，只取第一个部分
                    if '/' in value:
                        value = value.split('/')[0]
                    try:
                        return int(value)
                    except (ValueError, TypeError):
                        return 0
                
                stock_name = data_parts[1]
                current_price = safe_float_parse(data_parts[3])
                close_price = safe_float_parse(data_parts[4])
                open_price = safe_float_parse(data_parts[5])
                volume = safe_int_parse(data_parts[6]) * 100  # 转换为股数
                change_amount = safe_float_parse(data_parts[31])  # 修正：涨跌额在第31个字段
                change = safe_float_parse(data_parts[32])  # 修正：涨跌幅在第32个字段
                high = safe_float_parse(data_parts[33])  # 修正：最高价在第33个字段
                low = safe_float_parse(data_parts[34])  # 修正：最低价在第34个字段
                
                # 获取换手率（腾讯接口第38项）
                turnover_rate = safe_float_parse(data_parts[37])  # 第38项（索引37）为换手率
                    
                # 计算成交额（如果接口不提供，可以估算）
                turnover = current_price * volume if volume > 0 else 0
                
                return JSONResponse(content={
                    "data": {
                        "symbol": symbol,
                        "name": stock_name,
                        "currentPrice": current_price,
                        "change": change,
                        "changeAmount": change_amount,
                        "volume": volume,
                        "turnover": turnover,
                        "turnoverRate": turnover_rate,  # 添加换手率字段
                        "high": high,
                        "low": low,
                        "open": open_price,
                        "close": close_price
                    }
                })
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/realtime_prices")
async def realtime_prices(symbols: str):
    """批量获取多个股票的实时价格数据"""
    try:
        import aiohttp
        
        # 分割股票代码（支持逗号分隔）
        symbol_list = [s.strip() for s in symbols.split(',')]
        
        # 处理股票代码格式（腾讯接口需要小写sh/sz前缀）
        clean_symbols = [symbol.lower() for symbol in symbol_list]
        
        # 构建腾讯接口URL（支持批量查询）
        symbol_param = ','.join(clean_symbols)
        url = f"http://qt.gtimg.cn/q={symbol_param}"
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return JSONResponse(content={"error": "获取实时数据失败"}, status_code=500)
                
                content = await response.text()
                
                # 腾讯接口返回多只股票数据，每只股票一行
                lines = content.strip().split(';')
                
                for line in lines:
                    if not line.strip():
                        continue
                        
                    # 提取数据部分
                    data_match = re.search(r'v_[^=]+="([^"]+)"', line)
                    if not data_match:
                        continue
                    
                    data_str = data_match.group(1)
                    data_parts = data_str.split('~')
                    
                    if len(data_parts) < 40:
                        continue
                    
                    # 解析腾讯接口数据字段
                    stock_code = data_parts[2]  # 股票代码
                    
                    # 查找对应的原始symbol（保持原始大小写格式）
                    original_symbol = None
                    for sym in symbol_list:
                        if sym.lower().replace('sh', '').replace('sz', '') == stock_code:
                            original_symbol = sym
                            break
                    
                    if not original_symbol:
                        continue
                    
                    stock_name = data_parts[1]
                    
                    # 安全解析价格数据，处理可能的复合数据格式
                    def safe_float_parse(value):
                        if not value:
                            return 0
                        # 如果包含斜杠，只取第一个部分
                        if '/' in value:
                            value = value.split('/')[0]
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return 0
                    
                    def safe_int_parse(value):
                        if not value:
                            return 0
                        # 如果包含斜杠，只取第一个部分
                        if '/' in value:
                            value = value.split('/')[0]
                        try:
                            return int(value)
                        except (ValueError, TypeError):
                            return 0
                    
                    current_price = safe_float_parse(data_parts[3])
                    close_price = safe_float_parse(data_parts[4])
                    open_price = safe_float_parse(data_parts[5])
                    volume = safe_int_parse(data_parts[6]) * 100  # 转换为股数
                    change_amount = safe_float_parse(data_parts[31])  # 修正：涨跌额在第31个字段
                    change = safe_float_parse(data_parts[32])  # 修正：涨跌幅在第32个字段
                    high = safe_float_parse(data_parts[33])  # 修正：最高价在第33个字段
                    low = safe_float_parse(data_parts[34])  # 修正：最低价在第34个字段
                    
                    # 获取换手率（腾讯接口第38项）
                    turnover_rate = safe_float_parse(data_parts[37])  # 第38项（索引37）为换手率
                    
                    # 计算成交额
                    turnover = current_price * volume if volume > 0 else 0
                    
                    results.append({
                        "symbol": original_symbol,
                        "name": stock_name,
                        "currentPrice": current_price,
                        "change": change,
                        "changeAmount": change_amount,
                        "volume": volume,
                        "turnover": turnover,
                        "turnoverRate": turnover_rate,  # 添加换手率字段
                        "high": high,
                        "low": low,
                        "open": open_price,
                        "close": close_price
                    })
        
        return JSONResponse(content={"data": results})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)