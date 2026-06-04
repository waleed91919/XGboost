import time
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import requests
import os
from datetime import datetime, timedelta

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
MODEL_PATH = 'xgboost_polymarket.pkl'
SCALER_PATH = 'robust_scaler.pkl'
BINANCE_API = "https://api.binance.com/api/v3/klines"
MEMPOOL_API = "https://mempool.space/api"

# Risk Management
INITIAL_CAPITAL = 10000.0
KELLY_MULTIPLIER = 0.25  # Quarter-Kelly as requested
MAX_BET_PCT = 0.05       # Max 5% per trade to prevent ruin

# Thresholds to test
THRESHOLDS = [0.57, 0.575, 0.58, 0.585, 0.59, 0.595, 0.60, 0.605]

print("🔍 Backtester: Multi-Threshold Matrix (v2) gestartet...")

# ==============================================================================
# 🛠️ DATA & FEATURE FUNCTIONS
# ==============================================================================

def load_artifacts():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    model.set_params(device="cpu")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def fetch_binance_history(limit=1000):
    params = {"symbol": "BTCUSDT", "interval": "5m", "limit": limit}
    try:
        response = requests.get(BINANCE_API, params=params)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'q_vol', 'trades', 'taker_base', 'taker_quote', 'ignore'
        ])
        df['date'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('date', inplace=True)
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].astype(float)
        return df[cols]
    except Exception as e:
        print(f"❌ Fehler bei Binance API: {e}")
        return None

def fetch_mempool_static():
    try:
        resp = requests.get(f"{MEMPOOL_API}/v1/blocks")
        blocks = resp.json()
        total_fees = sum([b['extras']['totalFees'] for b in blocks[:12]]) / 100000000
        total_txs = sum([b['tx_count'] for b in blocks[:12]])
        avg_size = sum([b['size'] for b in blocks[:12]]) / 12
        return total_fees / 12, total_txs / 12, avg_size
    except:
        return 0.01, 2500, 1500000

def calculate_all_features(df_binance, static_onchain):
    df = df_binance.copy()
    fee_avg, tx_avg, size_avg = static_onchain
    
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['date_only'] = df.index.date
    df['vol_price'] = df['typical_price'] * df['volume']
    df['cum_vol'] = df.groupby('date_only')['volume'].cumsum()
    df['cum_vol_price'] = df.groupby('date_only')['vol_price'].cumsum()
    df['vwap'] = df['cum_vol_price'] / df['cum_vol']
    df['dist_to_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    
    tp = df['typical_price']
    raw_flow = tp * df['volume']
    pos_flow = pd.Series(np.where(tp > tp.shift(1), raw_flow, 0), index=df.index)
    neg_flow = pd.Series(np.where(tp < tp.shift(1), raw_flow, 0), index=df.index)
    m_ratio = pos_flow.rolling(14).sum() / (neg_flow.rolling(14).sum() + 1e-8)
    df['mfi_14'] = 100 - (100 / (1 + m_ratio))
    
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['dist_to_bb_upper'] = (df['close'] - (sma20 + 2*std20)) / (sma20 + 2*std20)
    df['dist_to_bb_lower'] = (df['close'] - (sma20 - 2*std20)) / (sma20 - 2*std20)
    
    df['bp'] = (((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8)) * df['volume']
    df['buying_pressure_ema_5'] = df['bp'].ewm(span=5, adjust=False).mean()
    
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_60'] = df['returns'].rolling(60).std()
    
    df['fee_momentum_ratio'] = 1.0
    df['tx_1h_sum'] = tx_avg * 12
    df['blocksize_1h_avg'] = size_avg
    
    features = ['dist_to_vwap', 'mfi_14', 'dist_to_bb_upper', 'dist_to_bb_lower', 
                'buying_pressure_ema_5', 'returns', 'volatility_20', 'volatility_60',
                'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg']
    
    return df[features].dropna(), df['close']

# ==============================================================================
# 📈 BACKTEST EXECUTION
# ==============================================================================

def run_multi_threshold_backtest():
    model, scaler = load_artifacts()
    df_binance = fetch_binance_history(1000)
    if df_binance is None: return
    
    static_onchain = fetch_mempool_static()
    X, prices = calculate_all_features(df_binance, static_onchain)
    
    X_scaled = scaler.transform(X.values)
    probs = model.predict_proba(X_scaled)[:, 1]
    
    matrix_results = []
    
    for threshold in THRESHOLDS:
        current_capital = INITIAL_CAPITAL
        trades_count = 0
        wins = 0
        total_pnl = 0.0
        
        for i in range(len(probs) - 1):
            prob = probs[i]
            if prob >= threshold:
                entry_price = prices.iloc[i]
                exit_price = prices.iloc[i+1]
                
                is_win = exit_price > entry_price
                trades_count += 1
                if is_win: wins += 1
                
                # Quarter-Kelly Simulation
                # b=1 (Even Money Assumption for Test)
                b = 1.0
                q = 1.0 - prob
                kelly_pct = (prob * b - q) / (b + 1e-8)
                bet_amount = current_capital * min(max(kelly_pct * KELLY_MULTIPLIER, 0), MAX_BET_PCT)
                
                pnl = bet_amount * b if is_win else -bet_amount
                current_capital += pnl
                total_pnl += pnl
        
        win_rate = (wins / trades_count * 100) if trades_count > 0 else 0
        matrix_results.append({
            'Threshold (%)': f"{threshold:.1%}",
            'Trades': trades_count,
            'Win-Rate': f"{win_rate:.1%}",
            'Total PnL (USDT)': round(total_pnl, 2),
            'Final Capital': round(current_capital, 2)
        })
        
    # Matrix ausgeben
    df_matrix = pd.DataFrame(matrix_results)
    print("\n" + "="*85)
    print(f"📊 BACKTEST MATRIX (Letzte 1000 Kerzen | Quarter-Kelly: {KELLY_MULTIPLIER})")
    print("="*85)
    print(df_matrix.to_string(index=False))
    print("="*85)
    print(f"💡 Info: Berechnung basiert auf b=1 (Even Money) und Max {MAX_BET_PCT*100}% Risk/Trade.")
    print("="*85)

if __name__ == "__main__":
    run_multi_threshold_backtest()
