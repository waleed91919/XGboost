import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import time
from datetime import datetime

# ==============================================================================
# ⚙️ KONFIGURATION
# ==============================================================================
MODEL_PATH = 'xgboost_polymarket.pkl'
SCALER_PATH = 'robust_scaler.pkl'
BINANCE_API = "https://api.binance.com/api/v3/klines"
MEMPOOL_API = "https://mempool.space/api"
THRESHOLD = 0.601

print("🔍 Starting Recent History Diagnosis...")

# ==============================================================================
# 🛠️ DATA FETCHING
# ==============================================================================

def fetch_binance_1000():
    """Holt die letzten 1000 5m-Kerzen von Binance."""
    print("📥 Fetching last 1000 candles (5m) from Binance...")
    params = {"symbol": "BTCUSDT", "interval": "5m", "limit": 1000}
    try:
        response = requests.get(BINANCE_API, params=params, timeout=10)
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

def fetch_current_mempool():
    """Holt die aktuellen Mempool-Daten (letzter Block)."""
    print("📥 Fetching current mempool data...")
    try:
        response = requests.get(f"{MEMPOOL_API}/v1/blocks", timeout=10)
        blocks = response.json()
        latest = blocks[0]
        return {
            'fees': latest['extras']['totalFees'] / 100000000,
            'txs': latest['tx_count'],
            'size': latest['size']
        }
    except Exception as e:
        print(f"❌ Fehler bei Mempool API: {e}")
        return None

# ==============================================================================
# 🔄 FEATURE PIPELINE
# ==============================================================================

def calculate_features(df, mempool_data):
    """Berechnet Features für den kompletten DataFrame."""
    print("🔄 Calculating features...")
    
    # 1. Mikrostruktur
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['date_only'] = df.index.date
    df['vol_price'] = df['typical_price'] * df['volume']
    
    # VWAP
    df['cum_vol'] = df.groupby('date_only')['volume'].cumsum()
    df['cum_vol_price'] = df.groupby('date_only')['vol_price'].cumsum()
    df['vwap'] = df['cum_vol_price'] / df['cum_vol']
    df['dist_to_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    
    # MFI
    typical_price = df['typical_price']
    raw_money_flow = typical_price * df['volume']
    pos_flow = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0), index=df.index)
    neg_flow = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0), index=df.index)
    money_ratio = pos_flow.rolling(window=14).sum() / (neg_flow.rolling(window=14).sum() + 1e-8)
    df['mfi_14'] = 100 - (100 / (1 + money_ratio))
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['dist_to_bb_upper'] = (df['close'] - (sma_20 + 2*std_20)) / (sma_20 + 2*std_20)
    df['dist_to_bb_lower'] = (df['close'] - (sma_20 - 2*std_20)) / (sma_20 - 2*std_20)
    
    # Order Book Imbalance Proxy
    df['buying_pressure'] = (((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8)) * df['volume']
    df['buying_pressure_ema_5'] = df['buying_pressure'].ewm(span=5, adjust=False).mean()
    
    # Returns & Volatility
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    df['volatility_60'] = df['returns'].rolling(window=60).std()
    
    # 2. On-Chain (Hier nutzen wir die aktuellen Daten für die gesamte Historie wie gewünscht)
    df['total_fees_btc'] = mempool_data['fees']
    df['tx_count'] = mempool_data['txs']
    df['avg_block_size'] = mempool_data['size']
    
    fees_1h_sum = df['total_fees_btc'].rolling(window=12, min_periods=1).sum()
    fees_4h_sum = df['total_fees_btc'].rolling(window=48, min_periods=1).sum()
    df['fee_momentum_ratio'] = fees_1h_sum / ((fees_4h_sum / 4) + 1e-8)
    df['tx_1h_sum'] = df['tx_count'].rolling(window=12, min_periods=1).sum()
    df['blocksize_1h_avg'] = df['avg_block_size'].rolling(window=12, min_periods=1).mean()
    
    features = ['dist_to_vwap', 'mfi_14', 'dist_to_bb_upper', 'dist_to_bb_lower', 
                'buying_pressure_ema_5', 'returns', 'volatility_20', 'volatility_60',
                'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg']
    
    return df[features].ffill().fillna(0)

# ==============================================================================
# 🚀 MAIN
# ==============================================================================

def run_diagnosis():
    # 1. Load Artifacts
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("❌ Model or Scaler not found!")
        return
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    model.set_params(device="cpu") # Zwingend CPU
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    # 2. Fetch Data
    df_binance = fetch_binance_1000()
    mempool_data = fetch_current_mempool()
    
    if df_binance is None or mempool_data is None:
        return
        
    # 3. Features & Scaling
    X_raw = calculate_features(df_binance, mempool_data)
    X_scaled = scaler.transform(X_raw.values) # .values um Scikit-Learn Warnung zu vermeiden
    
    # 4. Predictions
    print("🔮 Running model predictions...")
    probas = model.predict_proba(X_scaled)[:, 1]
    
    # 5. Output
    results_df = pd.DataFrame({
        'timestamp': df_binance.index,
        'probability': probas
    })
    
    # Top 5
    top_5 = results_df.sort_values(by='probability', ascending=False).head(5)
    
    print("\n" + "="*50)
    print("📊 TOP 5 HIGHEST PROBABILITIES (LAST 3.5 DAYS)")
    print("="*50)
    for idx, row in top_5.iterrows():
        print(f"🕒 {row['timestamp']} | 📈 Prob: {row['probability']:.2%}")
    
    # Threshold Check
    signals = results_df[results_df['probability'] >= THRESHOLD]
    print("\n" + "="*50)
    print(f"🎯 THRESHOLD CHECK (>={THRESHOLD*100}%)")
    print("="*50)
    if not signals.empty:
        print(f"✅ Found {len(signals)} potential signals!")
        print(f"First Signal: {signals.iloc[0]['timestamp']} ({signals.iloc[0]['probability']:.2%})")
        print(f"Last Signal: {signals.iloc[-1]['timestamp']} ({signals.iloc[-1]['probability']:.2%})")
    else:
        max_prob = results_df['probability'].max()
        print(f"❌ No signals reached the {THRESHOLD*100}% threshold.")
        print(f"Max Probability found: {max_prob:.2%}")
    print("="*50)

if __name__ == "__main__":
    run_diagnosis()
