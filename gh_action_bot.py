import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import requests
import os
from datetime import datetime

# ==============================================================================
# ⚙️ KONFIGURATION
# ==============================================================================
MODEL_PATH = 'xgboost_polymarket.pkl'
SCALER_PATH = 'robust_scaler.pkl'
LOG_FILE = 'paper_trades_log.csv'
CONFIDENCE_THRESHOLD = 0.601  # Optuna Validierter Threshold
FRACTIONAL_KELLY = 0.2
INITIAL_CAPITAL = 10000.0

# API Endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
BINANCE_API = "https://api.binance.com/api/v3/klines"
MEMPOOL_API = "https://mempool.space/api"

# ==============================================================================
# 🛠️ DATA FETCHING & FEATURE ENGINEERING
# ==============================================================================

def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Modell oder Scaler Datei fehlt im Repository!")
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # FIX: GitHub Actions haben keine GPU -> Erzwungene CPU Inferenz
    model.set_params(device="cpu")
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def fetch_binance_5m():
    params = {"symbol": "BTCUSDT", "interval": "5m", "limit": 100}
    response = requests.get(BINANCE_API, params=params, timeout=15)
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

def fetch_mempool_onchain_history():
    response = requests.get(f"{MEMPOOL_API}/v1/blocks", timeout=15)
    blocks = response.json()
    data = []
    for b in blocks:
        data.append({
            'time': pd.to_datetime(b['timestamp'], unit='s'),
            'fees': b['extras']['totalFees'] / 100000000,
            'txs': b['tx_count'],
            'size': b['size']
        })
    return pd.DataFrame(data).set_index('time').sort_index()

def calculate_live_features(df_binance, df_onchain):
    """Berechnet Features exakt wie im Training."""
    df = df_binance.copy()
    
    # 1. Mikrostruktur
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['date_only'] = df.index.date
    df['vol_price'] = df['typical_price'] * df['volume']
    df['cum_vol'] = df.groupby('date_only')['volume'].cumsum()
    df['cum_vol_price'] = df.groupby('date_only')['vol_price'].cumsum()
    df['vwap'] = df['cum_vol_price'] / (df['cum_vol'] + 1e-8)
    df['dist_to_vwap'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-8)
    
    typical_price = df['typical_price']
    raw_money_flow = typical_price * df['volume']
    pos_flow = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0), index=df.index)
    neg_flow = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0), index=df.index)
    money_ratio = pos_flow.rolling(window=14).sum() / (neg_flow.rolling(window=14).sum() + 1e-8)
    df['mfi_14'] = 100 - (100 / (1 + money_ratio))
    
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['dist_to_bb_upper'] = (df['close'] - (sma_20 + 2*std_20)) / (sma_20 + 2*std_20 + 1e-8)
    df['dist_to_bb_lower'] = (df['close'] - (sma_20 - 2*std_20)) / (sma_20 - 2*std_20 + 1e-8)
    
    df['buying_pressure'] = (((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8)) * df['volume']
    df['buying_pressure_ema_5'] = df['buying_pressure'].ewm(span=5, adjust=False).mean()
    
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    df['volatility_60'] = df['returns'].rolling(window=60).std()
    
    # 2. On-Chain Mapping
    df_onchain_resampled = df_onchain.reindex(df.index, method='ffill').bfill().fillna(0)
    df['total_fees_btc'] = df_onchain_resampled['fees']
    df['tx_count'] = df_onchain_resampled['txs']
    df['avg_block_size'] = df_onchain_resampled['size']
    
    fees_1h_sum = df['total_fees_btc'].rolling(window=12, min_periods=1).sum()
    fees_4h_sum = df['total_fees_btc'].rolling(window=48, min_periods=1).sum()
    df['fee_momentum_ratio'] = fees_1h_sum / ((fees_4h_sum / 4) + 1e-8)
    df['tx_1h_sum'] = df['tx_count'].rolling(window=12, min_periods=1).sum()
    df['blocksize_1h_avg'] = df['avg_block_size'].rolling(window=12, min_periods=1).mean()
    
    features = ['dist_to_vwap', 'mfi_14', 'dist_to_bb_upper', 'dist_to_bb_lower', 
                'buying_pressure_ema_5', 'returns', 'volatility_20', 'volatility_60',
                'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg']
    
    # Sicherstellen, dass wir keine NaN-Werte haben und NAs füllen
    df_feats = df[features].ffill().dropna()
    
    print(f"📊 Debug: Feature DataFrame Shape {df_feats.shape}")
    if not df_feats.empty:
        print(f"📊 Letzter Zeitstempel im Feature-Set: {df_feats.index[-1]}")
    
    if df_feats.empty:
        print("⚠️ Feature DataFrame ist nach dropna() leer.")
        return None
        
    return df_feats.tail(1) # Sicherere Methode als iloc[-1:] falls wir nur 1 Zeile haben

# ==============================================================================
# 🏛️ POLYMARKET API
# ==============================================================================

def get_active_btc_markets():
    params = {"active": "true", "closed": "false", "search": "Bitcoin Price", "limit": 10}
    try:
        response = requests.get(f"{GAMMA_API}/markets", params=params, timeout=15)
        return [m for m in response.json() if "Bitcoin" in m['question']]
    except: return []

def get_live_orderbook(token_id):
    try:
        response = requests.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=15)
        return response.json()
    except: return None

# ==============================================================================
# 🚀 EXECUTION (SINGLE RUN)
# ==============================================================================

def main():
    print(f"--- ⏱️ GitHub Action Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    # 1. Init
    try:
        model, scaler = load_artifacts()
    except Exception as e:
        print(f"❌ Fehler beim Laden der Artefakte: {e}")
        return

    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=['timestamp', 'market', 'prob', 'side', 'price', 'liquidity', 'kelly_bet']).to_csv(LOG_FILE, index=False)

    # 2. Daten holen
    try:
        df_binance = fetch_binance_5m()
        df_onchain = fetch_mempool_onchain_history()
    except Exception as e:
        print(f"❌ Fehler beim Datenabruf: {e}")
        return
    
    if df_binance is None or df_onchain is None or len(df_binance) < 60:
        print("⚠️ Nicht genügend Daten für Feature-Engineering.")
        return

    # 3. Features & Inferenz
    X_live = calculate_live_features(df_binance, df_onchain)
    
    if X_live is None or X_live.empty:
        print("⚠️ Feature DataFrame ist leer.")
        return

    print(f"📊 Feature-Check: Shape {X_live.shape}")
    
    X_scaled = scaler.transform(X_live.values)
    prob_up = model.predict_proba(X_scaled)[:, 1][0]
    print(f"🔮 Modell Vorhersage (UP): {prob_up:.2%}")

    # 4. Polymarket Check
    markets = get_active_btc_markets()
    found_trade = False
    
    for market in markets:
        market_question = market['question']
        token_ids = market.get('clobTokenIds', [])
        if len(token_ids) < 2: continue
        
        if prob_up >= CONFIDENCE_THRESHOLD:
            token_id, side, confidence = token_ids[0], "YES", prob_up
        elif (1-prob_up) >= CONFIDENCE_THRESHOLD:
            token_id, side, confidence = token_ids[1], "NO", 1 - prob_up
        else: continue
            
        book = get_live_orderbook(token_id)
        if book and book.get('asks'):
            best_ask = float(book['asks'][0]['price'])
            size_available = float(book['asks'][0]['size'])
            
            # Kelly
            b = (1.0 - best_ask) / (best_ask + 1e-8)
            q = 1.0 - confidence
            kelly_pct = (confidence * b - q) / (b + 1e-8)
            bet_amount = INITIAL_CAPITAL * min(max(kelly_pct * FRACTIONAL_KELLY, 0), 0.10)
            
            if bet_amount > 0 and (size_available * best_ask) >= bet_amount:
                print(f"🎯 SIGNAL! {side} @ {best_ask} for {market_question}")
                log_entry = {'timestamp': datetime.now(), 'market': market_question, 'prob': confidence,
                             'side': side, 'price': best_ask, 'liquidity': size_available, 'kelly_bet': bet_amount}
                pd.DataFrame([log_entry]).to_csv(LOG_FILE, mode='a', header=False, index=False)
                found_trade = True

    if not found_trade:
        print("😴 Keine passenden Trading-Gelegenheiten gefunden.")

if __name__ == "__main__":
    main()
