import time
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import requests
import os
from datetime import datetime, timedelta

# ==============================================================================
# ⚙️ KONFIGURATION
# ==============================================================================
MODEL_PATH = 'xgboost_polymarket.pkl'
SCALER_PATH = 'robust_scaler.pkl'
LOG_FILE = 'paper_trades_log.csv'
CONFIDENCE_THRESHOLD = 0.55  # TEST-THRESHOLD (Später wieder auf 0.601)
FRACTIONAL_KELLY = 0.2
INITIAL_CAPITAL = 10000.0

# API Endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
BINANCE_API = "https://api.binance.com/api/v3/klines"
MEMPOOL_API = "https://mempool.space/api"

print("🚀 Polymarket Live Paper-Bot (v2 - Fixed History) gestartet...")

# ==============================================================================
# 🛠️ DATA FETCHING & FEATURE ENGINEERING
# ==============================================================================

def load_artifacts():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    # Erzwinge CPU für Inferenz (verhindert DMatrix Fallback & reduziert Latenz bei 1 Zeile)
    model.set_params(device="cpu")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Modell und Scaler geladen.")
    return model, scaler

def fetch_binance_5m():
    """Holt die letzten 100 5m-Kerzen von Binance."""
    params = {"symbol": "BTCUSDT", "interval": "5m", "limit": 100}
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

def fetch_mempool_onchain_history():
    """Holt die letzten 15 Blöcke von Mempool.space für echte Rolling-Windows."""
    try:
        response = requests.get(f"{MEMPOOL_API}/v1/blocks", timeout=10)
        blocks = response.json()
        data = []
        for b in blocks:
            data.append({
                'time': pd.to_datetime(b['timestamp'], unit='s'),
                'fees': b['extras']['totalFees'] / 100000000,
                'txs': b['tx_count'],
                'size': b['size']
            })
        df_onchain = pd.DataFrame(data).set_index('time').sort_index()
        return df_onchain
    except Exception as e:
        print(f"❌ Fehler bei Mempool History: {e}")
        return None

def calculate_live_features(df_binance, df_onchain):
    """Berechnet Features exakt wie im Training."""
    df = df_binance.copy()
    
    # 1. Mikrostruktur
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['date_only'] = df.index.date
    df['vol_price'] = df['typical_price'] * df['volume']
    df['cum_vol'] = df.groupby('date_only')['volume'].cumsum()
    df['cum_vol_price'] = df.groupby('date_only')['vol_price'].cumsum()
    df['vwap'] = df['cum_vol_price'] / df['cum_vol']
    df['dist_to_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    
    typical_price = df['typical_price']
    raw_money_flow = typical_price * df['volume']
    pos_flow = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0), index=df.index)
    neg_flow = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0), index=df.index)
    money_ratio = pos_flow.rolling(window=14).sum() / (neg_flow.rolling(window=14).sum() + 1e-8)
    df['mfi_14'] = 100 - (100 / (1 + money_ratio))
    
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['dist_to_bb_upper'] = (df['close'] - (sma_20 + 2*std_20)) / (sma_20 + 2*std_20)
    df['dist_to_bb_lower'] = (df['close'] - (sma_20 - 2*std_20)) / (sma_20 - 2*std_20)
    
    df['buying_pressure'] = (((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8)) * df['volume']
    df['buying_pressure_ema_5'] = df['buying_pressure'].ewm(span=5, adjust=False).mean()
    
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    df['volatility_60'] = df['returns'].rolling(window=60).std()
    
    # 2. On-Chain Mapping
    df_onchain_resampled = df_onchain.reindex(df.index, method='ffill').fillna(method='bfill').fillna(0)
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
    
    return df[features].iloc[-1:].ffill().fillna(0)

# ==============================================================================
# 🏛️ POLYMARKET API
# ==============================================================================

def get_active_btc_markets():
    params = {"active": "true", "closed": "false", "search": "Bitcoin Price", "limit": 10}
    try:
        response = requests.get(f"{GAMMA_API}/markets", params=params)
        return [m for m in response.json() if "Bitcoin" in m['question']]
    except: return []

def get_live_orderbook(token_id):
    try:
        response = requests.get(f"{CLOB_API}/book", params={"token_id": token_id})
        return response.json()
    except: return None

# ==============================================================================
# 📈 MAIN LOOP
# ==============================================================================

def run_bot():
    model, scaler = load_artifacts()
    capital = INITIAL_CAPITAL
    
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=['timestamp', 'market', 'prob', 'side', 'price', 'liquidity', 'kelly_bet']).to_csv(LOG_FILE, index=False)

    while True:
        now = datetime.now()
        print(f"\n--- ⏱️ Live Check: {now.strftime('%Y-%m-%d %H:%M:%S')} ---")
        
        # 1. Daten holen
        df_binance = fetch_binance_5m()
        df_onchain = fetch_mempool_onchain_history()
        
        if df_binance is None or df_onchain is None:
            time.sleep(10)
            continue
            
        # 2. Features berechnen & Skalieren
        X_live = calculate_live_features(df_binance, df_onchain)
        X_scaled = scaler.transform(X_live.values)
        
        # 3. Modell-Vorhersage
        prob_up = model.predict_proba(X_scaled)[:, 1][0]
        print(f"🔮 Modell Vorhersage (UP): {prob_up:.2%}")
        
        # 4. Polymarket Check
        markets = get_active_btc_markets()
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
                
                # Kelly (EV Check)
                b = (1.0 - best_ask) / best_ask
                q = 1.0 - confidence
                kelly_pct = (confidence * b - q) / (b + 1e-8)
                bet_amount = capital * min(max(kelly_pct * FRACTIONAL_KELLY, 0), 0.10)
                
                if bet_amount > 0 and (size_available * best_ask) >= bet_amount:
                    print(f"🎯 TRADING SIGNAL! {side} @ {best_ask} for {market_question}")
                    log_entry = {'timestamp': datetime.now(), 'market': market_question, 'prob': confidence,
                                 'side': side, 'price': best_ask, 'liquidity': size_available, 'kelly_bet': bet_amount}
                    pd.DataFrame([log_entry]).to_csv(LOG_FILE, mode='a', header=False, index=False)
                    print(f"📝 Paper-Trade geloggt!")
                else:
                    print(f"💤 Signal ({confidence:.1%}) vorhanden, aber Liquidität zu gering oder kein EV.")
            
        # Nächster Check in 30 Sekunden
        time.sleep(30)

if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        print("\n🛑 Bot manuell gestoppt.")
