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
BASE_MODEL_PATH = 'xgboost_polymarket.pkl'
META_MODEL_PATH = 'xgboost_risk_manager.pkl'
KMEANS_MODEL_PATH = 'kmeans_model.pkl'
SCALER_PATH = 'robust_scaler.pkl'
KMEANS_SCALER_PATH = 'kmeans_scaler.pkl'
LOG_FILE = 'paper_trades_log.csv'

# Thresholds aus Backtest-Optimierung
CONFIDENCE_THRESHOLD = 0.580    # Basis-Modell Threshold
META_THRESHOLD = 0.580          # Meta-Modell (Risk Manager) Threshold
REGIME_BLOCK_ID = 3             # Regime ID, die blockiert wird (Panik/Hohe Vola)

FRACTIONAL_KELLY = 0.25         # Quarter-Kelly
INITIAL_CAPITAL = 10000.0

# API Endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
BINANCE_API = "https://api.binance.com/api/v3/klines"
MEMPOOL_API = "https://mempool.space/api"

print("🚀 Polymarket Triple-Check Live Bot (v6 - Regime Filter + AI Risk Manager) gestartet...")

# ==============================================================================
# 🛠️ DATA FETCHING & FEATURE ENGINEERING
# ==============================================================================

def load_artifacts():
    with open(BASE_MODEL_PATH, 'rb') as f:
        base_model = pickle.load(f)
    base_model.set_params(device="cpu") # CPU für Live-Inferenz
    
    with open(META_MODEL_PATH, 'rb') as f:
        meta_model = pickle.load(f)
    meta_model.set_params(device="cpu") # CPU für Live-Inferenz
    
    with open(KMEANS_MODEL_PATH, 'rb') as f:
        kmeans_model = pickle.load(f)
        
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    with open(KMEANS_SCALER_PATH, 'rb') as f:
        kmeans_scaler = pickle.load(f)
        
    print("✅ Basis-Modell, Risk Manager, K-Means und Scaler geladen.")
    return base_model, meta_model, kmeans_model, scaler, kmeans_scaler

def fetch_binance_5m():
    """Holt die letzten 300 5m-Kerzen von Binance (24h+ History)."""
    params = {"symbol": "BTCUSDT", "interval": "5m", "limit": 300}
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
    """Berechnet Features für alle drei Modelle."""
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
    
    # 2. On-Chain Mapping (Resampled)
    df_onchain_resampled = df_onchain.reindex(df.index, method='ffill').bfill().fillna(0)
    df['total_fees_btc'] = df_onchain_resampled['fees']
    df['tx_count'] = df_onchain_resampled['txs']
    df['avg_block_size'] = df_onchain_resampled['size']
    
    fees_1h_sum = df['total_fees_btc'].rolling(window=12, min_periods=1).sum()
    fees_4h_sum = df['total_fees_btc'].rolling(window=48, min_periods=1).sum()
    df['fee_momentum_ratio'] = fees_1h_sum / ((fees_4h_sum / 4) + 1e-8)
    df['tx_1h_sum'] = df['tx_count'].rolling(window=12, min_periods=1).sum()
    df['blocksize_1h_avg'] = df['avg_block_size'].rolling(window=12, min_periods=1).mean()

    # 3. ATR (für Risk Manager & Regime Filter)
    high_low = df['high'] - df['low']
    high_cp = np.abs(df['high'] - df['close'].shift())
    low_cp = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    
    base_features = ['dist_to_vwap', 'mfi_14', 'dist_to_bb_upper', 'dist_to_bb_lower', 
                    'buying_pressure_ema_5', 'returns', 'volatility_20', 'volatility_60',
                    'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg']
    
    regime_features = ['atr_14', 'volume', 'mfi_14', 'volatility_20', 'avg_block_size']
    
    return df.iloc[-1:], base_features, regime_features


# ==============================================================================
# 📊 TRADE RESOLUTION LOGIC
# ==============================================================================

def resolve_trades(current_btc_price):
    """Überprüft offene Trades und rechnet sie ab."""
    if not os.path.exists(LOG_FILE):
        return
        
    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty: return
        
        updated = False
        for i, row in df.iterrows():
            if row['status'] == 'OPEN':
                trade_time = pd.to_datetime(row['timestamp'])
                # Wenn der Trade älter als 5 Minuten ist (Kerze geschlossen)
                if datetime.now() - trade_time > timedelta(minutes=5):
                    entry_btc = float(row['entry_btc_price'])
                    side = row['side']
                    bet_amount = float(row['kelly_bet'])
                    polymarket_entry_price = float(row['price'])
                    
                    b = (1.0 - polymarket_entry_price) / (polymarket_entry_price + 1e-8)
                    
                    is_win = False
                    if side == 'YES' and current_btc_price > entry_btc:
                        is_win = True
                    elif side == 'NO' and current_btc_price < entry_btc:
                        is_win = True
                        
                    if is_win:
                        df.at[i, 'status'] = 'WON'
                        df.at[i, 'pnl'] = bet_amount * b
                    else:
                        df.at[i, 'status'] = 'LOST'
                        df.at[i, 'pnl'] = -bet_amount
                        
                    df.at[i, 'exit_price'] = current_btc_price
                    updated = True
                    print(f"✅ Trade aufgelöst: {row['market']} | {df.at[i, 'status']} | PnL: {df.at[i, 'pnl']:.2f} USDT")
        
        if updated:
            df.to_csv(LOG_FILE, index=False)
            
    except Exception as e:
        print(f"❌ Fehler bei Trade Resolution: {e}")

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
    base_model, meta_model, kmeans_model, scaler, kmeans_scaler = load_artifacts()
    capital = INITIAL_CAPITAL
    
    if not os.path.exists(LOG_FILE):
        cols = ['timestamp', 'market', 'prob', 'meta_prob', 'regime', 'side', 'price', 'liquidity', 'kelly_bet', 'entry_btc_price', 'exit_price', 'status', 'pnl']
        pd.DataFrame(columns=cols).to_csv(LOG_FILE, index=False)

    while True:
        now = datetime.now()
        print(f"\n--- ⏱️ Live Check: {now.strftime('%Y-%m-%d %H:%M:%S')} ---")
        
        # 1. Daten holen
        df_binance = fetch_binance_5m()
        df_onchain = fetch_mempool_onchain_history()
        
        if df_binance is None or df_onchain is None:
            time.sleep(10)
            continue
            
        current_btc_price = df_binance.iloc[-1]['close']
        
        # 2. Trade Resolution
        resolve_trades(current_btc_price)
        
        # 3. Features berechnen
        row_live, base_features, regime_features = calculate_live_features(df_binance, df_onchain)
        
        # 4. Inferenz Stufe 1: Basis-Modell
        X_scaled = scaler.transform(row_live[base_features].values)
        base_prob = base_model.predict_proba(X_scaled)[:, 1][0]
        print(f"🔮 Basis-Modell (UP): {base_prob:.2%}")
        
        # 5. Signal & Risk Management (Triple-Check)
        if base_prob > CONFIDENCE_THRESHOLD:
            
            # Stufe 2: Regime Filter
            X_regime = kmeans_scaler.transform(row_live[regime_features].values)
            regime_id = kmeans_model.predict(X_regime)[0]
            print(f"🌐 Markt-Regime: {regime_id} {'⚠️ (PANIK/BLOCK)' if regime_id == REGIME_BLOCK_ID else '✅ (NORMAL)'}")
            
            if regime_id == REGIME_BLOCK_ID:
                print(f"🛑 Regime-Filter blockiert Trade (Regime {regime_id})")
                time.sleep(30)
                continue
                
            # Stufe 3: Meta-Modell (Risk Manager)
            meta_features = [
                'atr_14', 'volume', 'volatility_20', 'volatility_60', 
                'dist_to_bb_upper', 'dist_to_bb_lower',
                'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg'
            ]
            # Meta-Features vorbereiten (base_prob hinzufügen)
            X_meta_dict = row_live[meta_features].to_dict('records')[0]
            X_meta_dict['base_prob'] = base_prob
            
            # Sortierung der Features für Meta-Modell sicherstellen (muss 1:1 wie im Training sein)
            meta_feature_order = [
                'base_prob', 'atr_14', 'volume', 'volatility_20', 'volatility_60', 
                'dist_to_bb_upper', 'dist_to_bb_lower',
                'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg'
            ]
            X_meta_values = np.array([[X_meta_dict[f] for f in meta_feature_order]])
            
            meta_prob = meta_model.predict_proba(X_meta_values)[:, 1][0]
            print(f"🛡️ Risk Manager Konfidenz: {meta_prob:.2%}")
            
            if meta_prob > META_THRESHOLD:
                # Polymarket Check
                markets = get_active_btc_markets()
                for market in markets:
                    market_question = market['question']
                    token_ids = market.get('clobTokenIds', [])
                    if len(token_ids) < 2: continue
                    
                    token_id, side = token_ids[0], "YES"
                    book = get_live_orderbook(token_id)
                    
                    if book and book.get('asks'):
                        best_ask = float(book['asks'][0]['price'])
                        size_available = float(book['asks'][0]['size'])
                        
                        # Kelly (EV Check)
                        b = (1.0 - best_ask) / (best_ask + 1e-8)
                        q = 1.0 - base_prob
                        kelly_pct = (base_prob * b - q) / (b + 1e-8)
                        bet_amount = capital * min(max(kelly_pct * FRACTIONAL_KELLY, 0), 0.15) # Cap auf 15% erhöht wie im Backtest
                        
                        if bet_amount > 0 and (size_available * best_ask) >= bet_amount:
                            print(f"🎯 TRADING SIGNAL! {side} @ {best_ask} for {market_question}")
                            log_entry = {
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'market': market_question, 
                                'prob': base_prob,
                                'meta_prob': meta_prob,
                                'regime': regime_id,
                                'side': side, 
                                'price': best_ask, 
                                'liquidity': size_available, 
                                'kelly_bet': bet_amount,
                                'entry_btc_price': current_btc_price,
                                'exit_price': 0.0,
                                'status': 'OPEN',
                                'pnl': 0.0
                            }
                            pd.DataFrame([log_entry]).to_csv(LOG_FILE, mode='a', header=False, index=False)
                            print(f"📝 Paper-Trade geloggt! (Entry BTC: {current_btc_price})")
                        else:
                            print(f"💤 Signal vorhanden, aber Liquidität zu gering oder kein EV.")
            else:
                print(f"⚠️ Risk Manager blockiert Trade (Konfidenz {meta_prob:.2%} <= {META_THRESHOLD:.2%})")
        
        # Nächster Check in 30 Sekunden
        time.sleep(30)

if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        print("\n🛑 Bot manuell gestoppt.")
