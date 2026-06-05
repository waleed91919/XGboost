import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

# --- KONFIGURATION ---
INPUT_FILE = 'final_data_ready_for_backtest.csv'
BASE_MODEL_PATH = 'xgboost_polymarket.pkl'
META_MODEL_PATH = 'xgboost_risk_manager.pkl'
KMEANS_MODEL_PATH = 'kmeans_model.pkl'
ROBUST_SCALER_PATH = 'robust_scaler.pkl'
KMEANS_SCALER_PATH = 'kmeans_scaler.pkl'

START_CAPITAL = 10000.0
BASE_THRESHOLD = 0.58
META_THRESHOLD = 0.60
KELLY_FRACTION = 0.25 # Fixed position sizing logic (Quarter-Kelly equivalent at threshold)

def calculate_all_features(df):
    """Berechnet alle Features für Basis-Modell, Meta-Modell und Regime-Clustering."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # 1. Mikrostruktur Features
    df['typical_price'] = (high + low + close) / 3
    df['vwap'] = (df['typical_price'] * volume).cumsum() / (volume.cumsum() + 1e-8)
    df['dist_to_vwap'] = (close - df['vwap']) / (df['vwap'] + 1e-8)

    # MFI
    raw_money_flow = df['typical_price'] * volume
    pos_flow = pd.Series(np.where(df['typical_price'] > df['typical_price'].shift(1), raw_money_flow, 0), index=df.index)
    neg_flow = pd.Series(np.where(df['typical_price'] < df['typical_price'].shift(1), raw_money_flow, 0), index=df.index)
    money_ratio = pos_flow.rolling(window=14).sum() / (neg_flow.rolling(window=14).sum() + 1e-8)
    df['mfi_14'] = 100 - (100 / (1 + money_ratio))

    # Bollinger Bands
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    df['bb_upper'] = sma_20 + 2*std_20
    df['bb_lower'] = sma_20 - 2*std_20
    df['dist_to_bb_upper'] = (close - df['bb_upper']) / (df['bb_upper'] + 1e-8)
    df['dist_to_bb_lower'] = (close - df['bb_lower']) / (df['bb_lower'] + 1e-8)

    # Buying Pressure
    df['clv'] = ((close - low) - (high - close)) / (high - low + 1e-8)
    df['buying_pressure_ema_5'] = (df['clv'] * volume).ewm(span=5, adjust=False).mean()

    # Returns & Vola
    df['returns'] = close.pct_change()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    df['volatility_60'] = df['returns'].rolling(window=60).std()

    # 2. On-Chain Features
    fee_col = 'total_fees_btc'
    tx_col = 'tx_count' if 'tx_count' in df.columns else 'transaction_count'
    block_col = 'avg_block_size'
    
    df['fee_momentum_ratio'] = df[fee_col].rolling(window=12).sum() / ((df[fee_col].rolling(window=48).sum() / 4) + 1e-8)
    df['tx_1h_sum'] = df[tx_col].rolling(window=12).sum()
    df['blocksize_1h_avg'] = df[block_col].rolling(window=12).mean()

    # 3. ATR
    high_low = high - low
    high_cp = np.abs(high - close.shift())
    low_cp = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    
    base_features = [
        'dist_to_vwap', 'mfi_14', 'dist_to_bb_upper', 'dist_to_bb_lower', 
        'buying_pressure_ema_5', 'returns', 'volatility_20', 'volatility_60',
        'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg'
    ]
    
    meta_features = [
        'base_prob', 'atr_14', 'volume', 'volatility_20', 'volatility_60', 
        'dist_to_bb_upper', 'dist_to_bb_lower',
        'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg'
    ]
    
    regime_features = [
        'atr_14', 'volume', 'mfi_14', 'volatility_20', 'avg_block_size'
    ]
    
    df.dropna(subset=base_features + ['atr_14'], inplace=True)
    return df, base_features, meta_features, regime_features

def run_triple_backtest():
    if not all(os.path.exists(p) for p in [BASE_MODEL_PATH, META_MODEL_PATH, KMEANS_MODEL_PATH, ROBUST_SCALER_PATH, KMEANS_SCALER_PATH]):
        print("❌ Fehler: Eines der Modelle oder Scaler fehlt.")
        return

    print(f"📥 Lade Daten und alle Modelle für Triple-Check Backtest...")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    with open(BASE_MODEL_PATH, 'rb') as f: base_model = pickle.load(f)
    with open(META_MODEL_PATH, 'rb') as f: meta_model = pickle.load(f)
    with open(KMEANS_MODEL_PATH, 'rb') as f: kmeans_model = pickle.load(f)
    with open(ROBUST_SCALER_PATH, 'rb') as f: robust_scaler = pickle.load(f)
    with open(KMEANS_SCALER_PATH, 'rb') as f: kmeans_scaler = pickle.load(f)

    df, base_features, meta_features, regime_features = calculate_all_features(df)
    
    # Vorhersagen (Batch für Basis)
    X_base = robust_scaler.transform(df[base_features])
    df['base_prob'] = base_model.predict_proba(X_base)[:, 1]
    
    # Regime-Klassifizierung (Batch)
    X_regime = kmeans_scaler.transform(df[regime_features])
    df['regime_id'] = kmeans_model.predict(X_regime)
    
    capital = START_CAPITAL
    trades = 0
    wins = 0
    blocked_by_meta = 0
    blocked_by_regime = 0
    
    print(f"🚀 Starte Triple-Check Backtest (Base: {BASE_THRESHOLD}, Meta: {META_THRESHOLD}, No-Regime-3)...")
    
    for i in range(len(df) - 1):
        row = df.iloc[i]
        base_prob = row['base_prob']
        
        # Bedingung A: Basis-Modell
        if base_prob > BASE_THRESHOLD:
            
            # Bedingung B: Risk Manager
            X_meta = row[meta_features].values.reshape(1, -1)
            meta_prob = meta_model.predict_proba(X_meta)[0, 1]
            
            if meta_prob < META_THRESHOLD:
                blocked_by_meta += 1
                continue
                
            # Bedingung C: Regime-Filter (Kein Regime 3 / Panik)
            if row['regime_id'] == 3:
                blocked_by_regime += 1
                continue
            
            # Alle Checks bestanden -> TRADE
            # Einsatzberechnung (Quarter-Kelly)
            p = base_prob
            q = 1.0 - p
            kelly_f = max(0, p - q)
            risk_fraction = kelly_f * KELLY_FRACTION
            risk_fraction = min(risk_fraction, 0.15) # Safety Cap
            
            bet_amount = capital * risk_fraction
            cost_per_share = p
            
            # Ergebnis
            entry_price = row['close']
            exit_price = df['close'].iloc[i+1]
            is_win = exit_price > entry_price
            
            if is_win:
                trade_pnl = bet_amount * (1.0 - cost_per_share) / (cost_per_share + 1e-8)
                wins += 1
            else:
                trade_pnl = -bet_amount
            
            capital += trade_pnl
            trades += 1

    # Auswertung
    total_pnl_pct = (capital - START_CAPITAL) / START_CAPITAL * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0

    print("\n" + "="*65)
    print("📊 ERGEBNIS: TRIPLE-CHECK SYSTEM BACKTEST")
    print("="*65)
    print(f"Startkapital:        {START_CAPITAL:,.2f} USDT")
    print(f"Endkapital:          {capital:,.2f} USDT")
    print(f"Total PnL:           {total_pnl_pct:+.2f}%")
    print(f"Win-Rate:            {win_rate:.2f}% ({wins}/{trades} Trades)")
    print("-" * 65)
    print(f"Blockiert (Risk Manager): {blocked_by_meta}")
    print(f"Blockiert (Regime 3):     {blocked_by_regime}")
    print("="*65)

if __name__ == "__main__":
    run_triple_backtest()
