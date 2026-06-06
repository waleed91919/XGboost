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
REGIME_BLOCK_ID = 3

START_CAPITAL = 100.0
BET_FRACTION = 0.10 # 10% des aktuellen Gesamtkapitals

STRATEGIES = [
    {"name": "Aggressiv (10% Risk)", "base_thresh": 0.57, "meta_thresh": 0.58}
]

def calculate_all_features(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Mikrostruktur
    df['typical_price'] = (high + low + close) / 3
    df['date_only'] = df.index.date
    daily_groups = df.groupby('date_only')
    df['cum_vol'] = daily_groups['volume'].cumsum()
    df['cum_vol_price'] = daily_groups.apply(lambda x: (x['typical_price'] * x['volume']).cumsum()).reset_index(level=0, drop=True)
    df['vwap'] = df['cum_vol_price'] / (df['cum_vol'] + 1e-8)
    df['dist_to_vwap'] = (close - df['vwap']) / (df['vwap'] + 1e-8)

    # MFI
    raw_money_flow = df['typical_price'] * volume
    pos_flow = pd.Series(np.where(df['typical_price'] > df['typical_price'].shift(1), raw_money_flow, 0), index=df.index)
    neg_flow = pd.Series(np.where(df['typical_price'] < df['typical_price'].shift(1), raw_money_flow, 0), index=df.index)
    money_ratio = pos_flow.rolling(window=14).sum() / (neg_flow.rolling(window=14).sum() + 1e-8)
    df['mfi_14'] = 100 - (100 / (1 + money_ratio))

    # BB
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

    # On-Chain
    fee_col = 'total_fees_btc'
    tx_col = 'tx_count' if 'tx_count' in df.columns else 'transaction_count'
    block_col = 'avg_block_size'
    
    df['fee_momentum_ratio'] = df[fee_col].rolling(window=12).sum() / ((df[fee_col].rolling(window=48).sum() / 4) + 1e-8)
    df['tx_1h_sum'] = df[tx_col].rolling(window=12).sum()
    df['blocksize_1h_avg'] = df[block_col].rolling(window=12).mean()

    # ATR
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

def main():
    print("📥 Lade Daten und Modelle...")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    with open(BASE_MODEL_PATH, 'rb') as f: base_model = pickle.load(f)
    with open(META_MODEL_PATH, 'rb') as f: meta_model = pickle.load(f)
    with open(KMEANS_MODEL_PATH, 'rb') as f: kmeans_model = pickle.load(f)
    with open(ROBUST_SCALER_PATH, 'rb') as f: robust_scaler = pickle.load(f)
    with open(KMEANS_SCALER_PATH, 'rb') as f: kmeans_scaler = pickle.load(f)

    df, base_features, meta_features, regime_features = calculate_all_features(df)
    
    X_base = robust_scaler.transform(df[base_features])
    df['base_prob'] = base_model.predict_proba(X_base)[:, 1]
    
    X_regime = kmeans_scaler.transform(df[regime_features])
    df['regime_id'] = kmeans_model.predict(X_regime)
    
    X_meta = df[meta_features].values
    df['meta_prob'] = meta_model.predict_proba(X_meta)[:, 1]

    print(f"\n🚀 Starte Compounding Simulation mit Startkapital {START_CAPITAL} USDT...\n")

    for strat in STRATEGIES:
        capital = START_CAPITAL
        min_capital = START_CAPITAL
        trades = 0
        wins = 0
        
        for i in range(len(df) - 1):
            row = df.iloc[i]
            
            if row['base_prob'] > strat['base_thresh']:
                if row['regime_id'] != REGIME_BLOCK_ID:
                    if row['meta_prob'] >= strat['meta_thresh']:
                        
                        bet_amount = capital * BET_FRACTION
                        cost_per_share = row['base_prob']
                        is_win = df['close'].iloc[i+1] > row['close']
                        
                        if is_win:
                            profit = bet_amount * (1.0 - cost_per_share) / (cost_per_share + 1e-8)
                            capital += profit
                            wins += 1
                        else:
                            capital -= bet_amount
                            
                        trades += 1
                        if capital < min_capital:
                            min_capital = capital
                            
        net_profit = capital - START_CAPITAL
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        print(f"==================================================")
        print(f"📊 Strategie: {strat['name']} (Base {strat['base_thresh']} / Meta {strat['meta_thresh']})")
        print(f"   Trades gemacht     : {trades} ({win_rate:.2f}% Win-Rate)")
        print(f"   Endkapital         : {capital:.2f} USDT")
        print(f"   Reingewinn         : {net_profit:+.2f} USDT")
        print(f"   Min. Kontostand    : {min_capital:.2f} USDT")

if __name__ == "__main__":
    main()
