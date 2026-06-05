import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

# --- KONFIGURATION ---
INPUT_FILE = 'final_data_ready_for_backtest.csv'
BASE_MODEL_PATH = 'xgboost_polymarket.pkl'
META_MODEL_PATH = 'xgboost_risk_manager.pkl'
SCALER_PATH = 'robust_scaler.pkl'
START_CAPITAL = 10000.0
META_THRESHOLD = 0.60  # Festgehaltener Meta-Threshold aus Day 2
BASE_THRESHOLDS = [round(x * 0.01, 2) for x in range(55, 66)]

def calculate_features(df):
    """Exakte Feature-Logik für Basis- und Meta-Modell."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # 1. Mikrostruktur Features (Basis)
    df['typical_price'] = (high + low + close) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['dist_to_vwap'] = (close - df['vwap']) / df['vwap']

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
    df['dist_to_bb_upper'] = (close - df['bb_upper']) / df['bb_upper']
    df['dist_to_bb_lower'] = (close - df['bb_lower']) / df['bb_lower']

    # Buying Pressure
    df['clv'] = ((close - low) - (high - close)) / (high - low + 1e-8)
    df['buying_pressure_ema_5'] = (df['clv'] * volume).ewm(span=5, adjust=False).mean()

    # Returns & Vola
    df['returns'] = close.pct_change()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    df['volatility_60'] = df['returns'].rolling(window=60).std()

    # 2. On-Chain Features (Basis)
    fee_col = 'total_fees_btc'
    tx_col = 'transaction_count' if 'transaction_count' in df.columns else 'tx_count'
    block_col = 'avg_block_size'
    
    df['fee_momentum_ratio'] = df[fee_col].rolling(window=12).sum() / ((df[fee_col].rolling(window=48).sum() / 4) + 1e-8)
    df['tx_1h_sum'] = df[tx_col].rolling(window=12).sum()
    df['blocksize_1h_avg'] = df[block_col].rolling(window=12).mean()

    # 3. ATR (Meta)
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
    
    df.dropna(subset=base_features + ['atr_14'], inplace=True)
    
    return df, base_features, meta_features

def run_meta_sensitivity():
    if not os.path.exists(INPUT_FILE) or not os.path.exists(BASE_MODEL_PATH) or \
       not os.path.exists(META_MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("❌ Fehler: Modelle oder Daten fehlen.")
        return

    print(f"📥 Lade Daten und beide Modelle (Dual-Architecture)...")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    with open(BASE_MODEL_PATH, 'rb') as f:
        base_model = pickle.load(f)
    with open(META_MODEL_PATH, 'rb') as f:
        meta_model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    df, base_features, meta_features = calculate_features(df)
    
    # Basis-Inferenz (Batch)
    X_base = scaler.transform(df[base_features])
    df['base_prob'] = base_model.predict_proba(X_base)[:, 1]
    
    results = []

    print(f"\n🚀 Starte Meta-Sensitivitätsanalyse (Risk Manager @ {META_THRESHOLD})...")
    print("-" * 75)
    print(f"{'Base Thr':>10} | {'PnL %':>12} | {'Win-Rate':>10} | {'Trades':>8} | {'Blocked':>8}")
    print("-" * 75)

    for base_threshold in BASE_THRESHOLDS:
        capital = START_CAPITAL
        trades_count = 0
        wins = 0
        blocked_count = 0
        
        for i in range(len(df) - 1):
            row = df.iloc[i]
            base_prob = row['base_prob']
            
            if base_prob >= base_threshold:
                # Meta-Modell Check
                X_meta = row[meta_features].values.reshape(1, -1)
                meta_prob = meta_model.predict_proba(X_meta)[0, 1]
                
                if meta_prob > META_THRESHOLD:
                    trades_count += 1
                    entry_price = row['close']
                    exit_price = df['close'].iloc[i+1]
                    
                    # Quarter-Kelly
                    p = base_prob
                    q = 1.0 - p
                    kelly_f = max(0, p - q)
                    risk_fraction = kelly_f * 0.25
                    
                    cost_per_share = p
                    bet_amount = min(capital * risk_fraction, capital * 0.10) # Max 10% pro Trade
                    
                    if exit_price > entry_price:
                        trade_pnl = bet_amount * (1.0 - cost_per_share) / (cost_per_share + 1e-8)
                        wins += 1
                    else:
                        trade_pnl = -bet_amount
                    
                    capital += trade_pnl
                else:
                    blocked_count += 1

        win_rate = (wins / trades_count * 100) if trades_count > 0 else 0
        total_pnl_pct = (capital - START_CAPITAL) / START_CAPITAL * 100
        
        print(f"{base_threshold:>10.2f} | {total_pnl_pct:>11.2f}% | {win_rate:>9.2f}% | {trades_count:>8} | {blocked_count:>8}")
        
        results.append({
            'Threshold': base_threshold,
            'PnL_Pct': total_pnl_pct,
            'WinRate': win_rate,
            'Trades': trades_count,
            'Blocked': blocked_count
        })

    print("-" * 75)
    if results:
        best = max(results, key=lambda x: x['PnL_Pct'])
        print(f"🏆 Bestes Ensemble bei Base-Threshold {best['Threshold']:.2f} mit {best['PnL_Pct']:.2f}% Profit!")

if __name__ == "__main__":
    run_meta_sensitivity()
