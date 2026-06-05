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
BASE_THRESHOLD = 0.58
META_THRESHOLD_MIN = 0.60
KELLY_FRACTION = 0.25
MAX_PORTFOLIO_RISK = 0.15 # 15% Cap pro Trade

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

    # 3. ATR & Circuit Breaker Logik
    high_low = high - low
    high_cp = np.abs(high - close.shift())
    low_cp = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    df['atr_mean_24h'] = df['atr_14'].rolling(window=288).mean().bfill()

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
    
    df.dropna(subset=base_features + ['atr_14', 'atr_mean_24h'], inplace=True)
    
    return df, base_features, meta_features

def run_dynamic_sizing_backtest():
    if not os.path.exists(INPUT_FILE) or not os.path.exists(BASE_MODEL_PATH) or \
       not os.path.exists(META_MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("❌ Fehler: Modelle oder Daten fehlen.")
        return

    print(f"📥 Lade Daten und beide Modelle für dynamisches Sizing...")
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
    
    capital = START_CAPITAL
    blocked_by_cb = 0
    tier_results = {
        'Tier 1 (Normal)': {'trades': 0, 'wins': 0, 'pnl': 0},
        'Tier 2 (Aggressiv)': {'trades': 0, 'wins': 0, 'pnl': 0},
        'Tier 3 (Maximal)': {'trades': 0, 'wins': 0, 'pnl': 0}
    }
    
    print(f"🚀 Starte Triple-Layer Backtest (Base: {BASE_THRESHOLD}, Meta > {META_THRESHOLD_MIN}, CB: 200% ATR)...")
    
    for i in range(len(df) - 1):
        row = df.iloc[i]
        base_prob = row['base_prob']
        
        if base_prob > BASE_THRESHOLD:
            # --- STUFE 1: Circuit Breaker (Kill-Switch) ---
            if row['atr_14'] > (row['atr_mean_24h'] * 2.00):
                blocked_by_cb += 1
                continue

            # --- STUFE 2: Meta-Modell Check ---
            X_meta = row[meta_features].values.reshape(1, -1)
            meta_prob = meta_model.predict_proba(X_meta)[0, 1]
            
            if meta_prob > META_THRESHOLD_MIN:
                # --- STUFE 3: Dynamisches Sizing ---
                multiplier = 1.0
                tier_name = 'Tier 1 (Normal)'
                
                if 0.60 < meta_prob <= 0.65:
                    multiplier = 1.0
                    tier_name = 'Tier 1 (Normal)'
                elif 0.65 < meta_prob <= 0.70:
                    multiplier = 2.0
                    tier_name = 'Tier 2 (Aggressiv)'
                elif meta_prob > 0.70:
                    multiplier = 3.0
                    tier_name = 'Tier 3 (Maximal)'
                
                # Einsatzberechnung (Quarter-Kelly * Multiplier)
                p = base_prob
                q = 1.0 - p
                kelly_f = max(0, p - q)
                risk_fraction = kelly_f * KELLY_FRACTION * multiplier
                
                # Safety Cap: Max 15%
                risk_fraction = min(risk_fraction, MAX_PORTFOLIO_RISK)
                
                bet_amount = capital * risk_fraction
                cost_per_share = p
                
                # Trade Ergebnis
                entry_price = row['close']
                exit_price = df['close'].iloc[i+1]
                is_win = exit_price > entry_price
                
                if is_win:
                    trade_pnl = bet_amount * (1.0 - cost_per_share) / (cost_per_share + 1e-8)
                    tier_results[tier_name]['wins'] += 1
                else:
                    trade_pnl = -bet_amount
                
                capital += trade_pnl
                tier_results[tier_name]['trades'] += 1
                tier_results[tier_name]['pnl'] += trade_pnl

    # Auswertung
    total_trades = sum(t['trades'] for t in tier_results.values())
    total_wins = sum(t['wins'] for t in tier_results.values())
    total_pnl_pct = (capital - START_CAPITAL) / START_CAPITAL * 100
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    print("\n" + "="*65)
    print("📊 ERGEBNIS: DYNAMIC POSITION SIZING BACKTEST")
    print("="*65)
    print(f"Startkapital:    {START_CAPITAL:,.2f} USDT")
    print(f"Endkapital:      {capital:,.2f} USDT")
    print(f"Total PnL:       {total_pnl_pct:+.2f}%")
    print(f"Gesamt Win-Rate: {win_rate:.2f}% ({total_wins}/{total_trades} Trades)")
    print("-" * 65)
    print(f"{'Tier':<20} | {'Trades':>8} | {'Win-Rate':>10} | {'Net PnL':>12}")
    print("-" * 65)
    
    for name, data in tier_results.items():
        tr_wr = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
        print(f"{name:<20} | {data['trades']:>8} | {tr_wr:>9.2f}% | {data['pnl']:>11.2f} USDT")
    
    print("="*65)

if __name__ == "__main__":
    run_dynamic_sizing_backtest()
