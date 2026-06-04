import pandas as pd
import numpy as np
import pickle
import os

# --- KONFIGURATION ---
INPUT_FILE = 'final_data_ready_for_backtest.csv'
MODEL_PATH = 'xgboost_polymarket.pkl'
SCALER_PATH = 'robust_scaler.pkl'
START_CAPITAL = 10000.0
CONFIDENCE_THRESHOLD = 0.58
KELLY_FRACTION = 0.25 # Quarter-Kelly

def calculate_features(df):
    """Feature-Logik inkl. ATR und 24h ATR-Durchschnitt."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # 1. Mikrostruktur Features
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
    df['dist_to_bb_upper'] = (close - (sma_20 + 2*std_20)) / (sma_20 + 2*std_20)
    df['dist_to_bb_lower'] = (close - (sma_20 - 2*std_20)) / (sma_20 - 2*std_20)

    # Buying Pressure
    df['clv'] = ((close - low) - (high - close)) / (high - low + 1e-8)
    df['buying_pressure_ema_5'] = (df['clv'] * volume).ewm(span=5, adjust=False).mean()

    # Returns & Vola
    df['returns'] = close.pct_change()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    df['volatility_60'] = df['returns'].rolling(window=60).std()

    # 2. On-Chain Features
    df['fee_momentum_ratio'] = df['total_fees_btc'].rolling(window=12).sum() / ((df['total_fees_btc'].rolling(window=48).sum() / 4) + 1e-8)
    df['tx_1h_sum'] = df['transaction_count'].rolling(window=12).sum()
    df['blocksize_1h_avg'] = df['avg_block_size'].rolling(window=12).mean()

    # 3. ATR & Circuit Breaker Logik
    high_low = high - low
    high_cp = np.abs(high - close.shift())
    low_cp = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    df['atr_mean_24h'] = df['atr_14'].rolling(window=288).mean().bfill()

    feature_cols = [
        'dist_to_vwap', 'mfi_14', 'dist_to_bb_upper', 'dist_to_bb_lower', 
        'buying_pressure_ema_5', 'returns', 'volatility_20', 'volatility_60',
        'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg'
    ]
    
    mask = df[feature_cols].notna().all(axis=1)
    df = df[mask].copy()
    
    return df, feature_cols

def run_backtest():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Datei {INPUT_FILE} nicht gefunden.")
        return

    print(f"📥 Lade Daten und Modell...")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    df, features = calculate_features(df)
    
    X = scaler.transform(df[features])
    df['prob'] = model.predict_proba(X)[:, 1]
    
    capital = START_CAPITAL
    trades = []
    blocked_count = 0
    
    print(f"🚀 Starte Hybrid-Backtest (Threshold: {CONFIDENCE_THRESHOLD}, Kill-Switch: 200% ATR)...")
    
    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i+1]
        
        prob = row['prob']
        
        # Check: Signal vorhanden? (NUR YES)
        if prob >= CONFIDENCE_THRESHOLD:
            # Circuit Breaker Check
            if row['atr_14'] > (row['atr_mean_24h'] * 2.00):
                blocked_count += 1
                continue
                
            # Trade Ausführung
            p = prob
            q = 1.0 - p
            kelly_f = max(0, p - q)
            risk_fraction = kelly_f * KELLY_FRACTION
            bet_amount = min(capital * risk_fraction, capital * 0.10)
            
            if bet_amount > 0:
                is_win = next_row['close'] > row['close']
                if is_win:
                    pnl = bet_amount * (1.0 - p) / (p + 1e-8)
                else:
                    pnl = -bet_amount
                
                capital += pnl
                trades.append({'win': is_win, 'pnl': pnl})

    # Auswertung
    df_trades = pd.DataFrame(trades)
    print("\n" + "="*45)
    print("📊 ERGEBNIS: HYBRID CIRCUIT BREAKER BACKTEST")
    print("="*45)
    print(f"Startkapital:    {START_CAPITAL:,.2f} USDT")
    print(f"Endkapital:      {capital:,.2f} USDT")
    print(f"Total PnL:       {((capital/START_CAPITAL)-1)*100:+.2f}%")
    print("-" * 45)
    print(f"Ausgeführte Trades: {len(df_trades)}")
    if len(df_trades) > 0:
        win_rate = (df_trades['win'].sum() / len(df_trades)) * 100
        print(f"Win-Rate:           {win_rate:.2f}%")
    print(f"Blockierte Trades:  {blocked_count} (Circuit Breaker)")
    print("="*45)

if __name__ == "__main__":
    run_backtest()
