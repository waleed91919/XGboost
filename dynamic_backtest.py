import pandas as pd
import numpy as np
import pickle
import os

# --- KONFIGURATION ---
INPUT_FILE = 'final_data_ready_for_backtest.csv'
MODEL_PATH = 'xgboost_polymarket.pkl'
SCALER_PATH = 'robust_scaler.pkl'
START_CAPITAL = 10000.0
KELLY_FRACTION = 0.25 # Quarter-Kelly

def calculate_features(df):
    """Exakt dieselbe Feature-Logik wie in long_term_backtest.py PLUS ATR."""
    df = df.copy()
    
    # Sicherstellen, dass Spaltennamen klein geschrieben sind
    df.columns = [c.lower() for c in df.columns]
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # 1. VWAP
    df['typical_price'] = (high + low + close) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['dist_to_vwap'] = (close - df['vwap']) / df['vwap']

    # 2. MFI
    raw_money_flow = df['typical_price'] * volume
    pos_flow = pd.Series(np.where(df['typical_price'] > df['typical_price'].shift(1), raw_money_flow, 0), index=df.index)
    neg_flow = pd.Series(np.where(df['typical_price'] < df['typical_price'].shift(1), raw_money_flow, 0), index=df.index)
    pos_flow_sum = pos_flow.rolling(window=14).sum()
    neg_flow_sum = neg_flow.rolling(window=14).sum()
    money_ratio = pos_flow_sum / (neg_flow_sum + 1e-8)
    df['mfi_14'] = 100 - (100 / (1 + money_ratio))

    # 3. Bollinger Bands
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    df['dist_to_bb_upper'] = (close - bb_upper) / bb_upper
    df['dist_to_bb_lower'] = (close - bb_lower) / bb_lower

    # 4. Buying Pressure EMA
    df['close_location_value'] = ((close - low) - (high - close)) / (high - low + 1e-8)
    df['buying_pressure'] = df['close_location_value'] * volume
    df['buying_pressure_ema_5'] = df['buying_pressure'].ewm(span=5, adjust=False).mean()

    # 5. Returns & Volatilität
    df['returns'] = close.pct_change()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    df['volatility_60'] = df['returns'].rolling(window=60).std()

    # 6. On-Chain Features
    fees_1h_sum = df['total_fees_btc'].rolling(window=12, min_periods=1).sum()
    fees_4h_sum = df['total_fees_btc'].rolling(window=48, min_periods=1).sum()
    tx_1h_sum = df['transaction_count'].rolling(window=12, min_periods=1).sum()
    blocksize_1h_avg = df['avg_block_size'].rolling(window=12, min_periods=1).mean()

    df['fee_momentum_ratio'] = fees_1h_sum / ((fees_4h_sum / 4) + 1e-8)
    df['tx_1h_sum'] = tx_1h_sum
    df['blocksize_1h_avg'] = blocksize_1h_avg

    # 7. ATR (Average True Range) für Dynamischen Threshold
    high_low = high - low
    high_cp = np.abs(high - close.shift())
    low_cp = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    
    # 24h ATR Durchschnitt (288 Kerzen @ 5m)
    df['atr_mean_24h'] = df['atr_14'].rolling(window=288).mean().bfill()

    # Drop NaNs / Inf für Modell-Features (ATR brauchen wir behalten)
    feature_cols = [
        'dist_to_vwap', 'mfi_14', 'dist_to_bb_upper', 'dist_to_bb_lower', 
        'buying_pressure_ema_5', 'returns', 'volatility_20', 'volatility_60',
        'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg'
    ]
    
    # Wir erstellen eine Maske für Zeilen mit gültigen Features
    mask = df[feature_cols].notna().all(axis=1) & (~np.isinf(df[feature_cols])).all(axis=1)
    df = df[mask].copy()
    
    return df, feature_cols

def run_backtest():
    if not os.path.exists(INPUT_FILE) or not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("❌ Fehler: Modell, Scaler oder Datensatz fehlt.")
        return

    print(f"📥 Lade Daten und Modell...")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # Features berechnen
    df, features = calculate_features(df)
    
    print(f"🧠 Nutze {len(features)} Features für Inferenz...")
    X = scaler.transform(df[features])
    df['prob'] = model.predict_proba(X)[:, 1]
    
    print(f"DEBUG: Prob Range: {df['prob'].min():.4f} - {df['prob'].max():.4f}")
    print(f"DEBUG: Mean Prob: {df['prob'].mean():.4f}")

    # Target: BTC Preis steigt in der nächsten 5m Kerze
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Simulation
    capital = START_CAPITAL
    trades = []
    
    counts = {"aggressive": 0, "normal": 0, "conservative": 0}
    
    print(f"🚀 Starte Dynamische Simulation...")
    
    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i+1]
        
        # Dynamischer Threshold (Striktere Multiplikatoren)
        current_atr = row['atr_14']
        avg_atr = row['atr_mean_24h']
        
        if current_atr < (avg_atr * 0.75):
            threshold = 0.575
            counts["aggressive"] += 1
        elif current_atr > (avg_atr * 1.20):
            threshold = 0.605
            counts["conservative"] += 1
        else:
            threshold = 0.580
            counts["normal"] += 1
            
        prob = row['prob']
        side = None
        confidence = 0
        
        # NUR YES (UP) TRADES
        if prob >= threshold:
            side = "YES"
            confidence = prob
            
        if side:
            # Polymarket Logic: Price = Probability
            p = confidence 
            q = 1.0 - p
            
            # Simplified Kelly from long_term_backtest.py: kelly_f = p - q
            kelly_f = max(0, p - q)
            risk_fraction = kelly_f * KELLY_FRACTION
            
            bet_amount = capital * risk_fraction
            
            # Max 10% pro Trade zur Sicherheit
            bet_amount = min(bet_amount, capital * 0.10)
            
            if bet_amount > 0:
                is_win = False
                if side == "YES" and next_row['close'] > row['close']:
                    is_win = True
                
                if is_win:
                    # Gewinn: ROI = (1.0 - p) / p
                    trade_pnl = bet_amount * (1.0 - p) / (p + 1e-8)
                else:
                    # Verlust: Einsatz weg
                    trade_pnl = -bet_amount
                
                capital += trade_pnl
                trades.append({
                    'time': df.index[i],
                    'threshold': threshold,
                    'side': side,
                    'confidence': confidence,
                    'bet': bet_amount,
                    'win': is_win,
                    'pnl': trade_pnl,
                    'capital': capital
                })

    # Auswertung
    df_trades = pd.DataFrame(trades)
    
    print("\n" + "="*40)
    print("📊 BACKTEST ZUSAMMENFASSUNG (DYNAMISCH)")
    print("="*40)
    print(f"Startkapital:    {START_CAPITAL:,.2f} USDT")
    print(f"Endkapital:      {capital:,.2f} USDT")
    print(f"Total PnL:       {((capital/START_CAPITAL)-1)*100:+.2f}%")
    print(f"Anzahl Trades:   {len(df_trades)}")
    
    if len(df_trades) > 0:
        win_rate = (df_trades['win'].sum() / len(df_trades)) * 100
        print(f"Win-Rate:        {win_rate:.2f}%")
        
    print("-" * 40)
    print("Threshold Nutzung:")
    print(f"  Aggressiv (0.575):   {counts['aggressive']} Kerzen")
    print(f"  Normal (0.580):      {counts['normal']} Kerzen")
    print(f"  Konservativ (0.605): {counts['conservative']} Kerzen")
    
    if len(df_trades) > 0:
        print("\nTrade Verteilung:")
        print(df_trades.groupby('threshold').size())
    print("="*40)

if __name__ == "__main__":
    run_backtest()
