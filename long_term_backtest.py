import pandas as pd
import numpy as np
import pickle
import os

# --- KONFIGURATION ---
INPUT_FILE = 'final_data_ready_for_backtest.csv'
MODEL_PATH = 'xgboost_polymarket.pkl'
SCALER_PATH = 'robust_scaler.pkl'
START_CAPITAL = 10000.0
THRESHOLDS = [57.0, 57.5, 58.0, 58.5, 59.0, 59.5, 60.0, 60.5]

def calculate_features(df):
    """Exakt dieselbe Feature-Logik wie in xgboost_polymarket_train.py."""
    df = df.copy()
    
    # Sicherstellen, dass Spaltennamen klein geschrieben sind
    df.columns = [c.lower() for c in df.columns]
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # 1. VWAP (Volume Weighted Average Price) - Intraday Style
    df['typical_price'] = (high + low + close) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['dist_to_vwap'] = (close - df['vwap']) / df['vwap']

    # 2. MFI (Money Flow Index) - 14 Perioden
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    pos_flow = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0), index=df.index)
    neg_flow = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0), index=df.index)
    pos_flow_sum = pos_flow.rolling(window=14).sum()
    neg_flow_sum = neg_flow.rolling(window=14).sum()
    money_ratio = pos_flow_sum / (neg_flow_sum + 1e-8)
    df['mfi_14'] = 100 - (100 / (1 + money_ratio))

    # 3. Bollinger Bands (Distanz) - 20 Perioden
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

    # 6. On-Chain Features (Mixed-Frequency Engineering)
    fees_1h_sum = df['total_fees_btc'].rolling(window=12, min_periods=1).sum()
    fees_4h_sum = df['total_fees_btc'].rolling(window=48, min_periods=1).sum()
    tx_1h_sum = df['transaction_count'].rolling(window=12, min_periods=1).sum()
    blocksize_1h_avg = df['avg_block_size'].rolling(window=12, min_periods=1).mean()

    df['fee_momentum_ratio'] = fees_1h_sum / ((fees_4h_sum / 4) + 1e-8)
    df['tx_1h_sum'] = tx_1h_sum
    df['blocksize_1h_avg'] = blocksize_1h_avg

    # Drop NaNs / Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

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
    df = calculate_features(df)
    
    features = [
        'dist_to_vwap', 'mfi_14', 'dist_to_bb_upper', 'dist_to_bb_lower', 
        'buying_pressure_ema_5', 'returns', 'volatility_20', 'volatility_60',
        'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg'
    ]
    
    print(f"🧠 Nutze {len(features)} Features für Inferenz...")

    X = scaler.transform(df[features])
    df['prob'] = model.predict_proba(X)[:, 1] * 100

    results = []

    print(f"\n🚀 Starte Polymarket-Backtest-Simulation...")
    print("-" * 85)
    print(f"{'Threshold':>10} | {'Trades':>8} | {'Win-Rate':>10} | {'End Capital':>15} | {'Total PnL %':>12}")
    print("-" * 85)

    for threshold in THRESHOLDS:
        capital = START_CAPITAL
        trades_count = 0
        wins = 0
        
        for i in range(len(df) - 1):
            prob = df['prob'].iloc[i]
            
            if prob >= threshold:
                trades_count += 1
                entry_price = df['close'].iloc[i]
                exit_price = df['close'].iloc[i+1]
                
                # Quarter-Kelly Risiko Management
                p = prob / 100.0
                q = 1.0 - p
                kelly_f = max(0, p - q) 
                risk_fraction = kelly_f * 0.25 
                
                # Polymarket Trade Resolution (Binäre Optionen Simulation)
                cost_per_share = p # Modell-Wahrscheinlichkeit als Kaufpreis
                bet_amount = capital * risk_fraction
                
                if exit_price > entry_price:
                    # WIN: Share ist $1.00 wert
                    trade_pnl = bet_amount * (1.0 - cost_per_share) / cost_per_share
                    wins += 1
                else:
                    # LOSS: Share ist $0.00 wert
                    trade_pnl = -bet_amount
                
                capital += trade_pnl

        win_rate = (wins / trades_count * 100) if trades_count > 0 else 0
        total_pnl_pct = (capital - START_CAPITAL) / START_CAPITAL * 100
        
        print(f"{threshold:>10.1f} | {trades_count:>8} | {win_rate:>9.2f}% | {capital:>12.2f} USDT | {total_pnl_pct:>11.2f}%")
        
        results.append({
            'Threshold': threshold,
            'Trades': trades_count,
            'WinRate': win_rate,
            'FinalCapital': capital,
            'PnL_Pct': total_pnl_pct
        })

    print("-" * 85)
    
    if results:
        best = max(results, key=lambda x: x['FinalCapital'])
        print(f"🏆 Bestes Ergebnis bei Threshold {best['Threshold']} mit {best['PnL_Pct']:.2f}% Profit!")

if __name__ == "__main__":
    run_backtest()
