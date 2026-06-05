import pandas as pd
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb

# --- KONFIGURATION ---
DATA_FILE = 'BTCUSDT_5m_enriched_clean.csv'
BASE_MODEL_PATH = 'xgboost_polymarket.pkl'
BASE_SCALER_PATH = 'robust_scaler.pkl'
KMEANS_SCALER_PATH = 'kmeans_scaler.pkl'
KMEANS_MODEL_PATH = 'kmeans_model.pkl'

def calculate_all_features(df):
    """Berechnet alle Features für Clustering und Basis-Modell Inferenz."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # 1. Basis-Modell Features (11 Stück)
    # Mikrostruktur
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

    # On-Chain
    fee_col = 'total_fees_btc'
    tx_col = 'tx_count' if 'tx_count' in df.columns else 'transaction_count'
    block_col = 'avg_block_size'
    
    df['fee_momentum_ratio'] = df[fee_col].rolling(window=12).sum() / ((df[fee_col].rolling(window=48).sum() / 4) + 1e-8)
    df['tx_1h_sum'] = df[tx_col].rolling(window=12).sum()
    df['blocksize_1h_avg'] = df[block_col].rolling(window=12).mean()

    # 2. Clustering Features (Regime)
    high_low = high - low
    high_cp = np.abs(high - close.shift())
    low_cp = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    
    # Feature Listen
    base_features = [
        'dist_to_vwap', 'mfi_14', 'dist_to_bb_upper', 'dist_to_bb_lower', 
        'buying_pressure_ema_5', 'returns', 'volatility_20', 'volatility_60',
        'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg'
    ]
    
    regime_features = [
        'atr_14', 'volume', 'mfi_14', 'volatility_20', 'avg_block_size'
    ]
    
    df.dropna(subset=base_features + regime_features, inplace=True)
    return df, base_features, regime_features

def run_regime_analysis():
    print(f"📥 Lade Daten: {DATA_FILE}...")
    df_raw = pd.read_csv(DATA_FILE)
    df, base_features, regime_features = calculate_all_features(df_raw)
    
    # --- SCHRITT 1 & 2: K-Means Clustering ---
    print(f"🤖 Trainiere K-Means Clustering (n=4) auf {regime_features}...")
    regime_scaler = StandardScaler()
    X_regime = regime_scaler.fit_transform(df[regime_features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['regime_id'] = kmeans.fit_transform(X_regime).argmin(axis=1) # Note: labels_ is standard, fit_predict or fit then labels_
    df['regime_id'] = kmeans.labels_
    
    # Speichern der Modelle
    with open(KMEANS_SCALER_PATH, 'wb') as f:
        pickle.dump(regime_scaler, f)
    with open(KMEANS_MODEL_PATH, 'wb') as f:
        pickle.dump(kmeans, f)
    
    # --- SCHRITT 3: Basis-Modell Inferenz ---
    if not os.path.exists(BASE_MODEL_PATH) or not os.path.exists(BASE_SCALER_PATH):
        print("❌ Basis-Modell oder Scaler fehlt!")
        return
        
    print(f"🔮 Führe Inferenz mit Basis-Modell durch...")
    with open(BASE_MODEL_PATH, 'rb') as f:
        base_model = pickle.load(f)
    with open(BASE_SCALER_PATH, 'rb') as f:
        base_scaler = pickle.load(f)
        
    X_base = base_scaler.transform(df[base_features])
    df['base_prob'] = base_model.predict_proba(X_base)[:, 1]
    
    # Simuliere Trades
    # Trade: entry at current close, exit at next close (since we are on 5m intervals)
    # df['next_close'] = df['close'].shift(-1) # Need to shift based on original sequence
    # Since we dropped rows, we must ensure 'next_close' is correctly aligned.
    # Re-calculating correctly:
    df['is_win'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Filtere nur die Trades (> 0.58)
    trades_df = df[df['base_prob'] > 0.580].copy()
    
    # --- SCHRITT 4: Regime-Analyse ---
    print("\n" + "="*80)
    print("📊 MARKT-REGIME ANALYSE (UNSUPERVISED LEARNING)")
    print("="*80)
    
    stats = df.groupby('regime_id').agg({
        'atr_14': 'mean',
        'volume': 'mean',
        'volatility_20': 'mean'
    }).rename(columns={
        'atr_14': 'Avg ATR',
        'volume': 'Avg Volume',
        'volatility_20': 'Avg Vola'
    })
    
    # Win-Rate und Trade-Count pro Regime
    regime_trades = trades_df.groupby('regime_id').agg({
        'is_win': ['count', 'mean']
    })
    regime_trades.columns = ['Trades', 'Win-Rate']
    regime_trades['Win-Rate'] = regime_trades['Win-Rate'] * 100
    
    final_report = pd.concat([stats, regime_trades], axis=1).fillna(0)
    
    print(final_report.to_string(formatters={
        'Avg ATR': '{:,.2f}'.format,
        'Avg Volume': '{:,.2f}'.format,
        'Avg Vola': '{:.6f}'.format,
        'Trades': '{:.0f}'.format,
        'Win-Rate': '{:.2f}%'.format
    }))
    
    print("\n" + "="*80)
    print("Interpretation der Regimes (Vorschlag):")
    # Sort regimes by ATR to help identify "Panic" vs "Quiet"
    sorted_regimes = stats.sort_values('Avg ATR').index.tolist()
    print(f"Regime {sorted_regimes[0]}: Ruhiger Markt (Low ATR)")
    print(f"Regime {sorted_regimes[-1]}: Panik / Hohe Vola (Max ATR)")
    print("="*80)

if __name__ == "__main__":
    run_regime_analysis()
