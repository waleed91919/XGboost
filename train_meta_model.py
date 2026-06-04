import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import time

# --- KONFIGURATION ---
INPUT_FILE = 'BTCUSDT_5m_enriched_clean.csv'
BASE_MODEL_PATH = 'xgboost_polymarket.pkl'
SCALER_PATH = 'robust_scaler.pkl'
META_MODEL_OUTPUT = 'xgboost_risk_manager.pkl'
THRESHOLD = 0.580

print("✅ Starting Meta-Labeling Training Pipeline (Phase 2)...")

# ==============================================================================
# 1. Daten Laden & Feature Engineering
# ==============================================================================
if not os.path.exists(INPUT_FILE):
    print(f"❌ Fehler: {INPUT_FILE} nicht gefunden.")
    exit(1)

print(f"📥 Lade Daten von {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
df.columns = [c.lower() for c in df.columns]

def calculate_all_features(df):
    """Berechnet alle Features für das Basis-Modell und zusätzliche Meta-Features."""
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # --- Mikrostruktur (für Basis-Modell) ---
    df['typical_price'] = (high + low + close) / 3
    # VWAP (Intraday style approximation)
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

    # --- On-Chain (für Basis-Modell) ---
    # Beachte: Spaltennamen in final_data_ready_for_backtest.csv könnten abweichen
    # Wir prüfen 'total_fees_btc', 'transaction_count' (oder 'tx_count'), 'avg_block_size'
    fee_col = 'total_fees_btc'
    tx_col = 'transaction_count' if 'transaction_count' in df.columns else 'tx_count'
    block_col = 'avg_block_size'

    df['fee_momentum_ratio'] = df[fee_col].rolling(window=12).sum() / ((df[fee_col].rolling(window=48).sum() / 4) + 1e-8)
    df['tx_1h_sum'] = df[tx_col].rolling(window=12).sum()
    df['blocksize_1h_avg'] = df[block_col].rolling(window=12).mean()

    # --- ATR (für Meta-Modell) ---
    high_low = high - low
    high_cp = np.abs(high - close.shift())
    low_cp = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()

    return df

df = calculate_all_features(df)

# Basis-Features Liste (muss 1:1 mit dem Training übereinstimmen)
base_features = [
    'dist_to_vwap', 'mfi_14', 'dist_to_bb_upper', 'dist_to_bb_lower', 
    'buying_pressure_ema_5', 'returns', 'volatility_20', 'volatility_60',
    'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg'
]

# Bereinigung NaNs
df.dropna(subset=base_features + ['atr_14'], inplace=True)

# ==============================================================================
# 2. Basis-Modell Vorhersagen (base_prob)
# ==============================================================================
print(f"🧠 Lade Basis-Modell: {BASE_MODEL_PATH}...")
with open(BASE_MODEL_PATH, 'rb') as f:
    base_model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Skalierung und Prediction
X_base = scaler.transform(df[base_features])
df['base_prob'] = base_model.predict_proba(X_base)[:, 1]

# ==============================================================================
# 3. Meta-Target Generierung
# ==============================================================================
print(f"🎯 Generiere Meta-Target (Threshold: {THRESHOLD})...")

# Das Target ist: Hat die NÄCHSTE Kerze im Plus geschlossen?
# 1 = Win (UP), 0 = Loss (DOWN)
df['next_is_up'] = (df['close'].shift(-1) > df['close']).astype(int)

# Filtere auf Situationen, in denen das Basis-Modell einen Trade empfohlen hätte
meta_df = df[df['base_prob'] > THRESHOLD].copy()
meta_df.dropna(subset=['next_is_up'], inplace=True) # Letzte Zeile entfernen

# Das Ziel für den Risk Manager ist es vorherzusagen, ob der Trade ein WIN (1) oder LOSS (0) wird.
meta_df['meta_target'] = meta_df['next_is_up']

print(f"📊 Meta-Dataset Größe: {len(meta_df)} Zeilen")
print(f"   Win-Rate in diesem Sample: {meta_df['meta_target'].mean():.2%}")

# ==============================================================================
# 4. Meta-Training
# ==============================================================================
print("\n🚀 Trainiere Meta-Modell (Risk Manager) auf GPU...")

# Meta-Features definieren (Erweitert um On-Chain Features)
meta_features = [
    'base_prob', 'atr_14', 'volume', 'volatility_20', 'volatility_60', 
    'dist_to_bb_upper', 'dist_to_bb_lower',
    'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg'
]

# Zeitbasierter Split (80% Train, 20% Val)
split_idx = int(len(meta_df) * 0.8)
train_df = meta_df.iloc[:split_idx]
val_df = meta_df.iloc[split_idx:]

X_train = train_df[meta_features]
y_train = train_df['meta_target']
X_val = val_df[meta_features]
y_val = val_df['meta_target']

# XGBoost Parameter
meta_params = {
    'n_estimators': 2000,
    'max_depth': 6, 
    'learning_rate': 0.005, # Noch langsamer lernen
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42,
    'eval_metric': 'logloss',
    'early_stopping_rounds': 100
}

meta_model = xgb.XGBClassifier(**meta_params)
meta_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

# Feature Importance
importances = pd.Series(meta_model.feature_importances_, index=meta_features).sort_values(ascending=False)
print("\n📊 Meta-Feature Importance:")
print(importances)

# ==============================================================================
# 5. Export
# ==============================================================================
print(f"💾 Speichere Meta-Modell als {META_MODEL_OUTPUT}...")
with open(META_MODEL_OUTPUT, 'wb') as f:
    pickle.dump(meta_model, f)

print("\n✅ Meta-Labeling Prozess abgeschlossen!")
print(f"Modell '{META_MODEL_OUTPUT}' ist bereit für den Einsatz im Risk Management.")
