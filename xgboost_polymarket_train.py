import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import RobustScaler
import pickle
import json
import os
import time

print("✅ Starting Polymarket XGBoost Pipeline (Phase 1)...")

# ==============================================================================
# 1. Daten Laden (Enriched 5m Data)
# ==============================================================================
CSV_PATH = 'BTCUSDT_5m_enriched.csv'
if not os.path.exists(CSV_PATH):
    print(f"❌ Fehler: {CSV_PATH} nicht gefunden. Führe erst merge_onchain_data.py aus.")
    exit(1)

print(f"📥 Lade angereicherte Daten von {CSV_PATH}...")
start_time = time.time()
df = pd.read_csv(CSV_PATH, parse_dates=['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)
df.columns = [c.lower() for c in df.columns]

print(f"✅ Geladen: {len(df)} Zeilen in {time.time() - start_time:.2f} Sekunden.")

# ==============================================================================
# 2. Feature Engineering (Fokus: Micro-structure)
# ==============================================================================
print("🔄 Berechne Micro-structure Features...")
start_time = time.time()

close = df['close']
high = df['high']
low = df['low']
volume = df['volume']

# VWAP (Volume Weighted Average Price) - Intraday
df['typical_price'] = (high + low + close) / 3
# Berechne kumulatives Volumen und kumulativen typischen Preis * Volumen pro Tag
df['date_only'] = df.index.date
daily_groups = df.groupby('date_only')
df['cum_vol'] = daily_groups['volume'].cumsum()
df['cum_vol_price'] = daily_groups.apply(lambda x: (x['typical_price'] * x['volume']).cumsum()).reset_index(level=0, drop=True)
df['vwap'] = df['cum_vol_price'] / df['cum_vol']
df['dist_to_vwap'] = (close - df['vwap']) / df['vwap']
df.drop(['typical_price', 'date_only', 'cum_vol', 'cum_vol_price'], axis=1, inplace=True)

# MFI (Money Flow Index) - 14 Perioden
typical_price = (high + low + close) / 3
raw_money_flow = typical_price * volume
pos_flow = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0), index=df.index)
neg_flow = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0), index=df.index)
pos_flow_sum = pos_flow.rolling(window=14).sum()
neg_flow_sum = neg_flow.rolling(window=14).sum()
money_ratio = pos_flow_sum / neg_flow_sum
df['mfi_14'] = 100 - (100 / (1 + money_ratio))

# Bollinger Bands (Distanz) - 20 Perioden
sma_20 = close.rolling(window=20).mean()
std_20 = close.rolling(window=20).std()
df['bb_upper'] = sma_20 + (2 * std_20)
df['bb_lower'] = sma_20 - (2 * std_20)
df['dist_to_bb_upper'] = (close - df['bb_upper']) / df['bb_upper']
df['dist_to_bb_lower'] = (close - df['bb_lower']) / df['bb_lower']

# Order Book Imbalance Proxy (via Volumen & Preis-Action)
# Da wir kein echtes L2 Orderbuch in dieser CSV haben, schätzen wir den Kauf/Verkauf-Druck
# basierend darauf, ob der Close näher am High (Käufer dominieren) oder Low (Verkäufer dominieren) ist.
df['close_location_value'] = ((close - low) - (high - close)) / (high - low + 1e-8)
df['buying_pressure'] = df['close_location_value'] * volume
df['buying_pressure_ema_5'] = df['buying_pressure'].ewm(span=5, adjust=False).mean()

# Basis-Features (Returns, Volatilität)
df['returns'] = close.pct_change()
df['volatility_20'] = df['returns'].rolling(window=20).std()
df['volatility_60'] = df['returns'].rolling(window=60).std()

# On-Chain Features: Mixed-Frequency Engineering (liesen.txt)
# Ziel: Diskrete Blöcke (viele Nullen) in kontinuierlichen "Netzwerk-Druck" verwandeln.

# 1h Fenster (12 * 5 Min) und 4h Fenster (48 * 5 Min)
fees_1h_sum = df['total_fees_btc'].rolling(window=12, min_periods=1).sum()
fees_4h_sum = df['total_fees_btc'].rolling(window=48, min_periods=1).sum()
tx_1h_sum = df['tx_count'].rolling(window=12, min_periods=1).sum()
blocksize_1h_avg = df['avg_block_size'].rolling(window=12, min_periods=1).mean()

# "Network Heat" Signal: Aktuelle Stunde im Vergleich zum 4h-Durchschnitt
# Verhindert Division durch Null
df['fee_momentum_ratio'] = fees_1h_sum / ((fees_4h_sum / 4) + 1e-8)
df['tx_1h_sum'] = tx_1h_sum
df['blocksize_1h_avg'] = blocksize_1h_avg

# WICHTIG: Rohe Spalten droppen (Sparsity entfernen), um Modell-Rauschen zu verhindern
df.drop(['total_fees_btc', 'tx_count', 'avg_block_size'], axis=1, inplace=True)

print(f"✅ Features berechnet in {time.time() - start_time:.2f} Sekunden.")

# ==============================================================================
# 3. Polymarket Target Design
# ==============================================================================
print("🎯 Generiere Polymarket-Bedingung (Target)...")
# Polymarket Target: Wird der Preis in der *nächsten* 5-Minuten-Kerze steigen?
# Dies entspricht: Close[t+1] > Close[t]
target_horizon = 1
df['target'] = (df['close'].shift(-target_horizon) > df['close']).astype(int)

# Bereinigung (Warmup-Phase entfernen & letzte unbekannte Target-Zeile)
df = df.iloc[60:-target_horizon].copy() # 60 wegen volatility_60 und Rolling Windows

features = ['dist_to_vwap', 'mfi_14', 'dist_to_bb_upper', 'dist_to_bb_lower', 
            'buying_pressure_ema_5', 'returns', 'volatility_20', 'volatility_60',
            'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg']

df[features] = df[features].ffill().fillna(0)
df[features] = df[features].replace([np.inf, -np.inf], 0)

print(f"📊 Finales Dataset: {df.shape}")
print(f"   Zielverteilung: {df['target'].value_counts().to_dict()}")

# ==============================================================================
# 4. Walk-Forward Split (wie im alten Skript)
# ==============================================================================
def create_walk_forward_splits(df_idx, train_months=24, test_months=6, purge_days=1, embargo_days=1):
    splits = []
    dates = df_idx
    start_date = dates[0]
    end_date = dates[-1]
    current_train_start = start_date
    
    while True:
        raw_train_end = current_train_start + pd.DateOffset(months=train_months)
        test_start = raw_train_end + pd.Timedelta(days=1)
        test_end = raw_train_end + pd.DateOffset(months=test_months)
        
        if test_end > end_date:
            break
            
        purged_train_end = raw_train_end - pd.Timedelta(days=purge_days)
        train_mask = (dates >= current_train_start) & (dates <= purged_train_end)
        test_mask = (dates >= test_start) & (dates < test_end)
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        if len(train_idx) > 50 and len(test_idx) > 5:
            splits.append({
                'train_idx': train_idx,
                'test_idx': test_idx
            })
        current_train_start = current_train_start + pd.DateOffset(months=test_months) + pd.Timedelta(days=embargo_days)
    return splits

splits = create_walk_forward_splits(df.index)
print(f"✅ {len(splits)} Walk-Forward Splits erstellt.")

# ==============================================================================
# 5. XGBoost GPU Training
# ==============================================================================
print("\n🚀 Starte GPU-beschleunigtes XGBoost Training...")

# XGBoost Parameter für GPU (Bestätigt durch statistisch signifikantes Optuna Tuning)
xgb_params = {
    'n_estimators': 204,
    'max_depth': 5,
    'learning_rate': 0.01996536469782752,
    'subsample': 0.8367131097841103,
    'colsample_bytree': 0.6809298120078194,
    'gamma': 2.297145942276206,
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42,
}

all_preds = []
all_actuals = []
all_probas = []
all_dates = []
last_model = None

start_train_time = time.time()

for i, split in enumerate(splits):
    scaler = RobustScaler()
    X_train_raw = df.iloc[split['train_idx']][features].values
    X_test_raw = df.iloc[split['test_idx']][features].values
    
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    y_train = df.iloc[split['train_idx']]['target'].values
    y_test = df.iloc[split['test_idx']]['target'].values
    test_dates = df.index[split['test_idx']]
    
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    
    all_preds.extend(preds)
    all_actuals.extend(y_test)
    all_probas.extend(probas)
    all_dates.extend(test_dates)
    last_model = model
    
    print(f"   Split {i+1}/{len(splits)} beendet.")

print(f"\n⏱️ Training abgeschlossen in {time.time() - start_train_time:.2f} Sekunden (GPU Accelerated!).")

# Evaluierung
accuracy = accuracy_score(all_actuals, all_preds)
precision = precision_score(all_actuals, all_preds, zero_division=0)
print(f"✅ Baseline Accuracy: {accuracy:.4f} | Precision: {precision:.4f}")

# Polymarket Confidence Filter (Statistisch signifikantes Optuna Resultat)
threshold = 0.601  # Optuna Validierter Threshold (N >= 200)
probs = np.array(all_probas)
acts = np.array(all_actuals)
preds = np.array(all_preds)

confidence = np.maximum(probs, 1 - probs)
high_conf_mask = confidence >= threshold
n_executed = high_conf_mask.sum()

if n_executed > 0:
    filtered_preds = preds[high_conf_mask]
    filtered_actuals = acts[high_conf_mask]
    conf_acc = accuracy_score(filtered_actuals, filtered_preds)
    print(f"\n🎯 Polymarket Execution Simulation (Confidence >= {threshold*100}%):")
    print(f"   Ausgeführte Trades: {n_executed}/{len(preds)} ({(n_executed/len(preds))*100:.2f}%)")
    print(f"   Win Rate (Accuracy): {conf_acc:.4f}")
else:
    print(f"\n🎯 Keine Trades mit Confidence >= {threshold*100}% gefunden.")

# ==============================================================================
# 6. Speichern der Modelle & Metadaten
# ==============================================================================
print("\n💾 Speichere Artefakte in @xgboost/ ...")

model_path = 'xgboost_polymarket.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(last_model, f)
    
scaler_path = 'robust_scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

# Speichere Vorhersagen für den Backtester
preds_df = pd.DataFrame({
    'date': all_dates,
    'actual': all_actuals,
    'prediction': all_preds,
    'probability': all_probas
})
preds_df.to_csv('predictions.csv', index=False)

meta = {
    'features': features,
    'target': 'target',
    'threshold_tested': threshold,
    'accuracy': float(accuracy),
    'n_samples': len(df)
}
with open('metadata.json', 'w') as f:
    json.dump(meta, f)

print("✅ Alle Dateien erfolgreich gespeichert!")
