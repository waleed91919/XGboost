import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import json
from sklearn.preprocessing import RobustScaler
import pickle
import os
import time

print("🔍 Starting Optuna Hyperparameter & Threshold Tuning for Polymarket...")

# ==============================================================================
# 1. Daten Laden (Enriched 5m Data)
# ==============================================================================
CSV_PATH = 'BTCUSDT_5m_enriched.csv'
if not os.path.exists(CSV_PATH):
    print(f"❌ Fehler: {CSV_PATH} nicht gefunden.")
    exit(1)

df = pd.read_csv(CSV_PATH, parse_dates=['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)
df.columns = [c.lower() for c in df.columns]

# ==============================================================================
# 2. Feature Engineering (Bestätigte Mikrostruktur + Mixed-Frequency On-Chain)
# ==============================================================================
close = df['close']
high = df['high']
low = df['low']
volume = df['volume']

# VWAP (Volume Weighted Average Price)
df['typical_price'] = (high + low + close) / 3
df['date_only'] = df.index.date
df['cum_vol'] = df.groupby('date_only')['volume'].cumsum()
df['cum_vol_price'] = df.groupby('date_only').apply(lambda x: (x['typical_price'] * x['volume']).cumsum()).reset_index(level=0, drop=True)
df['vwap'] = df['cum_vol_price'] / df['cum_vol']
df['dist_to_vwap'] = (close - df['vwap']) / df['vwap']

# MFI (Money Flow Index - 14)
typical_price = (high + low + close) / 3
raw_money_flow = typical_price * volume
pos_flow = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0), index=df.index)
neg_flow = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0), index=df.index)
money_ratio = pos_flow.rolling(window=14).sum() / (neg_flow.rolling(window=14).sum() + 1e-8)
df['mfi_14'] = 100 - (100 / (1 + money_ratio))

# BB Distanz
sma_20 = close.rolling(window=20).mean()
std_20 = close.rolling(window=20).std()
df['dist_to_bb_upper'] = (close - (sma_20 + 2*std_20)) / (sma_20 + 2*std_20)
df['dist_to_bb_lower'] = (close - (sma_20 - 2*std_20)) / (sma_20 - 2*std_20)

# Buying Pressure Proxy
df['buying_pressure'] = (((close - low) - (high - close)) / (high - low + 1e-8)) * volume
df['buying_pressure_ema_5'] = df['buying_pressure'].ewm(span=5, adjust=False).mean()

# On-Chain Mixed Frequency
fees_1h_sum = df['total_fees_btc'].rolling(window=12, min_periods=1).sum()
fees_4h_sum = df['total_fees_btc'].rolling(window=48, min_periods=1).sum()
df['fee_momentum_ratio'] = fees_1h_sum / ((fees_4h_sum / 4) + 1e-8)
df['tx_1h_sum'] = df['tx_count'].rolling(window=12, min_periods=1).sum()
df['blocksize_1h_avg'] = df['avg_block_size'].rolling(window=12, min_periods=1).mean()

# Basis
df['returns'] = close.pct_change()
df['volatility_20'] = df['returns'].rolling(window=20).std()

# Cleanup
df.drop(['typical_price', 'date_only', 'cum_vol', 'cum_vol_price', 'total_fees_btc', 'tx_count', 'avg_block_size', 'buying_pressure'], axis=1, inplace=True)
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df = df.iloc[60:-1].copy()

features = ['dist_to_vwap', 'mfi_14', 'dist_to_bb_upper', 'dist_to_bb_lower', 
            'buying_pressure_ema_5', 'returns', 'volatility_20', 
            'fee_momentum_ratio', 'tx_1h_sum', 'blocksize_1h_avg']

# ==============================================================================
# 3. Optuna Objective (Maximierung des ROI nach Gebühren/Slippage)
# ==============================================================================
# Letzte 3 Jahre für das Tuning
train_df = df.iloc[-300000:] 

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'gamma': trial.suggest_float('gamma', 0, 3),
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Threshold Suche (etwas breiter)
    conf_threshold = trial.suggest_float('conf_threshold', 0.60, 0.85)
    
    # TimeSeriesSplit (3 Folds)
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    
    fold_rois = []
    total_trades = 0
    
    X = train_df[features].values
    y = train_df['target'].values
    
    for train_index, test_index in tscv.split(X):
        X_train_raw, X_test_raw = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_test)[:, 1]
        
        capital = 1000.0
        slippage = 0.002 # 0.2% Slippage (Realistisch für Maker/Limit Orders)
        fee_rate = 0.0   # 0.0% Fees (Maker Ziel)
        
        for p, actual in zip(probs, y_test):
            p_win = p if p > 0.5 else (1-p)
            if p_win >= conf_threshold:
                # Realistischerer Marktpreis bei Maker-Orders
                market_price = 0.50 + ((p_win - 0.50) * 0.3) 
                eff_price = market_price + slippage + (market_price * fee_rate)
                
                if eff_price >= 0.99: continue
                
                b = (1.0 - eff_price) / eff_price
                q = 1.0 - p_win
                # Kelly (sehr konservativ)
                kelly = (p_win * b - q) / (b + 1e-8)
                bet_size = min(max(kelly * 0.2, 0), 0.05) # Max 5% pro Trade
                
                if bet_size > 0:
                    bet_amount = capital * bet_size
                    if (p > 0.5 and actual == 1) or (p <= 0.5 and actual == 0):
                        capital += bet_amount * b
                    else:
                        capital -= bet_amount
                    total_trades += 1
        
        fold_rois.append((capital - 1000.0) / 1000.0)
                
    # STATISTISCHE SIGNIFIKANZ STRAFE:
    # Wir brauchen mindestens 200 Trades über alle Folds, um statistisch valide zu sein.
    if total_trades < 200:
        return -999.0 
        
    return float(np.mean(fold_rois))

# ==============================================================================
# 4. Tuning Starten
# ==============================================================================
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50) # Mehr Versuche


print("\n" + "="*50)
print("🏆 OPTUNA BEST PARAMS")
print("="*50)
print(f"Bester Gewinn im Testzeitraum: ${study.best_value:,.2f}")
print(f"Beste Parameter: {study.best_params}")

# Speichere die besten Parameter für das finale Training
with open('best_optuna_params.json', 'w') as f:
    json.dump(study.best_params, f)
