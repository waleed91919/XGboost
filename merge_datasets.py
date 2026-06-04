import pandas as pd
import numpy as np
import os

def merge_datasets(binance_path='binance_3months_5m.csv', bq_path='BTC_last_90.csv', output_path='BTCUSDT_5m_90d_enriched.csv'):
    print(f"🔄 Zusammenführen von Binance-Preisdaten ({binance_path}) und BigQuery On-Chain Daten ({bq_path})...")

    # Überprüfen ob Dateien existieren
    if not os.path.exists(binance_path):
        print(f"❌ Fehler: {binance_path} nicht gefunden.")
        return
    if not os.path.exists(bq_path):
        print(f"❌ Fehler: {bq_path} nicht gefunden.")
        return

    # 1. Binance Daten laden
    print(f"📥 Lade Binance Daten ({binance_path})...")
    # Wir nehmen an, dass Binance CSV einen Zeitstempel hat (z.B. 'timestamp' oder 'date')
    df_binance = pd.read_csv(binance_path)
    
    # Automatische Erkennung der Datumsspalte
    date_col = None
    for col in ['date', 'timestamp', 'datetime', 'time']:
        if col in df_binance.columns:
            date_col = col
            break
    
    if date_col:
        df_binance[date_col] = pd.to_datetime(df_binance[date_col])
        df_binance.set_index(date_col, inplace=True)
    else:
        print("⚠️ Keine Datumsspalte in Binance-Daten gefunden. Verwende erste Spalte.")
        df_binance.iloc[:, 0] = pd.to_datetime(df_binance.iloc[:, 0])
        df_binance.set_index(df_binance.columns[0], inplace=True)

    df_binance.index = df_binance.index.round('5min')

    # 2. BigQuery Daten laden
    print(f"📥 Lade BigQuery Daten ({bq_path})...")
    df_bq = pd.read_csv(bq_path, parse_dates=['date'])
    df_bq.set_index('date', inplace=True)
    
    # Entferne Zeitzone falls vorhanden
    if df_bq.index.tz is not None:
        df_bq.index = df_bq.index.tz_localize(None)
    
    df_bq.index = df_bq.index.round('5min')

    # 3. Zusammenführen (Left Join)
    print("🔗 Führe Datensätze zusammen...")
    df_merged = df_binance.join(df_bq, how='left')

    # 4. Forward Fill für On-Chain Daten (wie im Live-Bot gefordert)
    print("🧪 Wende Forward Fill (ffill) auf On-Chain Metriken an...")
    onchain_cols = ['total_fees_btc', 'transaction_count', 'avg_block_size']
    # Nur Spalten füllen die auch existieren
    existing_onchain_cols = [c for c in onchain_cols if c in df_merged.columns]
    df_merged[existing_onchain_cols] = df_merged[existing_onchain_cols].ffill().fillna(0)

    # Speichern
    df_merged.to_csv(output_path)
    print(f"✅ Erfolgreich gespeichert als {output_path}")
    print(f"📊 Neues Dataset: {df_merged.shape}")
    print(f"   Features: {list(df_merged.columns)}")

if __name__ == "__main__":
    merge_datasets()
