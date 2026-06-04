import pandas as pd
import numpy as np
import os

def clean_data(input_file='BTCUSDT_5m_90d_enriched.csv', output_file='final_data_ready_for_backtest.csv'):
    print(f"🧹 Bereinige Daten aus {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"❌ Fehler: {input_file} nicht gefunden.")
        return

    # 1. Datei einladen
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    
    # Die betroffenen On-Chain Spalten
    onchain_cols = ['total_fees_btc', 'transaction_count', 'avg_block_size']
    
    # Sicherstellen, dass die Spalten existieren
    existing_cols = [c for c in onchain_cols if c in df.columns]
    
    if not existing_cols:
        print("⚠️ Keine On-Chain Spalten zum Bereinigen gefunden.")
        return

    # 2. Ersetze 0.0 durch NaN
    # Wir machen das nur für die On-Chain Spalten, da im Preis oder Volume 0.0 theoretisch (wenn auch unwahrscheinlich) existieren könnte
    print(f"🔍 Ersetze 0.0 durch NaN in: {existing_cols}")
    df[existing_cols] = df[existing_cols].replace(0.0, np.nan)
    
    # 3. Forward Fill (ffill)
    print("➡️ Wende Forward Fill (ffill) an...")
    df[existing_cols] = df[existing_cols].ffill()
    
    # 4. Backward Fill (bfill) für Lücken am Anfang
    print("⬅️ Wende Backward Fill (bfill) für verbleibende Lücken am Start an...")
    df[existing_cols] = df[existing_cols].bfill()
    
    # 5. Speichern
    df.to_csv(output_file)
    print(f"✅ Saubere Daten gespeichert als: {output_file}")
    print(f"📊 Finale Zeilen: {len(df)}")
    
    # Kurzer Check ob noch NaNs da sind
    nans = df[existing_cols].isna().sum().sum()
    if nans == 0:
        print("✨ Erfolg: Alle Lücken wurden gefüllt.")
    else:
        print(f"⚠️ Warnung: Es sind noch {nans} NaN-Werte vorhanden.")

if __name__ == "__main__":
    # Wir nutzen direkt die zuletzt erstellte Datei als Input
    clean_data(input_file='BTCUSDT_5m_90d_enriched.csv')
