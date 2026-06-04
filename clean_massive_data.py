import pandas as pd
import numpy as np
import os
import time

INPUT_FILE = 'BTCUSDT_5m_enriched.csv'
OUTPUT_FILE = 'BTCUSDT_5m_enriched_clean.csv'

def clean_massive_data():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Fehler: {INPUT_FILE} nicht gefunden.")
        return

    print(f"📥 Lade massive Daten von {INPUT_FILE}...")
    start_time = time.time()
    
    # Einlesen (70MB sind für Pandas kein Problem im RAM)
    df = pd.read_csv(INPUT_FILE)
    
    print(f"🔄 Ersetze 0.0 durch NaN in On-Chain Spalten...")
    # On-Chain Spalten identifizieren
    on_chain_cols = ['total_fees_btc', 'tx_count', 'avg_block_size']
    
    # 0.0 durch NaN ersetzen
    df[on_chain_cols] = df[on_chain_cols].replace(0.0, np.nan)
    
    # Lücken füllen (Forward Fill, dann Backward Fill für den Anfang)
    print(f"🧹 Wende Forward Fill & Backward Fill an...")
    df[on_chain_cols] = df[on_chain_cols].ffill().bfill()
    
    print(f"💾 Speichere bereinigte Daten in {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    duration = time.time() - start_time
    print(f"✅ Fertig! Daten bereinigt und gespeichert in {duration:.2f} Sekunden.")

if __name__ == "__main__":
    clean_massive_data()
