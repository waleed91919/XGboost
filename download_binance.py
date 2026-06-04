import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def download_binance_data(symbol='BTCUSDT', interval='5m', days=90):
    print(f"🚀 Starte Download für {symbol} ({interval}) der letzten {days} Tage...")
    
    url = "https://api.binance.com/api/v3/klines"
    
    # Berechne Startzeitpunkt (90 Tage zurück)
    end_time = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_candles = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': 1000
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_candles.extend(data)
            
            # Update Fortschritt
            last_timestamp = data[-1][0]
            current_start = last_timestamp + 1
            
            # Fortschrittsberechnung
            total_range = end_time - start_time
            current_progress = last_timestamp - start_time
            percent = (current_progress / total_range) * 100
            
            print(f"⏳ Fortschritt: {percent:.2f}% | Letzter Zeitstempel: {datetime.fromtimestamp(last_timestamp/1000)}")
            
            # Kurze Pause für API-Limits
            time.sleep(0.1)
            
            if len(data) < 1000:
                break
                
        except Exception as e:
            print(f"❌ Fehler beim Abruf: {e}")
            time.sleep(1)
            continue

    # DataFrame erstellen
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(all_candles, columns=columns)
    
    # Nur benötigte Spalten behalten und Formatierung anpassen
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Konvertiere Zeitstempel in lesbares Format (optional, aber oft gewünscht)
    # Falls du lieber den rohen Unix-Timestamp willst, kommentiere die nächste Zeile aus
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Sortierung sicherstellen
    df = df.sort_values('timestamp')
    
    # Speichern
    filename = 'binance_3months_5m.csv'
    df.to_csv(filename, index=False)
    
    print(f"✅ Download abgeschlossen! {len(df)} Kerzen gespeichert in {filename}")

if __name__ == "__main__":
    download_binance_data()
