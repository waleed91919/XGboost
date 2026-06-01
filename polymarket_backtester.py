import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

print("📊 Starting Polymarket Backtester (Phase 3 & 4)...")

# ==============================================================================
# 1. Konfiguration & Annahmen (Realistische Maker-Orders)
# ==============================================================================
INITIAL_CAPITAL = 10000.0
CONFIDENCE_THRESHOLD = 0.601 # Optuna Validierter Threshold (N >= 200)
FRACTIONAL_KELLY = 0.2       # Sehr konservativer Kelly
FEE_RATE = 0.0               # 0.0% Maker Fee Ziel
SLIPPAGE = 0.002             # 0.2% Slippage (Realistisch für Limit Orders)
ASSUMED_MARKET_PRICE = 0.50  # Basis-Preis

# ==============================================================================
# 2. Daten Laden
# ==============================================================================
PREDS_PATH = 'predictions.csv'
if not os.path.exists(PREDS_PATH):
    print(f"❌ Fehler: {PREDS_PATH} nicht gefunden. Führe erst xgboost_polymarket_train.py aus.")
    exit(1)

df = pd.read_csv(PREDS_PATH)
print(f"📥 {len(df)} Vorhersagen geladen.")

# ==============================================================================
# 3. Polymarket Handels-Logik (EV & Kelly Criterion)
# ==============================================================================
capital = INITIAL_CAPITAL
capital_history = []
trades = 0
wins = 0
losses = 0

for index, row in df.iterrows():
    prob_up = row['probability']
    prob_down = 1.0 - prob_up
    
    # Bestimme, auf welche Richtung wir wetten
    if prob_up > prob_down:
        bet_direction = 1
        model_prob = prob_up
    else:
        bet_direction = 0
        model_prob = prob_down
        
    # Phase 3 & 4: Ausführungs-Logik
    if model_prob >= CONFIDENCE_THRESHOLD:
        # REALISTISCHER MARKT-PENALTY:
        # Wir nehmen an, dass wir einen leichten Edge gegenüber dem Marktpreis haben.
        # Wenn das Modell 60% sagt, steht der Marktpreis vielleicht bei 52 Cent.
        # Marktpreis = 0.50 + (model_prob - 0.50) * 0.2
        base_market_price = 0.50 + ((model_prob - 0.50) * 0.2) 
        
        # 1. Berechne den effektiven Preis, den wir zahlen (inklusive Slippage & Fees)
        effective_price = base_market_price + SLIPPAGE + (base_market_price * FEE_RATE)
        
        if effective_price >= 0.99:
            continue
            
        # Polymarket Payout: Wenn wir gewinnen, bekommen wir $1.00 pro Share.
        # Netto-Profit = $1.00 - effective_price
        # Verlust = effective_price
        profit_if_win = 1.0 - effective_price
        loss_if_fail = effective_price
        
        # 2. Expected Value (EV)
        ev = (model_prob * profit_if_win) - ((1.0 - model_prob) * loss_if_fail)
        
        # Nur traden, wenn der EV (nach Gebühren und Slippage) positiv ist
        if ev > 0:
            # 3. Kelly Criterion für Positionsgröße
            # b = Netto-Quoten (Odds) = profit_if_win / loss_if_fail
            b = profit_if_win / loss_if_fail
            q = 1.0 - model_prob
            kelly_pct = (model_prob * b - q) / b
            
            # Verhindere Überhebelung (wir wetten max 10% des Portfolios pro Trade)
            bet_size_pct = min(kelly_pct * FRACTIONAL_KELLY, 0.10)
            
            if bet_size_pct > 0:
                bet_amount = capital * bet_size_pct
                shares_bought = bet_amount / effective_price

                # Trade Auswertung
                actual_direction = row['actual']
                if actual_direction == bet_direction:
                    # Gewonnen!
                    revenue = shares_bought * 1.0
                    capital += (revenue - bet_amount)
                    wins += 1
                else:
                    # Verloren!
                    capital -= bet_amount
                    losses += 1

                trades += 1

                # CAP CAPTIAL AT 10M USD (Realistische Liquiditätsgrenze)
                if capital > 10_000_000:
                    capital = 10_000_000
    # Speichere Kapitalverlauf (jeden Tag / jede Kerze)
    capital_history.append(capital)

# ==============================================================================
# 4. Resultate Anzeigen
# ==============================================================================
print("\n" + "="*50)
print("🏆 BACKTEST ERGEBNISSE (Phase 3 & 4)")
print("="*50)
print(f"Startkapital:       ${INITIAL_CAPITAL:,.2f}")
print(f"Endkapital:         ${capital:,.2f}")
roi = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
print(f"Return on Invest:   {roi:.2f}%")
print(f"Anzahl Trades:      {trades}")
if trades > 0:
    win_rate = (wins / trades) * 100
    print(f"Win Rate:           {win_rate:.2f}% ({wins} Wins / {losses} Losses)")
else:
    print("Keine Trades ausgeführt.")
print("="*50)

# Speichere den Equity-Verlauf
df['equity'] = capital_history
df[['date', 'equity']].to_csv('equity_curve.csv', index=False)
print("💾 Equity Curve gespeichert als 'equity_curve.csv'.")
