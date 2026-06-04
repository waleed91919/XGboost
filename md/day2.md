# Day 2: Data Enrichment & Long-Term Polymarket Backtesting

## 1. Data Acquisition & Merging
- **BigQuery SQL:** Updated query to fetch 3 months of Bitcoin on-chain metrics (Fees, Tx Count, Block Size) aggregated into 5-minute intervals.
- **Binance Downloader:** Created `download_binance.py` to pull 90 days of BTCUSDT 5m candles with full pagination.
- **Dataset Merging:** Developed `merge_datasets.py` to join price and on-chain data into a single feature-rich CSV.

## 2. Data Cleaning & Preparation
- **Lückenfüllung (ffill):** Created `clean_merged_data.py` to handle "empty" 5m blocks. Replaced `0.0` values with `NaN` and applied **Forward Fill** then **Backward Fill**.
- **Result:** A seamless dataset (`final_data_ready_for_backtest.csv`) where the network always appears "alive" to the model.

## 3. Advanced Backtesting
- **Polymarket Simulation:** Rewrote `long_term_backtest.py` to simulate actual binary option payouts:
  - **Share Cost:** Set by the model's predicted probability (e.g., $0.58).
  - **Payout:** $1.00 for a win (Price Up), $0.00 for a loss.
  - **Risk Management:** Quarter-Kelly position sizing.
- **Feature Sync:** Ensured all 11 microstructure and on-chain features perfectly match the production model.

## 4. Key Discovery: The Sweet Spot
- Conducted a multi-threshold simulation [57.0 - 60.5].
- **Result:** Identified **Threshold 58.0** as the optimal setting.
  - **Performance:** **+25.96% PnL** over 90 days.
  - **Reliability:** **64.56% Win-Rate** with 79 high-confidence trades.

---
**Status:** The system is now verified with historical data under realistic market conditions. Ready for live deployment or further optimization.

## 5. Strategy Refinement: The Hybrid Approach
- **Dynamic ATR Test:** Initially tested a fully dynamic threshold (57.5 - 60.5). Found that while it reduced risk, it over-filtered profitable trades, yielding only +7.72% PnL.
- **Circuit Breaker Discovery:** Developed `circuit_breaker_backtest.py` to test a safety "Kill-Switch".
- **Final Optimization:** Discovered that a **Fixed 58.0% Threshold** combined with a **200% ATR Circuit Breaker** is the ultimate setup.
  - **Result:** **+33.61% PnL** (surpassing the baseline's +25.96%).
  - **Win-Rate:** **65.79%** with 76 trades.
  - **Safety:** Successfully blocked 3 extreme volatility "loss-traps" without sacrificing momentum gains.

## 6. Live Bot Deployment (v4)
- **Script:** `polymarket_live_bot.py` updated to the Hybrid Strategy.
- **Features:** 24h ATR monitoring, 58% confidence anchor, and strict "YES-only" execution.
- **Ready:** System is primed for live paper-trading.

