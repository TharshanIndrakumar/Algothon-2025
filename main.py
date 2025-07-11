import numpy as np
import pandas as pd

# Global state variables to maintain state between calls to getMyPosition
nInst = 50
# position_state: 0 for no position, 1 for long, -1 for short
position_state = np.zeros(nInst, dtype=int)
# entry_price: Price at which the current position was entered
entry_price = np.zeros(nInst)
# tp_level: Tracks take-profit stages (0, 1, 2, 3)
tp_level = np.zeros(nInst, dtype=int)
# initial_pos_size: The total number of shares at the start of a trade
initial_pos_size = np.zeros(nInst)
# NEW: Sleep function state variable
skip_signals_count = np.zeros(nInst, dtype=int)

def getMyPosition(prcSoFar):
    """
    Calculates the target position for each instrument based on a trend-following strategy.

    The strategy uses two moving averages (15-day and 26-day) to generate buy and sell signals,
    and additionally requires the instrument to be in the top 25 by momentum.
    Positions are managed with a multi-stage take-profit system and a volatility-based stop-loss.
    A sleep function skips the next trade after any position is closed.
    """
    global position_state, entry_price, tp_level, initial_pos_size, skip_signals_count

    (n, nt) = prcSoFar.shape
    # Wait for enough data to calculate indicators
    if nt < 27:
        return np.zeros(nInst)

    # Determine the position we should theoretically be holding *before* today's logic
    current_pos = np.zeros(nInst)
    for i in range(nInst):
        if position_state[i] == 1:  # Current state is Long
            if tp_level[i] == 0:
                current_pos[i] = initial_pos_size[i]
            elif tp_level[i] == 1:
                current_pos[i] = np.round(2 * initial_pos_size[i] / 3)
            elif tp_level[i] == 2:
                current_pos[i] = np.round(initial_pos_size[i] / 3)
        elif position_state[i] == -1:  # Current state is Short
            if tp_level[i] == 0:
                current_pos[i] = -initial_pos_size[i]
            elif tp_level[i] == 1:
                current_pos[i] = -np.round(2 * initial_pos_size[i] / 3)
            elif tp_level[i] == 2:
                current_pos[i] = -np.round(initial_pos_size[i] / 3)

    # --- Calculate Indicators ---
    closing_prices = prcSoFar[:, -1]
    ma7 = np.mean(prcSoFar[:, -7:], axis=1)
    ma27 = np.mean(prcSoFar[:, -27:], axis=1)

    # --- Calculate Indicators for stricter entry ---
    momentum = prcSoFar[:, -1] / prcSoFar[:, -10] - 1
    sma_5 = np.mean(prcSoFar[:, -5:], axis=1)
    momentum_rank = np.argsort(np.abs(momentum))
    topN = 30
    top_idx = momentum_rank[-topN:]

    # --- Volatility Calculation (for stop-loss mechanism) ---
    # UPDATED: Using a 15-day lookback for log returns
    log_returns = np.log(prcSoFar[:, -19:] / prcSoFar[:, -20:-1])
    volatility = np.std(log_returns, axis=1)
    volatility = np.where(volatility == 0, 1e-6, volatility) # Avoid division by zero

    new_pos = np.copy(current_pos)

    for i in range(nInst):
        price = closing_prices[i]

        # --- I. MANAGE EXISTING POSITIONS ---
        if position_state[i] != 0:
            is_long = position_state[i] == 1
            entry = entry_price[i]
            trade_closed = False
            
            price_change = (price - entry) / entry
            vol_stop_threshold = volatility[i]

            if is_long:
                # Volatility-based Stop-Loss for Long
                if price_change < (-vol_stop_threshold - 1):
                    new_pos[i] = 0
                    trade_closed = True
                # Take-Profit for Long
                else:
                    if tp_level[i] < 3 and price >= entry * 1.10:
                        new_pos[i], tp_level[i], trade_closed = 0, 3, True
                    elif tp_level[i] < 2 and price >= entry * 1.05:
                        new_pos[i], tp_level[i] = np.round(initial_pos_size[i] / 3), 2
                    elif tp_level[i] < 1 and price >= entry * 1.05:
                        new_pos[i], tp_level[i] = np.round(2 * initial_pos_size[i] / 3), 1
            else:  # is_short
                # Volatility-based Stop-Loss for Short
                if price_change > (vol_stop_threshold + 1):
                    new_pos[i] = 0
                    trade_closed = True
                # Take-Profit for Short
                else:
                    if tp_level[i] < 3 and price <= entry * 0.90:
                        new_pos[i], tp_level[i], trade_closed = 0, 3, True
                    elif tp_level[i] < 2 and price <= entry * 0.95:
                        new_pos[i], tp_level[i] = -np.round(initial_pos_size[i] / 3), 2
                    elif tp_level[i] < 1 and price <= entry * 0.95:
                        new_pos[i], tp_level[i] = -np.round(2 * initial_pos_size[i] / 3), 1

            if trade_closed:
                position_state[i], entry_price[i], tp_level[i], initial_pos_size[i] = 0, 0, 0, 0
                # Activate the sleep function by setting the skip counter
                skip_signals_count[i] = 1

        # --- II. CHECK FOR NEW ENTRIES (with combined, stricter conditions) ---
        if position_state[i] == 0:
            mom = momentum[i]
            sma = sma_5[i]
            
            # Stricter buy signal: Combines MA alignment with momentum ranking and confirmation
            buy_signal = (price > ma7[i] and price > ma27[i]) and \
                         (i in top_idx and abs(mom) > 0.04 and mom > 0 and price > sma)
            
            # Stricter sell signal: Combines MA alignment with momentum ranking and confirmation
            sell_signal = (price < ma7[i] and price < ma27[i]) and \
                          (i in top_idx and abs(mom) > 0.04 and mom < 0 and price < sma)

            # Handle the sleep function
            if skip_signals_count[i] > 0:
                if buy_signal or sell_signal:
                    # A signal occurred, so decrement the counter and skip the trade
                    skip_signals_count[i] -= 1
            else:
                # Not sleeping, so trade normally
                if buy_signal:
                    position_state[i] = 1
                    entry_price[i] = price
                    size = np.floor(10000 / price)
                    initial_pos_size[i] = size
                    new_pos[i] = size
                elif sell_signal:
                    position_state[i] = -1
                    entry_price[i] = price
                    size = np.floor(10000 / price)
                    initial_pos_size[i] = size
                    new_pos[i] = -size
                
    return new_pos.astype(int)