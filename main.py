import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)
lastTradeDay = np.zeros(nInst)
entryPrice = np.zeros(nInst)

# Parameters
MOM_LOOKBACK = 20
MR_LOOKBACK = 5
VOL_LOOKBACK = 20
TOPN = 8
POS_LIMIT = 10000
COMMISSION = 0.0005
MIN_SIGNAL_THRESH = 0.04
VOL_STOP_MULT = 1.5


def getMyPosition(prcSoFar):
    global currentPos, lastTradeDay, entryPrice
    (nInst, nt) = prcSoFar.shape

    if nt < MOM_LOOKBACK + 1:
        currentPos = np.zeros(nInst)
        lastTradeDay = np.zeros(nInst)
        entryPrice = np.zeros(nInst)
        return currentPos

    # --- Signal Calculation ---
    logReturns = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])
    volatility = np.std(logReturns[:, -VOL_LOOKBACK:], axis=1)
    volatility = np.where(volatility == 0, 1e-6, volatility)

    # Momentum: 20-day
    momentum = prcSoFar[:, -1] / prcSoFar[:, -MOM_LOOKBACK] - 1
    # Mean reversion: deviation from 5-day SMA
    sma_5 = np.mean(prcSoFar[:, -MR_LOOKBACK:], axis=1)
    meanrev = (sma_5 - prcSoFar[:, -1]) / sma_5

    # Ensemble signal
    signal = momentum + meanrev

    # Select top/bottom N by signal
    rank = np.argsort(signal)
    topIdx = rank[-TOPN:]
    bottomIdx = rank[:TOPN]

    targetPos = np.zeros(nInst)
    currentPrices = prcSoFar[:, -1]

    for i in range(nInst):
        sig = signal[i]
        vol = volatility[i]
        price = currentPrices[i]
        maxShares = np.floor(POS_LIMIT / price)
        # Only trade if in top/bottom and signal is strong
        if (i in topIdx and sig > MIN_SIGNAL_THRESH):
            pos = 1
        elif (i in bottomIdx and sig < -MIN_SIGNAL_THRESH):
            pos = -1
        else:
            pos = 0
        # Volatility scaling
        if pos != 0:
            shares = np.floor(abs(sig) * (1/vol) * POS_LIMIT / price)
            shares = min(maxShares, shares)
            targetPos[i] = pos * shares
        else:
            targetPos[i] = 0
        # Volatility stop
        if currentPos[i] != 0 and entryPrice[i] != 0:
            priceChange = (price - entryPrice[i]) / entryPrice[i]
            volStop = VOL_STOP_MULT * vol
            if (currentPos[i] > 0 and priceChange < -volStop) or \
               (currentPos[i] < 0 and priceChange > volStop):
                targetPos[i] = 0

    # Commission-aware: only trade if change is significant
    min_trade_val = 2 * COMMISSION * POS_LIMIT
    newPos = np.copy(currentPos)
    for i in range(nInst):
        if abs(targetPos[i] - currentPos[i]) * currentPrices[i] > min_trade_val:
            newPos[i] = targetPos[i]
            if newPos[i] != currentPos[i]:
                entryPrice[i] = currentPrices[i] if newPos[i] != 0 else 0
                lastTradeDay[i] = nt - 1
        else:
            newPos[i] = currentPos[i]

    currentPos = newPos.astype(int)
    return currentPos