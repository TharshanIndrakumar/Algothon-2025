import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)

MOM_LOOKBACK = 20
TOPN = 5
POS_LIMIT = 10000
COMMISSION = 0.0005
VOL_LOOKBACK = 20
MIN_VOL = 0.01  # Minimum volatility to trade
MIN_TRADE_VAL = 2 * COMMISSION * POS_LIMIT


def getMyPosition(prcSoFar):
    global currentPos
    (nInst, nt) = prcSoFar.shape

    if nt < MOM_LOOKBACK + 1:
        currentPos = np.zeros(nInst)
        return currentPos

    # Calculate 20-day momentum
    momentum = prcSoFar[:, -1] / prcSoFar[:, -MOM_LOOKBACK] - 1
    # Calculate volatility
    logReturns = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])
    volatility = np.std(logReturns[:, -VOL_LOOKBACK:], axis=1)
    volatility = np.where(volatility == 0, 1e-6, volatility)

    # Volatility filter
    tradable = volatility > MIN_VOL
    tradable_idx = np.where(tradable)[0]
    if len(tradable_idx) < 2 * TOPN:
        # Not enough tradable instruments, hold cash
        currentPos = np.zeros(nInst)
        return currentPos

    # Rank by momentum among tradable
    tradable_mom = momentum[tradable]
    ranks = np.argsort(tradable_mom)
    long_idx = tradable_idx[ranks[-TOPN:]]
    short_idx = tradable_idx[ranks[:TOPN]]

    # Equal dollar allocation
    currentPrices = prcSoFar[:, -1]
    dollar_per_pos = POS_LIMIT  # Max $10k per instrument
    targetPos = np.zeros(nInst)
    for i in long_idx:
        shares = np.floor(dollar_per_pos / currentPrices[i])
        targetPos[i] = shares
    for i in short_idx:
        shares = np.floor(dollar_per_pos / currentPrices[i])
        targetPos[i] = -shares

    # Market neutrality: adjust so net dollar exposure is zero
    net_dollar = np.sum(targetPos * currentPrices)
    if abs(net_dollar) > 0:
        # Remove from largest positions to neutralize
        adjust_order = np.argsort(-np.abs(targetPos))
        for i in adjust_order:
            idx = i
            if targetPos[idx] == 0:
                continue
            pos_dollar = targetPos[idx] * currentPrices[idx]
            if net_dollar * pos_dollar > 0:  # Same sign, can reduce
                reduce_shares = np.ceil(abs(net_dollar) / abs(currentPrices[idx]))
                if abs(targetPos[idx]) < reduce_shares:
                    reduce_shares = abs(targetPos[idx])
                targetPos[idx] -= np.sign(targetPos[idx]) * reduce_shares
                net_dollar = np.sum(targetPos * currentPrices)
                if abs(net_dollar) < 1e-6:
                    break

    # Integer positions, clip to $10k per instrument
    maxShares = np.floor(POS_LIMIT / currentPrices)
    targetPos = np.clip(targetPos, -maxShares, maxShares)
    targetPos = np.round(targetPos).astype(int)

    # Only rebalance if position change is significant (commission-aware)
    newPos = np.copy(currentPos)
    for i in range(nInst):
        if abs(targetPos[i] - currentPos[i]) * currentPrices[i] > MIN_TRADE_VAL:
            newPos[i] = targetPos[i]
        else:
            newPos[i] = currentPos[i]

    currentPos = newPos
    return currentPos
