import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)
lastTradeDay = np.zeros(nInst)
entryPrice = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos, lastTradeDay, entryPrice
    (nInst, nt) = prcSoFar.shape
    
    if nt < 20:
        currentPos = np.zeros(nInst)
        return currentPos
    
    logReturns = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])
    
    momentum = prcSoFar[:, -1] / prcSoFar[:, -20] - 1
    
    volWindow = 20
    volatility = np.std(logReturns[:, -volWindow:], axis=1)
    volatility = np.where(volatility == 0, 1e-6, volatility)
    
    sma_5 = np.mean(prcSoFar[:, -5:], axis=1)
    
    currentPrices = prcSoFar[:, -1]
    
    momentumRank = np.argsort(np.abs(momentum))
    topN = 10
    topIdx = momentumRank[-topN:]
    
    targetPos = np.zeros(nInst)
    
    for i in range(nInst):
        price = currentPrices[i]
        sma = sma_5[i]
        mom = momentum[i]
        vol = volatility[i]
        
        if i in topIdx and abs(mom) > 0.09:
            if mom > 0 and price > sma:
                targetPos[i] = 1
            elif mom < 0 and price < sma:
                targetPos[i] = -1
            else:
                targetPos[i] = 0
        else:
            targetPos[i] = 0
        
        if currentPos[i] != 0 and entryPrice[i] != 0:
            priceChange = (price - entryPrice[i]) / entryPrice[i]
            volStop = 1.0 * vol
            if (currentPos[i] > 0 and priceChange < -volStop) or \
               (currentPos[i] < 0 and priceChange > volStop):
                targetPos[i] = 0
        
        if targetPos[i] != 0:
            maxShares = np.floor(10000 / price)
            trendStrength = abs(mom)
            invVol = 1 / vol
            selectedIdx = topIdx
            normFactor = np.sum(1 / volatility[selectedIdx]) if len(selectedIdx) > 0 else 1
            shares = np.floor(trendStrength * invVol * 10000 / price / normFactor)
            targetPos[i] = targetPos[i] * min(maxShares, shares)
    
    holdPeriod = 5
    newPos = np.copy(currentPos)
    currentDay = nt - 1
    for i in range(nInst):
        if (currentDay - lastTradeDay[i] >= holdPeriod or targetPos[i] == 0):
            newPos[i] = targetPos[i]
            if newPos[i] != currentPos[i]:
                lastTradeDay[i] = currentDay
                entryPrice[i] = price if newPos[i] != 0 else 0
    
    currentPos = newPos.astype(int)
    return currentPos