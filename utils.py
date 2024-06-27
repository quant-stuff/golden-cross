import numpy as np

# statistic utils
def calculate_DD(returns):
    """
    Calculates various drawdown metrics, such as maximum drawdown, maximum drawdown duration, date of maximum drawdown, average drawdown and drawdown array
    based on the provided returns.
    """
    returns = returns.dropna()

    cum_returns = returns.cumsum()

    highwatermark = np.zeros(cum_returns.shape)
    drawdown = np.zeros(cum_returns.shape)
    drawdown_duration = np.zeros(cum_returns.shape)
    
    for t in np.arange(1, len(cum_returns)):
        highwatermark[t] = np.maximum(highwatermark[t-1], cum_returns.iloc[t])
        drawdown[t] = (1 + cum_returns.iloc[t]) / (1 + highwatermark[t]) - 1
        
        if drawdown[t] == 0:
            drawdown_duration[t] = 0
        else:
            drawdown_duration[t] = drawdown_duration[t-1] + 1

    maxDD = np.min(drawdown)
    avgDD = np.mean(drawdown)
    i = np.argmin(drawdown)
    maxDDD = np.max(drawdown_duration)
    
    return maxDD, maxDDD, avgDD, returns.index[i], drawdown

# 