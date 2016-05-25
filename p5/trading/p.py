



# normed = prices / prices[0]
# alloced = normed * allocs
# pos_vals = allocated * start_val

# port_val = pos_vals.sum(axis=1)

# daily_rets = a / a.shift(1)

# daily_rets = daily_rets[1:]
# cum_ret = (port_val[-1] / port_val[0]) -1
# avg_daily_ret = daily_rets.mean()
# std_daily_ret = daily_rets.std()

# # risk adjusted return
# sharpe_ratio = 


# risk = volatility = std!




adr = 0.001
drf = 0.0002
sdr = 0.001

import math

k = math.sqrt(252)
print k * adr / sdr