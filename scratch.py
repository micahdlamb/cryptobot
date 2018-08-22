def get_coin_forecasts():
    print('Forecasting coin prices...')
    class Coin(collections.namedtuple("Coin", "name symbol expected_lt")): pass
    coins = []
    for symbol in symbols():
        times, prices = get_prices(symbol, '1h', limit=8*24)
        if len(times) < 4*24:
            print(f"Skipping {symbol} for missing data. len(ohlcv)={len(times)}")
            continue

        fit_days = [1, 2, 4, 8]
        fit_degs = [1, 1, 1, 1]
        fit_times  = [times [-days*24:] for days in fit_days]
        fit_prices = [prices[-days*24:] for days in fit_days]
        fits = [np.polyfit(t, p, deg) for t,p,deg in zip(fit_times, fit_prices, fit_degs)]
        predict_time = times[-1] + 4
        expected = np.average([np.polyval(fit, predict_time) for fit in fits])

        plot_times, plot_prices = times[-16:], prices[-16:]
        plots = {"lt actual": (plot_times, plot_prices, dict(linestyle='-', marker='o'))}
        for days, fit in zip(fit_days, fits):
            times = plot_times[-days*24:]
            fit_times  = np.linspace(times[0], times[-1], len(times) * 2)
            fit_prices = [np.polyval(fit, time) for time in fit_times]
            plots[f"{days} day fit"] = (fit_times, fit_prices, dict(linestyle='--'))


        name = symbol.split('/')[0]
        coin = Coin(name, symbol, expected) # coin.gain set in get_best_coins
        coin.plots = plots
        coin.zero_time = times[-1]
        coins.append(coin)

    return coins


def get_best_coins(coins):
    print('Looking for best coins...')
    tickers = binance.fetch_tickers()
    for coin in coins:
        times, prices = get_prices(coin.symbol, '5m', limit=8*12)

        fit_hours = [1, 2, 4, 8]
        fit_degs  = [1, 1, 1, 1]
        fit_times  = [times [-hour*12:] for hour in fit_hours]
        fit_prices = [prices[-hour*12:] for hour in fit_hours]
        fits = [np.polyfit(t, p, deg) for t,p,deg in zip(fit_times, fit_prices, fit_degs)]
        predict_time = times[-1] + 4
        coin.expected_st = np.average([np.polyval(fit, predict_time) for fit in fits])

        coin.plots["st actual"] = times, prices, dict(linestyle='-')
        for hours, fit in zip(fit_hours, fits):
            fit_times  = np.linspace(times[-hours*12], times[-1], 2)
            fit_prices = [np.polyval(fit, time) for time in fit_times]
            coin.plots[f"{hours} hour fit"] = (fit_times, fit_prices, dict(linestyle='--'))

        price = tickers[coin.symbol]['last']
        assert price > 0, coin.symbol
        coin.gain_st = (coin.expected_st - price) / price
        coin.gain_lt = (coin.expected_lt - price) / price
        coin.gain = mix(coin.gain_lt, -coin.gain_st, .5)

        coin.trend = np.polyfit(times[-48:], prices[-48:], 1)[0] / price
        coin.dy_dx = np.polyfit(times[ -6:], prices[ -6:], 1)[0] / price

    coins.sort(key=lambda coin: coin.gain, reverse=True)
    print('\n'.join(f"{coin.name}: {percentage(coin.gain)} lt={percentage(coin.gain_lt)} st={percentage(coin.gain_st)} "
                    f"dy/dx={percentage(coin.dy_dx)}/h" for coin in coins[:4]))
    return coins


def get_most_volatile_coins():
    print('Finding most spiky coins')
    class Coin(collections.namedtuple("Coin", "name goodness fit amplitude")): pass
    coins = []
    for symbol in symbols():
        ohlcv = binance.fetch_ohlcv(symbol, '15m', limit=6)
        prices = [np.average(candle[2:4]) for candle in ohlcv]
        times = [candle[0]/milli_seconds_in_minute for candle in ohlcv]
        fit = np.polyfit(times, prices, 1)
        amplitude = (sum(abs(candle[3]-candle[2]) for candle in ohlcv) - abs(fit[0]*30)) / len(ohlcv)
        assert amplitude > -1e-5
        goodness = min(fit[0]*15, .01) + amplitude

        volume = tickers[coin.symbol]['quoteVolume']
        if volume < 250:
            coin.goodness *= volume / 250

        name = symbol.split('/')[0]
        current_price = ohlcv[-1][-2]
        coin = Coin(name, goodness, fit[0]/current_price, amplitude/current_price)
        coins.append(coin)

    coins.sort(key=lambda coin: coin.goodness, reverse=True)
    print('\n'.join(f"{coin.name}: amplitude={percentage(coin.amplitude)} fit={percentage(coin.fit*60)}/h"
                    for coin in coins[:4]))
    return coins


starting_balance = get_balance()
for i in range(4):
    holding = get_holding_coin().name

    ohlcv = binance.fetch_ohlcv('TUSD/BTC', '5m', limit=12)
    prices = [np.average(candle[2:4]) for candle in ohlcv]
    times = [candle[0] for candle in ohlcv]
    fit = np.polyfit(times, prices, 1)

    best = 'TUSD' if fit[0] > 0 else 'BTC'
    if best != holding:
        try:
            trade_coin(holding, 'BTC' if holding != 'BTC' else best)
        except TimeoutError as error:
            print(f"Timeout: {error}")
    else:
        time.sleep(15*60)

ending_balance = get_balance()
gain = (ending_balance.btc - starting_balance.btc) / starting_balance.btc
result = f"Buy/sell TUSD ₿{percentage(gain)}"
print(result)
msg = EmailMessage()
msg['Subject'] = result
trades = '\n'.join(f"{t['side']} {t['symbol']} at {t['price']}" for t in reversed(trade_log))
msg.set_content(trades+'\n'+log.getvalue())
email_myself(msg)


coins = get_most_volatile_coins()
spiky = coins[0].name
print(f"Buy/sell {spiky}")
starting_balance = get_balance()
for i in range(4):
    holding = get_holding_coin().name
    try:
        trade_coin(holding, spiky if holding == 'BTC' else 'BTC', max_wait_minutes=30)
    except TimeoutError as error:
        print(f"Timeout: {error}")

ending_balance = get_balance()
gain = (ending_balance.btc - starting_balance.btc) / starting_balance.btc
result = f"Buy/sell {spiky} ₿{percentage(gain)}"
print(result)
msg = EmailMessage()
msg['Subject'] = result
trades = '\n'.join(f"{t['side']} {t['symbol']} at {t['price']}" for t in reversed(trade_log))
msg.set_content(trades+'\n'+log.getvalue())
email_myself(msg)


import numpy as np
from matplotlib import pyplot as plt

candles = binance.fetch_ohlcv('MCO/BTC', f'1h', limit=7*24)
times = [candle[0]/milli_seconds_in_hour for candle in candles]
times = [time-times[0] for time in times]
prices = [np.average(candle[1:-1]) for candle in candles]

def f(Y,x):
    total = 0
    N = len(Y)
    #for ctr in itertools.chain(range(20), range(N-20, N)):
    for ctr in range(N):
        total += Y[ctr] * (np.cos(x*ctr*2*np.pi/N) + 1j*np.sin(x*ctr*2*np.pi/N))
    return np.real(total)

plt.figure()
plt.plot(prices)
Y=np.fft.fft(prices)/len(prices)
N=len(Y)
xs = range(N, N+8)
plt.plot(xs, [f(Y, x) for x in xs])
plt.show()