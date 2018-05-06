import os, time, collections, io

import numpy as np
import matplotlib.pyplot as plt

import ccxt

binance = ccxt.binance()
tickers = binance.fetch_tickers()
def keep(symbol):
    return '/USDT' in symbol
    sym1, sym2 = symbol.split('/')
    if sym2 == 'USDT': return True
    if sym2 == 'BTC':  return sym1+'/USDT' not in tickers

symbols = [symbol for symbol in tickers if keep(symbol)]
symbols.insert(0, symbols.pop(symbols.index('BTC/USDT'))) # Used to covert BTC to USD for later coins
print(f"Tracking symbols: {' '.join(symbols)}")

wait = lambda: time.sleep(binance.rateLimit/1000)
class Coin(collections.namedtuple("Coin", "name gain")): pass
best = None

while True:
    tickers = binance.fetch_tickers(symbols)
    coins = []
    for symbol in symbols:
        ohlcv = binance.fetch_ohlcv(symbol, f'4h', limit=19)
        prices = [day[3] for day in ohlcv]
        milli_seconds_in_hour = 60*60*1000
        times = [day[0] / milli_seconds_in_hour for day in ohlcv]

        prices.append(tickers[symbol]['last'])
        times.append(tickers[symbol]['timestamp'] / milli_seconds_in_hour)
        assert times[-1] - times[-2] < 4.1

        # Make time in past negative
        times = [time-times[-1] for time in times]

        if symbol == 'BTC/USDT':
            btc_to_usd = prices
        elif '/BTC' in symbol:
            prices = [a*b for a,b in zip(prices, btc_to_usd)]

        predict_time = times[-1]+24
        fit = np.polyfit(times, prices, 2)
        expected = np.polyval(fit, predict_time)

        coin = Coin(symbol.split('/')[0], (expected - prices[-1]) / prices[-1])
        coins.append(coin)

        # For plotting
        coin.times  = times
        coin.prices = prices
        coin.fit    = fit
        coin.predict_time = predict_time

    coins.sort(key=lambda coin: coin.gain, reverse=True)
    results = '\n'.join(f"{coin.name} {coin.gain}" for coin in coins)
    print(results)

    newBest = coins[0].name if coins[0].gain > 0 else 'USDT'
    if newBest != best:
        best = newBest

        import smtplib
        from email.message import EmailMessage
        from email.utils import make_msgid

        msg = EmailMessage()
        msg['Subject'] = f"Buy {best}"
        msg.set_content(results)

        imgs = ""
        bufs = []
        for coin in coins[:5]:
            plt.title(f"{coin.name} ({round(coin.gain * 100, 2)}%)")
            plt.xlabel("hours")
            plt.xticks(range(-100*24, 10*24, 24))
            plt.plot(coin.times, coin.prices, marker='o')
            fit_times = np.linspace(coin.times[0], coin.predict_time, len(coin.times)*2)
            fit_prices = [np.polyval(coin.fit, time) for time in fit_times]
            plt.plot(fit_times, fit_prices, '--')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            #plt.show()
            plt.clf()

            cid = make_msgid()
            bufs.append((cid, buf))
            imgs += f"<img src='cid:{cid[1:-1]}'>"

        msg.add_alternative(f"<html><body>{imgs}</body></html>", subtype='html')
        for cid, buf in bufs:
            msg.get_payload()[1].add_related(buf.read(), "image", "png", cid=cid)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login("micah.d.lamb@gmail.com", os.environ['gmail_app_password'])
        header = f"Subject: Buy {coins[0].name}\n\n"
        server.send_message(msg, "micah.d.lamb@gmail.com", ["micah.d.lamb@gmail.com"])
        server.quit()

    time.sleep(3600)
    print('-'*30 + '\n')
