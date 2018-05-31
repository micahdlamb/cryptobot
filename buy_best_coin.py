"""
TODO
Implement calculate_expected that takes a coin parameters and history. Long term time is based on length of list.
Create test function to run code over history and create a plot of balance / time + text of holding coin at each point.
Use hill search to find best parameters.

Investigate using limit buy instead of market buy
"""

import os, sys, time, collections, io, contextlib
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid

import numpy as np
import matplotlib
matplotlib.use('Agg') # So I don't need tkinter
import matplotlib.pyplot as plt

import ccxt

binance = ccxt.binance({
    'apiKey': os.environ['binance_apiKey'],
    'secret': os.environ['binance_secret']
})


def get_symbols():
    def keep(symbol):
        # Not sure why these are missing the /
        if '/' not in symbol:
            print(f'Ignoring {symbol}')
            return False
        coin1, coin2 = symbol.split('/')
        if coin2 == 'USDT': return True
        if coin2 == 'BTC':  return coin1+'/USDT' not in tickers

    symbols = [symbol for symbol in tickers if keep(symbol)]
    symbols.insert(0, symbols.pop(symbols.index('BTC/USDT'))) # Used to covert BTC to USDT for later coins
    return symbols


def get_best_coins():
    print('Checking price history and choosing best coins...')
    coins = []
    for symbol in get_symbols():
        ohlcv = binance.fetch_ohlcv(symbol, f'4h', limit=30*6)
        if len(ohlcv) != 30*6:
            print(f"Skipping {symbol} for missing data. len(ohlcv)={len(ohlcv)}")
            continue
        prices = [day[3] for day in ohlcv]
        milli_seconds_in_hour = 60*60*1000
        times = [day[0] / milli_seconds_in_hour for day in ohlcv]

        prices.append(tickers[symbol]['last'])
        times.append(tickers[symbol]['timestamp'] / milli_seconds_in_hour)
        assert times[-1] - times[-2] < 4.1

        #print([time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t*3600)) for t in times])

        # Make time in past negative
        times = [time-times[-1] for time in times]

        #print('RAW', symbol, prices)

        if symbol == 'BTC/USDT':
            btc_to_usd = prices
            btc_times  = times
        elif '/BTC' in symbol:
            prices = [a*b for a,b in zip(prices, btc_to_usd)]
            if any(abs(a-b) > 2 for a,b in zip(times, btc_times)):
                print(f"Skipping {symbol} for bad times")
                continue

        #print('USDT', symbol, prices)
        fit_days  = [3, 7 ,14, 30]
        fit_times  = [times [-days*6:] for days in fit_days]
        fit_prices = [prices[-days*6:] for days in fit_days]
        fits = [np.polyfit(t, p, 2) for t,p in zip(fit_times, fit_prices)]
        predict_time = times[-1] + 24
        expected = np.average([np.polyval(fit, predict_time) for fit in fits])
        expected = (expected - prices[-1]) / prices[-1]

        coin = Coin(symbol.split('/')[0], expected)
        coins.append(coin)

        plot_times, plot_prices = fit_times[1], fit_prices[1]
        coin.plots = [(plot_times, plot_prices, 'actual', '-', 'o')]
        for days, fit in zip(fit_days, fits):
            times = plot_times[-days*6:]
            fit_times = np.linspace(times[0], predict_time, len(times) * 2)
            coin.plots.append((fit_times, [np.polyval(fit, time) for time in fit_times], f"{days} day fit", '--', None))

    coins.sort(key=lambda coin: coin.expected, reverse=True)
    return coins


def get_balance():
    def to_btc(coin, value):
        if coin == 'BTC':
            return value
        if coin == 'USDT':
            return value / tickers["BTC/USDT"]['last']
        return value * tickers[f"{coin}/BTC"]['last']

    def to_usdt(coin, value):
        return to_btc(coin, value) * tickers["BTC/USDT"]['last']

    balance = binance.fetch_balance()
    balance = [(coin, info['total']) for coin, info in balance.items() if coin == coin.upper()]
    balance = [(coin, value, to_usdt(coin, value), to_btc(coin, value)) for coin, value in balance if value]
    return max(balance, key=lambda item: item[2])


def buy_coin(coin):
    global holding, amount_coin, amount_usdt, amount_btc

    def buy(coin):
        global holding, amount_coin, amount_usdt, amount_btc

        if f"{holding}/{coin}" in tickers:
            side   = 'sell'
            symbol = f"{holding}/{coin}"
            amount = amount_coin * .98
        else:
            side   = 'buy'
            symbol = f"{coin}/{holding}"
            amount = amount_coin / tickers[symbol]['last'] * .98
            assert symbol in tickers

        print(f"{side} {amount} {symbol}")
        # TODO apparently limit is better
        result = binance.create_order(symbol, 'market', side, amount)
        print(result)
        holding, amount_coin, amount_usdt, amount_btc = get_balance()
        assert holding == coin, holding

    print(f'Transferring {holding} to {coin}...')
    if f"{holding}/{coin}" not in tickers and f"{coin}/{holding}" not in tickers:
        buy('BTC')
    buy(coin)


def email_myself_plots(subject, coins, log):
    msg = EmailMessage()
    msg['Subject'] = subject
    results = '\n'.join(f"{coin.name} {coin.expected}" for coin in coins[:5])
    msg.set_content(results)

    # Show top 5 coins + BTC
    show = coins[:5]
    btc  = next(c for c in coins if c.name == 'BTC')
    if btc not in show:
        show.append(btc)

    imgs = ""
    bufs = []
    for coin in show:
        plt.title(f"{coin.name}  {round(coin.expected * 100, 2)}%")
        plt.xlabel("hours")
        plt.xticks(range(-100 * 24, 10 * 24, 24))
        for x, y, label, linestyle, marker in coin.plots:
            plt.plot(x, y, linestyle, marker=marker, label=label)
        #x, y, *args = coin.plots[0]
        #for a,b in zip(x,y):
        #    plt.text(a, b, '%g' % b)
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        # plt.show()
        plt.clf()

        cid = make_msgid()
        bufs.append((cid, buf))
        imgs += f"<img src='cid:{cid[1:-1]}'>"

    msg.add_alternative(f"<html><body>{imgs}<br><pre>{log}</pre></body></html>", subtype='html')
    for cid, buf in bufs:
        msg.get_payload()[1].add_related(buf.read(), "image", "png", cid=cid)

    email_myself(msg)


def email_myself(msg):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("micah.d.lamb@gmail.com", os.environ['gmail_app_password'])
    server.send_message(msg, "micah.d.lamb@gmail.com", ["micah.d.lamb@gmail.com"])
    server.quit()


# MAIN #################################################################################################################
class Coin(collections.namedtuple("Coin", "name expected")): pass

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for file in self.files:
            file.write(data)

while True:
    try:
        tickers = binance.fetch_tickers()
        holding, amount_coin, amount_usdt, amount_btc = get_balance()

        with io.StringIO() as log, contextlib.redirect_stdout(Tee(log, sys.stdout)):
            coins = get_best_coins()
            best  = coins[0]
            usdt  = Coin('USDT', 0)
            btc   = next(c for c in coins if c.name == 'BTC')
            hodl  = next(c for c in coins if c.name == holding) if holding != 'USDT' else usdt

            if best.expected < .03:
                buy = btc if btc.expected > 0 else usdt
                result = f'Fallback from {hodl.name} to {buy.name}' if buy != hodl else f'HODL {hodl.name}'
            elif best.expected - hodl.expected < .01:
                buy = hodl
                result = f'HODL {hodl.name}'
            else:
                buy = best
                result = f'{hodl.name} transferred to {best.name}'

            try:
                if buy != hodl:
                    buy_coin(buy.name)
            except:
                result += '...failed'
                raise
            finally:
                print(result)
                email_myself_plots(result, coins, log.getvalue())

    except:
        import traceback
        error = traceback.format_exc()
        print(error, file=sys.stderr)
        msg = EmailMessage()
        msg['Subject'] = 'ERROR'
        msg.set_content(error)
        email_myself(msg)

    time.sleep(3600*4)
    print('-'*30 + '\n')
