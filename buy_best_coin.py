"""
TODO
Merge expected and goodness into expected.
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
        ohlcv = binance.fetch_ohlcv(symbol, f'4h', limit=7*6)
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
                print(f"Skipping {symbol} for bad data. len(times)={len(times)}")
                continue

        #print('USDT', symbol, prices)

        predict_time = times[-1]+24
        times_3day  = times [-3*6:]
        prices_3day = prices[-3*6:]
        fit_3day = np.polyfit(times_3day, prices_3day, 2)
        fit_7day = np.polyfit(times, prices, 1)
        expected_3day = np.polyval(fit_3day, predict_time)
        expected_7day = np.polyval(fit_7day, predict_time)

        current = prices[-1]
        gain_3day = (expected_3day - current) / current
        gain_7day = (expected_7day - current) / current
        change_3day  = (current - prices_3day[0]) / current
        change_8hour = (current - prices[-3]) / current
        change_4hour = (current - prices[-2]) / current

        weight = 3 if gain_3day < 0 else 1
        goodness = (gain_3day * weight - change_3day) * .5 + gain_7day + (change_8hour + change_4hour) * .25

        #print(symbol, gain, goodness)

        coin = Coin(symbol.split('/')[0], gain_3day, goodness)
        coins.append(coin)

        # For plotting
        fit_times = np.linspace(times_3day[0], predict_time, len(times_3day) * 2)

        coin.plots = [
            (times_3day, prices_3day, '-', 'o'),
            (fit_times,  [np.polyval(fit_3day, time) for time in fit_times], '--', None),
            (fit_times,  [np.polyval(fit_7day, time) for time in fit_times], '--', None)
        ]

    coins.sort(key=lambda coin: coin.goodness, reverse=True)
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
    results = '\n'.join(f"{coin.name} {coin.gain}" for coin in coins[:5])
    msg.set_content(results)

    # Show top 5 coins + BTC
    show = coins[:5]
    btc  = next(c for c in coins if c.name == 'BTC')
    if btc not in show:
        show.append(btc)

    imgs = ""
    bufs = []
    for coin in show:
        plt.title(f"{coin.name}  gain={round(coin.gain * 100, 2)}%, goodness={round(coin.goodness * 100, 2)}")
        plt.xlabel("hours")
        plt.xticks(range(-100 * 24, 10 * 24, 24))
        for x, y, linestyle, marker in coin.plots:
            plt.plot(x, y, linestyle, marker=marker)
        x, y, *args = coin.plots[0]
        for a,b in zip(x,y):
            plt.text(a, b, '%g' % b)
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
class Coin(collections.namedtuple("Coin", "name gain goodness")): pass

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
            usdt  = Coin('USDT', 0, 0)
            btc   = next(c for c in coins if c.name == 'BTC')
            hodl  = next(c for c in coins if c.name == holding) if holding != 'USDT' else usdt

            if best == hodl and best.goodness > 0:
                buy = hodl
                result = f'HODL {hodl.name}'
            elif best.goodness < .05:
                buy = btc if btc.goodness > 0 else usdt
                result = f'Fallback from {hodl.name} to {buy.name}' if buy != hodl else f'HODL {hodl.name}'
            elif best.goodness - hodl.goodness < .01:
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
