import os, sys, time, collections, io
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
        #return '/USDT' in symbol
        sym1, sym2 = symbol.split('/')
        if sym2 == 'USDT': return True
        if sym2 == 'BTC':  return sym1+'/USDT' not in tickers

    symbols = [symbol for symbol in tickers if keep(symbol)]
    symbols.insert(0, symbols.pop(symbols.index('BTC/USDT'))) # Used to covert BTC to USD for later coins
    return symbols


def get_best_coins(symbols):
    print('Checking price history and choosing best coins...')
    class Coin(collections.namedtuple("Coin", "name gain goodness")): pass
    coins = []
    for symbol in symbols:
        ohlcv = binance.fetch_ohlcv(symbol, f'4h', limit=19)
        prices = [day[3] for day in ohlcv]
        milli_seconds_in_hour = 60*60*1000
        times = [day[0] / milli_seconds_in_hour for day in ohlcv]

        prices.append(tickers[symbol]['last'])
        times.append(tickers[symbol]['timestamp'] / milli_seconds_in_hour)
        assert times[-1] - times[-2] < 4.1

        #print([time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t*3600)) for t in times])

        # Make time in past negative
        times = [time-times[-1] for time in times]

        if symbol == 'BTC/USDT':
            btc_to_usd = prices
        elif '/BTC' in symbol:
            prices = [a*b for a,b in zip(prices, btc_to_usd)]

        predict_time = times[-1]+24
        fit = np.polyfit(times, prices, 2)
        expected = np.polyval(fit, predict_time)
        gain = (expected - prices[-1]) / prices[-1]

        change   = (prices[-1] - prices[0]) / prices[-1]
        goodness = gain - change
        if gain < 0: goodness += gain * 19*4/24 # Extra penalty for downward trend

        #print(symbol, gain, goodness)

        coin = Coin(symbol.split('/')[0], gain, goodness)
        coins.append(coin)

        # For plotting
        coin.times  = times
        coin.prices = prices
        coin.fit    = fit
        coin.predict_time = predict_time

    coins.sort(key=lambda coin: coin.goodness, reverse=True)
    return coins


def buy_coin(coin):
    def to_btc(coin, value):
        if coin == 'BTC':
            return value
        if coin == 'USDT':
            return value / tickers["BTC/USDT"]['last']
        return value * tickers[f"{coin}/BTC"]['last']

    def to_usdt(coin, value):
        return to_btc(coin, value) * tickers["BTC/USDT"]['last']

    def get_balance():
        balance = binance.fetch_balance()
        balance = [(coin, info['total']) for coin, info in balance.items() if coin == coin.upper()]
        balance = [(coin, value, to_usdt(coin, value), to_btc(coin, value)) for coin, value in balance if value]
        return max(balance, key=lambda item: item[2])

    tickers = binance.fetch_tickers()  # In case previous code took a while
    holding, amount_coin, amount_usdt, amount_btc = get_balance()

    action = f'{holding} transferred to {coin}...' if coin != holding else f'HODL {coin}'
    if coin != holding:
        print(f'Transferring {holding} to {coin}...')
        if holding != 'BTC':
            if holding == 'USDT':
                side   = 'buy'
                symbol = 'BTC/USDT'
                amount = amount_btc * .999
            else:
                side   = 'sell'
                symbol = f"{holding}/BTC"
                amount = amount_coin

            print(f"{side} {amount} {symbol}")
            # TODO apparently limit is better
            binance.create_order(symbol, 'market', side, amount)
            holding, amount_coin, amount_usdt, amount_btc = get_balance()
            assert holding == 'BTC', holding

        if coin != 'BTC':
            if coin == 'USDT':
                side   = 'sell'
                symbol = 'BTC/USDT'
                amount = amount_btc
            else:
                side   = 'buy'
                symbol = f"{coin}/BTC"
                amount = amount_btc / tickers[symbol]['last'] * .98

            print(f"{side} {amount} {symbol}")
            binance.create_order(symbol, 'market', side, amount)
            holding, amount_coin, amount_usdt, amount_btc = get_balance()
            assert holding == coin, holding

    print(action)
    return action


def email_myself_plots(subject, coins):
    msg = EmailMessage()
    msg['Subject'] = subject
    results = '\n'.join(f"{coin.name} {coin.gain}" for coin in coins[:5])
    msg.set_content(results)

    imgs = ""
    bufs = []
    for coin in coins[:5]:
        plt.title(f"{coin.name}  gain={round(coin.gain * 100, 2)}%, goodness={round(coin.goodness * 100, 2)}")
        plt.xlabel("hours")
        plt.xticks(range(-100 * 24, 10 * 24, 24))
        plt.plot(coin.times, coin.prices, marker='o')
        fit_times = np.linspace(coin.times[0], coin.predict_time, len(coin.times) * 2)
        fit_prices = [np.polyval(coin.fit, time) for time in fit_times]
        plt.plot(fit_times, fit_prices, '--')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        # plt.show()
        plt.clf()

        cid = make_msgid()
        bufs.append((cid, buf))
        imgs += f"<img src='cid:{cid[1:-1]}'>"

    msg.add_alternative(f"<html><body>{imgs}</body></html>", subtype='html')
    for cid, buf in bufs:
        msg.get_payload()[1].add_related(buf.read(), "image", "png", cid=cid)

    email_myself(msg)


def email_myself(msg):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("micah.d.lamb@gmail.com", os.environ['gmail_app_password'])
    server.send_message(msg, "micah.d.lamb@gmail.com", ["micah.d.lamb@gmail.com"])
    server.quit()


if __name__ == '__main__':

    symbols = None

    while True:
        try:
            tickers = binance.fetch_tickers()
            if not symbols:
                symbols = get_symbols()
                print(f"Tracking symbols: {' '.join(symbols)}")

            coins = get_best_coins(symbols)
            best = coins[0].name if coins[0].goodness > 0 else 'USDT'
            result = buy_coin(best)
            email_myself_plots(result, coins)

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
