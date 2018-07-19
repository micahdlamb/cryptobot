"""
TODO
Implement calculate_expected that takes a coin parameters and history. Long term time is based on length of list.
Create test function to run code over history and create a plot of balance / time + text of holding coin at each point.
Use hill search to find best parameters.

Ideas to try:
Look into how altcoin and BTC change relative to each other
Show latest tickers in plot.  Show times of buys and sells.
Always have a sell order going for alt coins
Investigate multiple limit orders at once
"""

import os, sys, time, collections, io, contextlib, math
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

milli_seconds_in_hour = 60*60*1000

def get_coin_forecasts():
    print('Forecasting coin prices...')
    class Coin(collections.namedtuple("Coin", "name symbol expected_lt")): pass
    coins = []
    symbols = ['BTC/USDT'] + [symbol for symbol in tickers if symbol.endswith('/BTC')]
    for symbol in symbols:
        ohlcv = binance.fetch_ohlcv(symbol, f'4h', limit=30*6)
        if len(ohlcv) < 30*4:
            print(f"Skipping {symbol} for missing data. len(ohlcv)={len(ohlcv)}")
            continue
        prices = [candle[3] for candle in ohlcv]
        times  = [candle[0] / milli_seconds_in_hour for candle in ohlcv]
        # Make time in past negative
        zero_time = times[-1]
        times = [time-zero_time for time in times]

        fit_days  = [3, 7 ,14, 30]
        fit_times  = [times [-days*6:] for days in fit_days]
        fit_prices = [prices[-days*6:] for days in fit_days]
        fits = [np.polyfit(t, p, 2) for t,p in zip(fit_times, fit_prices)]
        predict_time = times[-1] + 4
        expected = np.average([np.polyval(fit, predict_time) for fit in fits])

        # Make expected price more realistic...
        #current    = tickers[symbol]['last']
        #difference = expected - current
        #expected   = current + difference/2

        name = symbol.split('/')[0]
        coin = Coin(name, symbol, expected) # coin.gain set in get_best_coins
        coins.append(coin)

        plot_times, plot_prices = fit_times[0], fit_prices[0]
        coin.zero_time = zero_time
        coin.plots = {"actual": (plot_times, plot_prices, '-', 'o')}
        for days, fit in zip(fit_days, fits):
            times = plot_times[-days*6:]
            fit_times = np.linspace(times[0], predict_time, len(times) * 2)
            coin.plots[f"{days} day fit"] = (fit_times, [np.polyval(fit, time) for time in fit_times], '--', None)

    return coins


def get_best_coins(coins):
    print('Looking for best coins...')
    for coin in coins:
        price = tickers[coin.symbol]['last']
        coin.gain_lt = (coin.expected_lt - price) / price

        ohlcv = binance.fetch_ohlcv(coin.symbol, f'5m', limit=12)
        prices = [candle[3] for candle in ohlcv]
        times  = [candle[0] / milli_seconds_in_hour - coin.zero_time for candle in ohlcv]
        fit = np.polyfit(times, prices, 2)
        expected_st = np.polyval(fit, times[-1]+1)
        coin.gain_st = (expected_st - price) / price
        # Cap out when spikes occur.  Its probably too late to get the gains...
        # TODO need to think about this...
        coin.gain_st = max(min(coin.gain_st, .01), -.01)

        coin.gain = (coin.gain_lt + coin.gain_st) / 2
        coin.plots['actual st'] = (times, prices, '-', None)

    coins.sort(key=lambda coin: coin.gain, reverse=True)
    print('\n'.join(f"{coin.name}: {coin.gain} lt={coin.gain_lt} st={coin.gain_st}" for coin in coins[:4]))
    return coins


def get_holding_coin():
    def to_btc(coin, amount):
        if coin == 'BTC':
            return amount
        if coin == 'USDT':
            return amount / tickers["BTC/USDT"]['last']
        return amount * tickers[f"{coin}/BTC"]['last']

    def to_usdt(coin, amount):
        return to_btc(coin, amount) * tickers["BTC/USDT"]['last']

    balance = binance.fetch_balance()
    for coin, amount in balance['used'].items():
        if amount:
            print(f"WARNING {amount} {coin} is used")
    balance = [(coin, amount) for coin, amount in balance['free'].items() if amount]
    balance = [(coin, amount, to_usdt(coin, amount), to_btc(coin, amount)) for coin, amount in balance]
    hodl = max(balance, key=lambda item: item[2])
    return collections.namedtuple("Holding", "coin amount amount_usdt amount_btc")(*hodl)


trades = []

def buy_coin(from_coin, coin, try_factors, factor_wait_minutes=10):
    assert from_coin != coin, coin
    print(f'Transferring {from_coin} to {coin}...')

    for factor in try_factors:
        # .999 for .1 % binance fee - not sure if needed?
        holding_amount = binance.fetch_balance()[from_coin]['free'] * .999
        if f"{from_coin}/{coin}" in tickers:
            side   = 'sell'
            symbol = f"{from_coin}/{coin}"
            price = tickers[symbol]['last'] * (1-factor)
            price = max(price, binance.fetch_ticker(symbol)['last'])
            amount = holding_amount
        else:
            side   = 'buy'
            symbol = f"{coin}/{from_coin}"
            price = tickers[symbol]['last'] * (1+factor)
            price = min(price, binance.fetch_ticker(symbol)['last'])
            amount = holding_amount / price

        print(f"{side} {amount} {symbol} at {price}")
        order = binance.create_order(symbol, 'limit', side, amount, price)
        print(order['info'])

        id = order['id']
        for i in range(3):
            time.sleep(60*factor_wait_minutes/3)
            order = binance.fetch_order(id, symbol=symbol)
            print(f"{order['filled']} / {order['amount']} filled at factor={round(factor, 3)}")
            if order['status'] == 'closed':
                break
        else:
            print(f"Cancelling order {id} {symbol}")
            binance.cancel_order(id, symbol=symbol)

        if order['status'] == 'closed':
            trades.append(order['info'])
            break

    else:
        raise TimeoutError(f"Buy of {coin} didn't get filled")

    return order


def email_myself_plots(subject, coins, log):
    msg = EmailMessage()

    holding = get_holding_coin()
    balance = f" ${round(holding.amount_usdt)} ₿{round(holding.amount_btc, 5)}"

    msg['Subject'] = subject+balance
    results = '\n'.join(f"{coin.name} {coin.gain}" for coin in coins)
    msg.set_content(results)

    history = '<br>'.join(f"{t['side']} {t['symbol']} at {t['price']}" for t in reversed(trades))

    imgs = ""
    bufs = []
    for coin in coins:
        plt.title(f"{coin.name}  {round(coin.gain * 100, 2)}%")
        plt.xlabel("hours")
        plt.xticks(range(-100 * 24, 10 * 24, 24))
        for name, (x, y, linestyle, marker) in coin.plots.items():
            plt.plot(x, y, linestyle=linestyle, marker=marker, label=name)

        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        # plt.show()
        plt.clf()

        cid = make_msgid()
        bufs.append((cid, buf))
        imgs += f"<img src='cid:{cid[1:-1]}'>"

    msg.add_alternative(f"<html><body>{history}<hr>{imgs}<hr><pre>{log}</pre></body></html>", subtype='html')
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

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for file in self.files:
            file.write(data)

while True:
    start_time = time.time()
    try:
        with io.StringIO() as log, contextlib.redirect_stdout(Tee(log, sys.stdout)):
            tickers = binance.fetch_tickers()
            holding = get_holding_coin()
            from_coin = holding.coin
            result = None
            coins = get_coin_forecasts()
            for i in range(24):
                tickers = binance.fetch_tickers()
                coins = get_best_coins(coins)
                best  = coins[0]
                hodl  = next(c for c in coins if c.name == holding.coin)

                if best != hodl:
                    try:
                        result = f"{from_coin} -> {best.name}"
                        better = best.gain - hodl.gain
                        try_factors = np.linspace(-.005, min(.005, better/8), 6)
                        direct_buy = f"{hodl.name}/{best.name}" in tickers or f"{best.name}/{hodl.name}" in tickers
                        buy_coin(hodl.name, best.name if direct_buy else 'BTC', try_factors=try_factors)
                        if not direct_buy:
                            holding = get_holding_coin()
                            continue
                    except TimeoutError:
                        result += '...timed out'
                        continue
                    except:
                        result += '...errored'
                        import traceback
                        print(traceback.format_exc()) # print_exc goes to stderr not stdout
                    break

                time.sleep(10*60)

            result = result or f'HODL {hodl.name}'
            print(result)

            # Show relevant plots
            plot_coins = [coin for coin in coins if coin.name in [from_coin, 'BTC', best.name]]
            email_myself_plots(result, plot_coins, log.getvalue())

    except:
        import traceback
        error = traceback.format_exc()
        print(error, file=sys.stderr)
        msg = EmailMessage()
        msg['Subject'] = 'ERROR'
        msg.set_content(error)
        email_myself(msg)

    # Not really needed but just in case...
    loop_hours = (time.time() - start_time) / 3600
    if loop_hours < 1:
        time.sleep(3600*(1 - loop_hours))

    print('-'*30 + '\n')
