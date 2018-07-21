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
clamp = lambda value,  frm, to: max(frm, min(to, value))
mix   = lambda factor, frm, to: frm + (to - frm) * factor
unmix = lambda value,  frm, to: (value - frm) / (to - frm)

def get_coin_forecasts():
    print('Forecasting coin prices...')
    class Coin(collections.namedtuple("Coin", "name symbol expected_lt")): pass
    coins = []
    symbols = ['BTC/USDT'] + [symbol for symbol in tickers if symbol.endswith('/BTC')]
    for symbol in symbols:
        ohlcv = binance.fetch_ohlcv(symbol, f'1h', limit=14*24)
        if len(ohlcv) < 14*24/2:
            print(f"Skipping {symbol} for missing data. len(ohlcv)={len(ohlcv)}")
            continue
        prices = [np.average(candle[1:-1]) for candle in ohlcv]
        times  = [candle[0] / milli_seconds_in_hour for candle in ohlcv]
        # Make time in past negative
        zero_time = times[-1]
        times = [time-zero_time for time in times]

        fit_days  = [3, 7 ,14]
        fit_times  = [times [-days*24:] for days in fit_days]
        fit_prices = [prices[-days*24:] for days in fit_days]
        fits = [np.polyfit(t, p, 3) for t,p in zip(fit_times, fit_prices)]
        predict_time = times[-1] + 4
        expected = np.average([np.polyval(fit, predict_time) for fit in fits])

        # Make expected price more realistic...
        #current    = tickers[symbol]['last']
        #difference = expected - current
        #expected   = current + difference/2

        name = symbol.split('/')[0]
        coin = Coin(name, symbol, expected) # coin.gain set in get_best_coins
        coins.append(coin)

        plot_times, plot_prices = times[-24:], prices[-24:]
        coin.zero_time = zero_time
        coin.plots = {"actual": (plot_times, plot_prices, '-', 'o')}
        for days, fit in zip(fit_days, fits):
            times = plot_times[-days*24:]
            fit_times = np.linspace(times[0], predict_time, len(times) * 2)
            coin.plots[f"{days} day fit"] = (fit_times, [np.polyval(fit, time) for time in fit_times], '--', None)

    return coins


def get_best_coins(coins):
    print('Looking for best coins...')
    for coin in coins:
        price = tickers[coin.symbol]['last']
        coin.gain_lt = (coin.expected_lt - price) / price

        ohlcv = binance.fetch_ohlcv(coin.symbol, f'5m', limit=36)
        prices = [np.average(candle[1:-1]) for candle in ohlcv]
        times  = [candle[0] / milli_seconds_in_hour - coin.zero_time for candle in ohlcv]
        fit = np.polyfit(times, prices, 1) # TODO 1 or 2 here?
        expected_st    = np.polyval(fit, times[-1]+1)
        price_on_curve = np.polyval(fit, times[-1])
        coin.gain_st = (expected_st - price_on_curve) / price
        # Cap out when spikes occur.  Its probably too late to get the gains...
        # TODO need to think about this...
        coin.gain_st = max(min(coin.gain_st, .02), -.02)

        coin.gain = (coin.gain_lt + coin.gain_st) / 2
        coin.gain_per_hour = np.polyfit(times[-6:], prices[-6:], 1)[0] / price
        coin.plots['actual st'] = (times, prices, '-', None)
        fit_prices = [np.polyval(fit, time) for time in times]
        coin.plots['fit st'] = (times, fit_prices, '-', None)

    coins.sort(key=lambda coin: coin.gain, reverse=True)
    rnd = lambda value: round(value, 4)
    print('\n'.join(f"{coin.name}: {rnd(coin.gain)} lt={rnd(coin.gain_lt)} st={rnd(coin.gain_st)} dy/dx={rnd(coin.gain_per_hour)}"
                    for coin in coins[:4]))
    return coins


def to_btc(coin, amount):
    if coin == 'BTC':
        return amount
    if coin == 'USDT':
        return amount / tickers["BTC/USDT"]['last']
    return amount * tickers[f"{coin}/BTC"]['last']


def to_usdt(coin, amount):
    return to_btc(coin, amount) * tickers["BTC/USDT"]['last']


def get_balance():
    balance = binance.fetch_balance()
    btc  = sum(to_btc (coin, amount) for coin, amount in balance['total'].items() if amount)
    usdt = sum(to_usdt(coin, amount) for coin, amount in balance['total'].items() if amount)
    return collections.namedtuple("Balance", "btc usdt")(btc, usdt)


def get_holding_coin():
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
    print(f"Transferring {from_coin} to {coin}...")
    print(f"try_factors={try_factors}")

    for factor in try_factors:
        # .999 for .1 % binance fee - not sure if needed?
        holding_amount = binance.fetch_balance()[from_coin]['free'] * .999
        if f"{from_coin}/{coin}" in tickers:
            side   = 'sell'
            symbol = f"{from_coin}/{coin}"
            price = tickers[symbol]['last'] * (1-factor)
            price = max(price, binance.fetch_ticker(symbol)['last']*.999)
            amount = holding_amount
        else:
            side   = 'buy'
            symbol = f"{coin}/{from_coin}"
            price = tickers[symbol]['last'] * (1+factor)
            price = min(price, binance.fetch_ticker(symbol)['last']*1.001)
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
            print('')
            break

    else:
        raise TimeoutError(f"Buy of {coin} didn't get filled")

    return order


def email_myself_plots(subject, coins, log):
    msg = EmailMessage()

    balance = get_balance()
    balance = f"₿{round(balance.btc, 5)} ${round(balance.usdt)}"

    msg['Subject'] = subject+balance
    results = '\n'.join(f"{coin.name} {coin.gain}" for coin in coins)
    msg.set_content(results)

    history = '<br>'.join(f"{t['side']} {t['symbol']} at {t['price']}" for t in reversed(trades))

    imgs = ""
    bufs = []
    for coin in coins:
        plt.title(f"{coin.name}  {round(coin.gain * 100, 2)}%")
        plt.xlabel("hours")
        plt.xticks(range(-100 * 4, 10 * 4, 4))
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
            for i in range(16):
                tickers = binance.fetch_tickers()
                coins = get_best_coins(coins)
                best  = coins[0]
                hodl  = next(c for c in coins if c.name == holding.coin)

                if best != hodl:
                    try:
                        result = f"{from_coin} -> {best.name}"
                        #direct_buy = f"{hodl.name}/{best.name}" in tickers or f"{best.name}/{hodl.name}" in tickers
                        if hodl.name == 'BTC':
                            buy       = best.name
                            good_rate = -best.gain_per_hour
                        else:
                            buy       = 'BTC'
                            good_rate = hodl.gain_per_hour

                        good_rate  = clamp(good_rate, -.03, .03)
                        factor     = unmix(good_rate, -.03, .03)
                        num_tries  = mix(factor, 3, 6)
                        start      = mix(factor, -.003, -.03)
                        end        = mix(factor, .015,   .003)

                        try_factors = np.linspace(start, end, int(num_tries))
                        buy_coin(hodl.name, buy, try_factors=try_factors)

                        if buy != best.name:
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

                time.sleep(15*60)

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
