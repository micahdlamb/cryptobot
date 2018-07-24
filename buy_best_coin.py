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

milli_seconds_in_hour   = 1000*60*60
milli_seconds_in_minute = 1000*60
clamp = lambda value,  frm, to: max(frm, min(to, value))
mix   = lambda factor, frm, to: frm + (to - frm) * factor
unmix = lambda value,  frm, to: (value - frm) / (to - frm)
rnd = lambda value: round(value, 4)


def get_coin_forecasts():
    print('Forecasting coin prices...')
    class Coin(collections.namedtuple("Coin", "name symbol expected_lt")): pass
    coins = []
    symbols = ['BTC/USDT'] + [symbol for symbol in tickers if symbol.endswith('/BTC')]
    for symbol in symbols:
        ohlcv = binance.fetch_ohlcv(symbol, '1h', limit=7*24)
        if len(ohlcv) < 7*24/2:
            print(f"Skipping {symbol} for missing data. len(ohlcv)={len(ohlcv)}")
            continue
        prices = [np.average(candle[2:-1]) for candle in ohlcv]
        times  = [candle[0] / milli_seconds_in_hour for candle in ohlcv]
        # Make time in past negative
        zero_time = times[-1]
        times = [time-zero_time for time in times]

        fit_days  = [1, 3, 7]
        fit_times  = [times [-days*24:] for days in fit_days]
        fit_prices = [prices[-days*24:] for days in fit_days]
        fits = [np.polyfit(t, p, 0) for t,p in zip(fit_times, fit_prices)]
        predict_time = times[-1] + 1
        expected = np.average([np.polyval(fit, predict_time) for fit in fits])

        # Make expected price more realistic...
        #current    = tickers[symbol]['last']
        #difference = expected - current
        #expected   = current + difference/2

        name = symbol.split('/')[0]
        coin = Coin(name, symbol, expected) # coin.gain set in get_best_coins
        coins.append(coin)

        plot_times, plot_prices = times[-16:], prices[-16:]
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

        ohlcv = binance.fetch_ohlcv(coin.symbol, '5m', limit=24)
        prices = [np.average(candle[2:-1]) for candle in ohlcv]
        times  = [candle[0] / milli_seconds_in_hour - coin.zero_time for candle in ohlcv]
        fit = np.polyfit(times, prices, 1)
        expected_st    = np.polyval(fit, times[-1]+1)
        price_on_curve = np.polyval(fit, times[-1])
        coin.gain_st = (expected_st - price_on_curve) / price
        # Cap out when spikes occur.  Its probably too late to get the gains...
        coin.gain_st = max(min(coin.gain_st, .03), -.03)

        coin.gain = (coin.gain_lt + coin.gain_st) / 2
        coin.gain_per_hour = np.polyfit(times[-6:], prices[-6:], 1)[0] / price
        coin.plots['actual st'] = (times, prices, '-', None)
        fit_prices = [np.polyval(fit, time) for time in times]
        coin.plots['fit st'] = (times, fit_prices, '-', None)

    coins.sort(key=lambda coin: coin.gain, reverse=True)
    print('\n'.join(f"{coin.name}: {rnd(coin.gain)} lt={rnd(coin.gain_lt)} st={rnd(coin.gain_st)} dy/dx={rnd(coin.gain_per_hour)}"
                    for coin in coins[:4]))
    return coins


def get_most_volatile_coins():
    print('Finding most spiky coins')
    class Coin(collections.namedtuple("Coin", "name fit amplitude")): pass
    coins = []
    symbols = [symbol for symbol in tickers if symbol.endswith('/BTC')]
    for symbol in symbols:
        ohlcv = binance.fetch_ohlcv(symbol, '5m', limit=6)
        prices = [np.average(candle[2:-1]) for candle in ohlcv]
        times = [candle[0]/milli_seconds_in_minute for candle in ohlcv]
        fit = np.polyfit(times, prices, 1)
        amplitude = (sum(abs(candle[3]-candle[2]) for candle in ohlcv) - abs(fit[0]*30)) / len(ohlcv)
        assert amplitude > -1e-5

        name = symbol.split('/')[0]
        current_price = ohlcv[-1][-2]
        coin = Coin(name, fit[0]/current_price, amplitude/current_price)
        coins.append(coin)

    coins.sort(key=lambda coin: coin.amplitude, reverse=True)
    print('\n'.join(f"{coin.name}: amplitude={rnd(coin.amplitude*100)}% fit={rnd(coin.fit*60)}%/h"
                    for coin in coins[:4]))
    return coins


def to_btc(coin, amount):
    if coin == 'BTC':
        return amount
    if coin == 'USDT':
        return amount / tickers["BTC/USDT"]['last']
    symbol = f"{coin}/BTC"
    if symbol in tickers:
        return amount * tickers[symbol]['last']
    print(f'Ignoring unknown coin in balance: {coin}')
    return 0


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

def buy_coin(from_coin, coin, max_change=.03, max_wait_minutes=60):
    assert from_coin != coin, coin
    print(f"Transferring {from_coin} to {coin}...")

    if f"{coin}/{from_coin}" in tickers:
        side = 'buy'
        symbol = f"{coin}/{from_coin}"
        good_direction = -1

    else:
        side = 'sell'
        symbol = f"{from_coin}/{coin}"
        good_direction = 1

    start_price = binance.fetch_ticker(symbol)['last']
    start_time  = time.time()

    while True:
        if (time.time() - start_time)/60 > max_wait_minutes:
            raise TimeoutError(f"{side} of {symbol} didn't get filled")

        # Ride any spikes
        ohlcv = binance.fetch_ohlcv(symbol, '1m', limit=5)
        prices = [np.average(candle[2:-1]) for candle in ohlcv]
        times = [candle[0]/milli_seconds_in_minute for candle in ohlcv]
        fit = np.polyfit(times, prices, 1)
        good_rate = fit[0] * good_direction
        if good_rate > 0:
            print('Wait while price moves in good direction...')
            time.sleep(5*60)
            continue

        ticker = binance.fetch_ticker(symbol)
        current_price = ticker['last']
        good_change   = good_direction * (current_price - start_price) / start_price
        if good_change < -max_change:
            raise TimeoutError(f"{side} of {symbol} aborted due to price change of {rnd(abs(good_change))}")

        ohlcv = binance.fetch_ohlcv(symbol, '5m', limit=6)
        prices = [np.average(candle[2:-1]) for candle in ohlcv]
        times = [candle[0]/milli_seconds_in_minute for candle in ohlcv]
        fit = np.polyfit(times, prices, 1)
        good_rate = fit[0] * good_direction
        amplitude = (sum(abs(candle[3]-candle[2]) for candle in ohlcv) - abs(fit[0]*30)) / len(ohlcv)
        assert amplitude > 0, amplitude
        print(f"good rate/hour {rnd(good_rate*60/current_price)} amplitude {rnd(amplitude/current_price)}")

        now = ticker['timestamp'] / milli_seconds_in_minute
        time_since_fit = now - times[-1]
        if not (0 <= time_since_fit <= 5):
            assert 0 <= time_since_fit <= 15, time_since_fit
            print(f"Warning: ticker time is {now - times[-1]} after ohlcv time")

        if good_rate > 0:
            price = np.polyval(fit, now + 15) + good_direction * amplitude / 2
        else:
            price = np.polyval(fit, now + 5) + good_direction * amplitude / 2

        # .999 for .1 % binance fee - not sure if needed?
        holding_amount = binance.fetch_balance()[from_coin]['free'] * .999
        if side == 'buy':
            price  = min(price, current_price*1.001)
            amount = holding_amount / price
        else:
            price  = max(price, current_price*.999)
            amount = holding_amount

        difference = (price - current_price) / current_price
        print(f"{side} {amount} {symbol} at {price} ({rnd(difference)})")
        try:
            order = binance.create_order(symbol, 'limit', side, amount, price)
        except Exception as error:
            # TODO not sure why this happens sometimes...
            time.sleep(5*60)
            continue
        print(order['info'])

        id = order['id']
        for i in range(3):
            time.sleep(5*60)
            order = binance.fetch_order(id, symbol=symbol)
            print(f"{order['filled']} / {order['amount']} filled")
            if order['status'] == 'closed':
                break
        else:
            print(f"Cancelling order {id} {symbol}")
            binance.cancel_order(id, symbol=symbol)

        if order['status'] == 'closed':
            trades.append(order['info'])
            print('')
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
            market_trend = np.average([v['percentage'] for k, v in tickers.items() if k.endswith('/BTC')])
            if market_trend > 0:
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
                            coin = best.name if hodl.name == 'BTC' else 'BTC'
                            buy_coin(hodl.name, coin)
                            if coin != best.name:
                                holding = get_holding_coin()
                                continue

                        except TimeoutError as error:
                            result += '...timed out'
                            print(error)
                            #continue uncomment eventually

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

            else:
                symbols = [symbol for symbol in tickers if symbol.endswith('/BTC')]
                coins = get_most_volatile_coins()
                spiky = coins[0].name
                print(f"Buy/sell {spiky}")
                starting_balance = get_balance()
                for i in range(4):
                    holding = get_holding_coin().coin
                    try:
                        buy_coin(holding, spiky if holding == 'BTC' else 'BTC', max_wait_minutes=30)
                    except TimeoutError as error:
                        print(f"Timeout: {error}")

                ending_balance = get_balance()
                gain = (ending_balance.btc - starting_balance.btc) / starting_balance.btc
                result = f"Buy/sell {spiky} ₿{rnd(gain*100)}%"
                print(result)
                msg = EmailMessage()
                msg['Subject'] = result
                msg.set_content(log.getvalue())
                email_myself(msg)

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
