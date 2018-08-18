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

np.fft.fft to get frequency and amplitude
"""

import os, sys, time, math, collections, io, contextlib, traceback
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
mix   = lambda frm, to, factor: frm + (to - frm) * factor
unmix = lambda frm, to, value: (value - frm) / (to - frm)
percentage = lambda value: f"{round(value*100, 2)}%"


def symbols():
    return ['BTC/USDT'] + [symbol for symbol, market in binance.markets.items()
                           if market['active'] and symbol.endswith('/BTC')]


def get_coin_forecasts():
    print('Forecasting coin prices...')
    class Coin(collections.namedtuple("Coin", "name symbol expected_lt")): pass
    coins = []
    for symbol in symbols():
        limit = 30*24
        ohlcv = binance.fetch_ohlcv(symbol, '1h', limit=limit)
        if len(ohlcv) < limit/2:
            print(f"Skipping {symbol} for missing data. len(ohlcv)={len(ohlcv)}")
            continue
        prices = [np.average(candle[2:4]) for candle in ohlcv]
        times  = [candle[0] / milli_seconds_in_hour for candle in ohlcv]

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
        ohlcv = binance.fetch_ohlcv(coin.symbol, '5m', limit=8*12)
        prices = [np.average(candle[2:4]) for candle in ohlcv]
        times  = [candle[0] / milli_seconds_in_hour for candle in ohlcv]

        fit_hours = [1, 2, 4, 8]
        fit_degs  = [1, 1, 1, 1]
        fit_times  = [times [-hour*12:] for hour in fit_hours]
        fit_prices = [prices[-hour*12:] for hour in fit_hours]
        fits = [np.polyfit(t, p, deg) for t,p,deg in zip(fit_times, fit_prices, fit_degs)]
        predict_time = times[-1] + 6
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
        coin.gain = mix(coin.gain_lt, coin.gain_st, .5)

        coin.trend = np.polyfit(times[-48:], prices[-48:], 1)[0] / price
        coin.dy_dx = np.polyfit(times[ -6:], prices[ -6:], 1)[0] / price

    coins.sort(key=lambda coin: coin.gain, reverse=True)
    print('\n'.join(f"{coin.name}: {percentage(coin.gain)} lt={percentage(coin.gain_lt)} st={percentage(coin.gain_st)} "
                    f"dy/dx={percentage(coin.dy_dx)}/h" for coin in coins[:4]))
    return coins

"""
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
"""

def get_balance():
    def to_btc(coin, amount):
        if coin == 'BTC':
            return amount
        if coin == 'USDT':
            return amount / tickers["BTC/USDT"]['last']
        return amount * tickers[f"{coin}/BTC"]['last']

    def to_usdt(coin, amount):
        return to_btc(coin, amount) * tickers["BTC/USDT"]['last']
    
    tickers = binance.fetch_tickers()
    tradeable = lambda coin: coin == 'BTC' or f"{coin}/BTC" in tickers
    balance = binance.fetch_balance()
    balance = {coin: info for coin, info in balance.items() if tradeable(coin)}
    
    Coin = collections.namedtuple("Coin", "name amount amount_free amount_used btc btc_free usdt")
    def coin(name, info):
        amount, free, used = info['total'], info['free'], info['used']
        return Coin(name, amount, free, used, to_btc(name, amount), to_btc(name, free), to_usdt(name, amount))

    coins = {name: coin(name, info) for name,info in balance.items()}  
    Balance = collections.namedtuple("Balance", "btc usdt coins")
    btc  = sum(coin.btc  for coin in coins.values())
    usdt = sum(coin.usdt for coin in coins.values())
    return Balance(btc, usdt, coins)


def get_holding_coin():
    balance = get_balance()
    for coin in balance.coins.values():
        if coin.amount_used:
            print(f"WARNING {coin.amount_used} {coin.name} is used")
    
    return max(balance.coins.values(), key=lambda coin: coin.btc_free)


def round_price_up(symbol, price):
    scale = pow(10, binance.markets[symbol]['precision']['price'])
    return math.ceil(price * scale) / scale

def round_price_down(symbol, price):
    scale = pow(10, binance.markets[symbol]['precision']['price'])
    return math.floor(price * scale) / scale


trade_log = []

def trade_coin(from_coin, to_coin, plots=None, max_change=.03, max_wait_minutes=90):
    assert from_coin != to_coin, to_coin
    print(f"Transferring {from_coin} to {to_coin}...")

    if from_coin == 'BTC':
        side = 'buy'
        symbol = f"{to_coin}/{from_coin}"
        good_direction = -1

    else:
        side = 'sell'
        symbol = f"{from_coin}/{to_coin}"
        good_direction = 1

    start_price = binance.fetch_ticker(symbol)['last']
    start_time  = time.time()

    while True:
        if (time.time() - start_time)/60 > max_wait_minutes:
            raise TimeoutError(f"{side} of {symbol} didn't get filled")

        # Ride any spikes
        ohlcv = binance.fetch_ohlcv(symbol, '1m', limit=30)
        prices = [np.average(candle[2:4]) for candle in ohlcv]
        times = [candle[0]/milli_seconds_in_minute for candle in ohlcv]
        fit = np.polyfit(times, prices, 1)
        good_rate = fit[0] * good_direction
        if good_rate > 0:
            print('Wait while price moves in good direction...')
            time.sleep(15*60)
            continue

        ticker = binance.fetch_ticker(symbol)
        current_price = ticker['last']
        good_change   = good_direction * (current_price - start_price) / start_price
        if good_change < -max_change:
            raise TimeoutError(f"{side} of {symbol} aborted due to price change of {percentage(abs(good_change))}")

        ohlcv = binance.fetch_ohlcv(symbol, '15m', limit=4)
        prices = [np.average(candle[2:4]) for candle in ohlcv]
        times = [candle[0]/milli_seconds_in_minute for candle in ohlcv]
        fit = np.polyfit(times, prices, 1)
        good_rate = fit[0] * good_direction
        amplitude = (sum(abs(candle[3]-candle[2]) for candle in ohlcv) - abs(fit[0]*15*4)) / len(ohlcv)
        assert amplitude > 0, amplitude
        print(f"good rate {percentage(good_rate*60/current_price)}/h amplitude {percentage(amplitude/current_price)}")

        if plots:
            times_in_hours = [t/60 for t in times]
            plots['trade actual'] = times_in_hours, prices, dict(linestyle='-')
            plots['trade fit']    = times_in_hours, [np.polyval(fit, t) for t in times], dict(linestyle='--')

        now = ticker['timestamp'] / milli_seconds_in_minute
        time_since_fit = now - times[-1]
        if not (0 <= time_since_fit <= 15):
            assert 0 <= time_since_fit <= 30, time_since_fit
            print(f"Warning: ticker time is {now - times[-1]} minutes after ohlcv time")

        if good_rate > 0:
            price = np.polyval(fit, now + 20) + good_direction * amplitude / 2
        else:
            price = np.polyval(fit, now + 10) + good_direction * amplitude / 2

        holding_amount = binance.fetch_balance()[from_coin]['free']
        if side == 'buy':
            price  = round_price_down(symbol, min(price, current_price*1.001))
            amount = binance.amount_to_lots(holding_amount / price)
        else:
            price  = round_price_up(symbol, max(price, current_price*.999))
            amount = holding_amount

        difference = (price - current_price) / current_price
        print(f"{side} {amount} {symbol} at {price} ({percentage(difference)})")

        try:
            filled_order = create_order_and_wait(symbol, side, amount, price)
            if filled_order:
                return filled_order

        except Exception:
            # Binance likes to error randomly when creating orders...
            print(traceback.format_exc())
            time.sleep(5*60)


def create_order_and_wait(symbol, side, amount, price, type='limit', timeout=30, poll=5):
    order = binance.create_order(symbol, type, side, amount, price)
    del order['info']
    print(order)

    id = order['id']
    for i in range(int(timeout/poll)):
        time.sleep(poll*60)
        order = binance.fetch_order(id, symbol=symbol)
        print(f"{order['filled']} / {order['amount']} filled")
        if order['status'] == 'closed':
            trade_log.append(order)
            print('')
            return order

    print(f"Cancelling order {id} {symbol}")
    binance.cancel_order(id, symbol=symbol)


def email_myself_plots(subject, coins, log):
    msg = EmailMessage()

    balance = get_balance()
    balance = f"₿{round(balance.btc, 5)} ${round(balance.usdt)}"

    msg['Subject'] = subject+balance
    trades = '<br>'.join(f"{t['side']} {t['symbol']} at {t['price']}" for t in reversed(trade_log))
    msg.set_content(trades)

    imgs = ""
    bufs = []
    for coin in coins:
        plt.title(f"{coin.name}  {percentage(coin.gain)}")
        plt.xlabel("hours")
        plt.xticks(range(-100 * 4, 10 * 4, 4))
        for name, (x, y, kwds) in coin.plots.items():
            x = [t-coin.zero_time for t in x]
            plt.plot(x, y, label=name, **kwds)

        for trade in trade_log:
            if trade['symbol'] == coin.symbol:
                x = trade['timestamp'] / milli_seconds_in_hour - coin.zero_time
                y = trade['price']
                plt.text(x, y, trade['side'][0])

        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        # plt.show()
        plt.clf()

        cid = make_msgid()
        bufs.append((cid, buf))
        imgs += f"<img src='cid:{cid[1:-1]}'>"

    msg.add_alternative(f"<html><body>{trades}<hr>{imgs}<hr><pre>{log}</pre></body></html>", subtype='html')
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

if __name__ == "__main__":
    while True:
        start_time = time.time()
        try:
            with io.StringIO() as log, contextlib.redirect_stdout(Tee(log, sys.stdout)):
                tickers = binance.fetch_tickers()
                market_delta = np.average([v['percentage'] for k, v in tickers.items() if k.endswith('/BTC')])
                print(f"24 hour alt coin change: {market_delta}%")
                holding = get_holding_coin()
                from_coin = holding.name
                result = None
                coins = get_coin_forecasts()
                for i in range(8):
                    hodl  = next(c for c in coins if c.name == holding.name)
                    coins = get_best_coins(coins)
                    trend = np.average([coin.trend for coin in coins])
                    print(f'trend={percentage(trend)}/h')
                    if trend > -.005:
                        best = coins[0]
                    else:
                        btc  = next(c for c in coins if c.name == 'BTC')
                        tusd = next(c for c in coins if c.name == 'TUSD')
                        best = btc if btc.gain > tusd.gain else tusd

                    if best != hodl:
                        try:
                            result = f"{from_coin} -> {best.name}"
                            coin  = best.name  if hodl.name == 'BTC' else 'BTC'
                            plots = best.plots if hodl.name == 'BTC' else hodl.plots
                            filled_order = trade_coin(hodl.name, coin, plots=plots)
                            if coin != best.name:
                                holding = get_holding_coin()
                                continue

                            if coin == best.name:
                                start_time = time.time()
                                if coin == "BTC":
                                    time.sleep(60*2)
                                else:
                                    price  = round_price_up(filled_order['price'] * (1 + max(best.gain, .02)))
                                    amount = binance.fetch_balance()[coin]['free'] * .999
                                    create_order_and_wait(best.symbol, 'sell', amount, price, timeout=60*4, poll=10)
                                elapsed_time = time.time() - start_time

                                ohlcv = binance.fetch_ohlcv(best.symbol, '5m', limit=int(elapsed_time/60/5))
                                prices = [np.average(candle[2:4]) for candle in ohlcv]
                                times = [candle[0] / milli_seconds_in_hour for candle in ohlcv]
                                plots['holding'] = times, prices, dict(linestyle='-')

                        except TimeoutError as error:
                            result += '...timed out'
                            print(error)
                            #continue uncomment eventually

                        except:
                            result += '...errored'
                            print(traceback.format_exc()) # print_exc goes to stderr not stdout

                        break

                    time.sleep(30*60)

                result = result or f'HODL {hodl.name}'
                print(result)

                # Show relevant plots
                plot_coins = [coin for coin in coins if coin.name in [from_coin, 'BTC', best.name]]
                email_myself_plots(result, plot_coins, log.getvalue())


                """
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
                """

                """
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
                """

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
