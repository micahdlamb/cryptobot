"""
Ideas to try:
Simulate cost function to evaluate how good...
Tensor flow
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


def get_symbols():
    return ['BTC/USDT'] + [symbol for symbol, market in binance.markets.items()
                           if market['active'] and symbol.endswith('/BTC')]


def get_prices(symbol, timeFrame, limit):
    ohlcv = binance.fetch_ohlcv(symbol, timeFrame, limit=limit)
    prices = [np.average(candle[2:4]) for candle in ohlcv]
    to_center = (ohlcv[-1][0] - ohlcv[-2][0])/2
    times = [(candle[0] + to_center) / milli_seconds_in_hour for candle in ohlcv]
    return times, prices


def get_coin_forecasts():
    print('Forecasting coin prices...')
    class Coin(collections.namedtuple("Coin", "name symbol expected_lt")): pass
    coins = []
    for symbol in get_symbols():
        times, prices = get_prices(symbol, '1h', limit=16*24)
        if len(times) < 8*24:
            print(f"Skipping {symbol} for missing data. len(ohlcv)={len(times)}")
            continue

        fit_days = [4, 8, 16]
        fit_degs = [1, 1, 1]
        fit_times  = [times [-days*24:] for days in fit_days]
        fit_prices = [prices[-days*24:] for days in fit_days]
        fits = [np.polyfit(t, p, deg) for t,p,deg in zip(fit_times, fit_prices, fit_degs)]
        predict_time = times[-1] + 12
        expected = np.average([np.polyval(fit, predict_time) for fit in fits])

        plot_times, plot_prices = times[-16:], prices[-16:]
        plots = {"lt actual": (plot_times, plot_prices, dict(linestyle='-', marker='o'))}
        #for days, fit in zip(fit_days, fits):
        #    times = plot_times[-days*24:]
        #    fit_times  = np.linspace(times[0], times[-1], len(times) * 2)
        #    fit_prices = [np.polyval(fit, time) for time in fit_times]
        #    plots[f"{days} day fit"] = (fit_times, fit_prices, dict(linestyle='--'))


        name = symbol.split('/')[0]
        coin = Coin(name, symbol, expected) # coin.gain set in get_best_coins
        coin.plots = plots
        coin.zero_time = times[-1]
        coins.append(coin)

    return coins


def get_best_coins(coins):
    print('Looking for best coins...')

    def reduce_order_book(symbol, bound=.04, limit=500):
        book = binance.fetch_order_book(symbol, limit=limit)
        ask_price = book['asks'][0][0]
        ask_volume = sum(max(0, unmix(ask_price * (1+bound), ask_price, price)) * volume for price, volume in book['asks'])
        bid_price = book['bids'][0][0]
        bid_volume = sum(max(0, unmix(bid_price * (1-bound), bid_price, price)) * volume for price, volume in book['bids'])
        return (bid_volume / (bid_volume + ask_volume) - .5) * 2

    tickers = binance.fetch_tickers()
    for coin in coins:
        times, prices = get_prices(coin.symbol, '5m', limit=8*12)
        coin.plots["st actual"] = times, prices, dict(linestyle='-')
        price = tickers[coin.symbol]['last']
        coin.trend = np.polyfit(times[-36:], prices[-36:], 1)[0] / price
        coin.dy_dx = np.polyfit(times[ -6:], prices[ -6:], 1)[0] / price
        coin.delta = (prices[-1] - prices[-12]) / price
        tickSize = 10 ** -binance.markets[coin.symbol]['precision']['price']
        coin.lt = (coin.expected_lt - price - tickSize) / price
        coin.ob = reduce_order_book(coin.symbol) * .12
        coin.gain = coin.ob + clamp(coin.delta, -.03, .03)

    coins.sort(key=lambda coin: coin.gain, reverse=True)
    print('\n'.join(f"{coin.name}: {percentage(coin.gain)} ob={percentage(coin.ob)} delta={percentage(coin.delta)} "
                    f"lt={percentage(coin.lt)} dy/dx={percentage(coin.dy_dx)}/h" for coin in coins[:4]))
    return coins


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
    scale = 10**binance.markets[symbol]['precision']['price']
    return math.ceil(price * scale) / scale

def round_price_down(symbol, price):
    scale = 10**binance.markets[symbol]['precision']['price']
    return math.floor(price * scale) / scale


trade_log = []

def trade_coin(from_coin, to_coin, plots=None, max_change=.02, max_wait_minutes=90):
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

        ticker = binance.fetch_ticker(symbol)
        current_price = ticker['last']
        good_change   = good_direction * (current_price - start_price) / start_price
        if good_change < -max_change:
            raise TimeoutError(f"{side} of {symbol} aborted due to price change of {percentage(abs(good_change))}")

        holding_amount = binance.fetch_balance()[from_coin]['free']
        times, prices = get_prices(symbol, '1m', limit=5)
        price = current_price + np.polyfit(times, prices, 1)[0] * 2/60

        if side == 'buy':
            price  = round_price_down(symbol, min(price, current_price*1.001))
            amount = binance.amount_to_lots(symbol, holding_amount / price)
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


def create_order_and_wait(symbol, side, amount, price, type='limit', timeout=15, poll=3):
    order = binance.create_order(symbol, type, side, amount, price)
    del order['info']
    print(order)

    id = order['id']
    for i in range(int(timeout/poll)):
        time.sleep(poll*60)
        order = binance.fetch_order(id, symbol=symbol)
        print(f"{order['filled']} / {order['amount']} filled")
        if order['status'] == 'closed':
            from datetime import datetime
            order['fill_time'] = datetime.now().timestamp() / 3600
            trade_log.append(order)
            print('')
            return order

    print(f"Cancelling order {id} {symbol}")
    binance.cancel_order(id, symbol=symbol)


def email_myself_plots(subject, coins, log):
    msg = EmailMessage()

    balance = get_balance()
    total_gain = f"₿{percentage((balance.btc - start_balance.btc) / start_balance.btc)}"
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
                x = trade['fill_time'] - coin.zero_time
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

    msg.add_alternative(f"<html><body>{total_gain}<br>{trades}<hr>{imgs}<hr><pre>{log}</pre></body></html>", subtype='html')
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
    def flush(self):
        for file in self.files:
            file.flush()

if __name__ == "__main__":
    start_balance = get_balance()

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
                for i in range(16):
                    hodl  = next(c for c in coins if c.name == holding.name)
                    coins = get_best_coins(coins)
                    trend = np.average([coin.trend for coin in coins])
                    print(f'trend={percentage(trend)}/h')
                    if True or trend > -.003:
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
                                    gain_factor = 1 + max(best.gain, .012)
                                    price  = round_price_up(best.symbol, filled_order['price'] * gain_factor)
                                    amount = binance.fetch_balance()[coin]['free']
                                    create_order_and_wait(best.symbol, 'sell', amount, price, timeout=60*2, poll=10)
                                elapsed_time = time.time() - start_time

                                times, prices = get_prices(best.symbol, '5m', limit=math.ceil(elapsed_time/60/5))
                                plots['holding'] = times, prices, dict(linestyle='-')

                        except TimeoutError as error:
                            result += '...timed out'
                            print(error)
                            #continue uncomment eventually

                        except:
                            result += '...errored'
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
