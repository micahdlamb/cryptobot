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
clamp = lambda value, frm, to: max(frm, min(to, value))
mix   = lambda frm, to, factor: frm + (to - frm) * factor
unmix = lambda value, frm, to: (value - frm) / (to - frm)
percentage = lambda value: f"{round(value*100, 2)}%"


def get_symbols():
    return ['BTC/USDT'] + [symbol for symbol, market in binance.markets.items()
                           if market['active'] and symbol.endswith('/BTC')]


class Candles(list):
    def __init__(self, symbol, timeFrame, limit):
        super().__init__(binance.fetch_ohlcv(symbol, timeFrame, limit=limit))

    @property
    def prices(self):
        prices = [np.average(candle[2:4]) for candle in self]
        to_center = (self[-1][0] - self[-2][0]) / 2
        times = [(candle[0] + to_center) / milli_seconds_in_hour for candle in self]
        return times, prices

    def polyfit(self, deg):
        return np.polyfit(*self.prices, deg)

    @property
    def avg_price(self):
        return self.polyfit(0)[0]

    @property
    def rate(self):
        return self.polyfit(1)[0]

    @property
    def acceleration(self):
        return self.polyfit(2)[0]

    @property
    def delta(self):
        return self.end_price * 2 - self.max - self.min

    @property
    def start_price(self):
        return self[0][1]

    @property
    def end_price(self):
        return self[-1][-2]

    @property
    def end_time(self):
        return self[-1][0] / milli_seconds_in_hour

    @property
    def min(self):
        return min(candle[3] for candle in self)

    @property
    def max(self):
        return max(candle[2] for candle in self)

    def __getitem__(self, item):
        if isinstance(item, slice):
            items = Candles.__new__(Candles)
            list.__init__(items, super().__getitem__(item))
            return items
        return super().__getitem__(item)


def get_coins():
    print('Getting coins...')
    class Coin(collections.namedtuple("Coin", "name symbol")): pass
    coins = []
    for symbol in get_symbols():
        candles = Candles(symbol, '1h', limit=12)
        if len(candles) < 8:
            print(f"Skipping {symbol} for missing data. len(ohlcv)={len(candles)}")
            continue

        name = symbol.split('/')[0]
        coin = Coin(name, symbol) # coin.gain set in get_best_coins
        coins.append(coin)

        times, prices = candles.prices
        coin.zero_time = times[-1]
        coin.plots = {"actual": (times, prices, dict(linestyle='-', marker='o'))}
        coin.trend = candles[-3:].rate / candles.end_price

    return coins


def get_best_coins(coins):
    print('Looking for best coins...')
    tickers = binance.fetch_tickers()
    for coin in coins:
        coin.price = tickers[coin.symbol]['last']
        candles = Candles(coin.symbol, '3m', limit=20)
        tick_size = 10 ** -binance.markets[coin.symbol]['precision']['price']
        coin.gain = (candles.acceleration - tick_size) / coin.price

        coin.plots["recent"] = *candles.prices, dict(linestyle='-')
        coin.dy_dx = candles[-3:].rate / coin.price

    coins.sort(key=lambda coin: coin.gain, reverse=True)

    best = coins[:4]
    vals = lambda coin: f"{coin.name}: {percentage(coin.gain)}/h² y'={percentage(coin.dy_dx)}/h"
    print('\n'.join(vals(coin) for coin in best))
    return coins


def hold_coin_while_gaining(coin):
    print(f"====== Holding {coin.name} ======")
    start_price = binance.fetch_ticker(coin.symbol)['last']
    start_time  = time.time()

    cell = lambda s: s.ljust(9)
    print(cell("y'"), cell("rate"), cell('gain'))

    while True:
        candles = Candles(coin.symbol, '3m', limit=20)
        deriv = np.polyder(candles.polyfit(2))
        now  = np.polyval(deriv, candles.end_time)     / start_price
        soon = np.polyval(deriv, candles.end_time+1/6) / start_price
        real = candles[-3:].rate / start_price
        gain = (binance.fetch_ticker(coin.symbol)['last'] - start_price) / start_price
        print(cell(f"{percentage(now)}/h"), cell(f"{percentage(real)}/h"), cell(percentage(gain)))

        if soon < 0:
            try:
                trade_coin(coin.name, 'BTC')
                break
            except TimeoutError as err:
                print(err)
        else:
            time.sleep(3*60)

    elapsed_time = time.time() - start_time
    candles = Candles(coin.symbol, '5m', limit=math.ceil(elapsed_time / 60 / 5))
    coin.plots['holding'] = *candles.prices, dict(linestyle='-')


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


def trade_coin(from_coin, to_coin, max_change=None):
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

    filled = 0
    for i in range(6):
        book = binance.fetch_order_book(symbol, limit=5)
        bid_price = book['bids'][0][0]
        ask_price = book['asks'][0][0]
        avg_price = np.average([bid_price, ask_price])

        if max_change:
            start_price, change = max_change
            bad_change   = -good_direction * (avg_price - start_price) / start_price
            if bad_change > change:
                raise TimeoutError(f"{side} of {symbol} aborted due to price change of {percentage(bad_change)}")

        holding_amount = binance.fetch_balance()[from_coin]['free']
        rate = Candles(symbol, '1m', limit=10).rate

        if side == 'buy':
            price  = round_price_down(symbol, min(ask_price*1.002, bid_price + rate/12))
            amount = binance.amount_to_lots(symbol, holding_amount / price)
        else:
            price  = round_price_up  (symbol, max(bid_price*.998, ask_price + rate/12))
            amount = holding_amount

        m1x    = unmix(price, bid_price, ask_price)
        spread = (ask_price - bid_price) / avg_price
        rate   = rate / avg_price
        print(f"{side} {amount} {symbol} at {price} mix={round(m1x, 1)} <->={percentage(spread)} y'={percentage(rate)}/h")

        order = create_order_and_wait(symbol, side, amount, price)
        if order['status'] == 'closed':
            return order

        filled += order['filled']
        if filled == 0:
            break

    raise TimeoutError(f"{side} of {symbol} didn't get filled")


def create_order_and_wait(symbol, side, amount, price, type='limit', timeout=5, poll=1):
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
    return order


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
        plt.title(coin.name+(f"  {percentage(coin.gain)}" if hasattr(coin, 'gain') else ""))
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

                coins = get_coins()
                holding = get_holding_coin()
                hodl = next(c for c in coins if c.name == holding.name)
                btc  = next(c for c in coins if c.name == 'BTC')
                tusd = next(c for c in coins if c.name == 'TUSD')

                trend = np.average([coin.trend for coin in coins])
                print(f'trend={percentage(trend)}/h')

                if hodl is btc and trend < -.02:
                    result = f"Hold BTC while market crashing..."
                    time.sleep(60*60)

                else:
                    result  = ""
                    if hodl is btc:
                        result = "BTC -> "
                        while True:
                            coins = get_best_coins(coins)
                            best = coins[0]

                            if best.gain + trend < .02:
                                print(f"{best.name} not good enough.  Hold BTC")
                                time.sleep(5*60)
                                continue

                            if best is btc:
                                print('Hold BTC')
                                time.sleep(30)
                                continue

                            try:
                                filled_order = trade_coin('BTC', best.name, max_change=(best.price, .02))
                                hodl = best
                                break
                            except TimeoutError as error:
                                print(error)

                    result += f"{hodl.name} -> BTC"
                    hold_coin_while_gaining(hodl)

                email_myself_plots(result, [btc, hodl], log.getvalue())

        except:
            import traceback
            error = traceback.format_exc()
            print(error, file=sys.stderr)
            msg = EmailMessage()
            msg['Subject'] = 'ERROR'
            msg.set_content(error)
            email_myself(msg)

        # Not really needed but just in case...
        loop_minutes = (time.time() - start_time) / 60
        if loop_minutes < 10:
            time.sleep(60*(10 - loop_minutes))

        print('-'*30 + '\n')
