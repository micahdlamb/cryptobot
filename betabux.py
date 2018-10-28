"""
Ideas to try:
Simulate cost function to evaluate how good...
Tensor flow
"""

import os, sys, time, math, collections, io, contextlib, traceback, datetime
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


def main():
    start_balance = get_balance()

    while True:
        start_time = time.time()
        with io.StringIO() as log, contextlib.redirect_stdout(Tee(log, sys.stdout)):
            try:
                holding = get_holding_coin()
                tickers = binance.fetch_tickers()
                market_delta = np.average([v['percentage'] for k, v in tickers.items() if k.endswith('/BTC')])
                print(f"24 hour alt coin change: {market_delta}%")

                #ignore = {'HOT', 'DENT', 'NPXS', 'KEY', 'SC', 'CDT', 'QTUM', 'TNB', 'VET', 'MFT', 'XVG'}
                tick_size = lambda symbol: 10 ** -binance.markets[symbol]['precision']['price'] / tickers[symbol]['last']
                keep = {'BTC/USDT', 'TUSD/BTC' f'{holding.name}/BTC'}
                symbols = [symbol for symbol, market in binance.markets.items() if symbol in keep or (
                           market['active'] and market['quote'] == 'BTC'
                           and tick_size(symbol) < .001
                           and tickers[symbol]['quoteVolume'] > 200)]

                class Coin(collections.namedtuple("Coin", "name symbol plots")): pass
                coins = [Coin(symbol.split('/')[0], symbol, {}) for symbol in symbols]
                hodl = next(c for c in coins if c.name == holding.name)
                btc  = next(c for c in coins if c.name == 'BTC')

                if hodl is btc:
                    while True:
                        best = get_best_coin(coins)

                        if not best:
                            time.sleep(10*60)
                            continue

                        if best is btc:
                            print('Hold BTC')
                            time.sleep(30)
                            continue

                        try:
                            order = trade_coin('BTC', best.name, max_change=(best.price, .015))
                            hodl = best
                            break
                        except TimeoutError as error:
                            print(error)

                    result = f"BTC -> {best.name} -> BTC"
                    timeout, poll = best.wave_length * 60 / 2, 5
                    order = create_order_and_wait(best.symbol, 'sell', order['amount'], best.peak, timeout, poll)
                    if order['status'] != 'closed':
                        hold_coin_while_gaining(best)

                else:
                    result = f"{hodl.name} -> BTC"
                    hold_coin_while_gaining(hodl)

                email_myself_plots(result, start_balance, [hodl], log.getvalue())

            except:
                print(traceback.format_exc())
                msg = EmailMessage()
                msg['Subject'] = 'ERROR'
                msg.set_content(log.getvalue())
                email_myself(msg)

        # Not really needed but just in case...
        loop_minutes = (time.time() - start_time) / 60
        if loop_minutes < 10:
            time.sleep(60*(10 - loop_minutes))

        print('-'*30 + '\n')


def get_best_coin(coins):
    print('Looking for best coin...')
    good_coins = []
    tickers = binance.fetch_tickers()
    for coin in coins:
        coin.price = tickers[coin.symbol]['last']
        candles = Candles(coin.symbol, '15m', limit=24*4)
        wave_fit = candles.wavefit()
        coin.amp  = wave_fit.amp / coin.price
        coin.freq = wave_fit.freq
        coin.wave_length = 24/wave_fit.freq
        coin.phase = unmix(abs(wave_fit.phase - np.pi), np.pi, 0)*2 -1
        coin.trend = wave_fit.trend / coin.price
        coin.gain = coin.amp * coin.freq**2 * coin.phase + clamp(coin.trend*coin.wave_length/2, -.01, .01)
        if coin.gain < 0: continue

        coin.peak = coin.price + wave_fit.amp*2
        coin.plots["actual"] = *candles.prices,  dict(linestyle='-')
        coin.plots["wave"]   = *wave_fit.prices, dict(linestyle='--')
        coin.plots["trend"]  = *wave_fit.trend_prices, dict(linestyle='--')
        good_coins.append(coin)

    good_coins.sort(key=lambda coin: coin.gain, reverse=True)
    col  = lambda s: s.ljust(6)
    rcol = lambda n: str(round(n, 2)).ljust(6)
    pcol = lambda n: percentage(n).ljust(6)
    print(col(''), col('gain'), col('amp'), col('freq'), col('phase'), col('trend'))
    for coin in good_coins[:10]:
        print(col(coin.name), pcol(coin.gain), pcol(coin.amp), rcol(coin.freq), rcol(coin.phase), pcol(coin.trend))

        #plt.figure()
        #plt.title(coin.name)
        #now_hours = datetime.datetime.now().timestamp() / 3600
        #for name, (x, y, kwds) in coin.plots.items():
        #    x = [t-now_hours for t in x]
        #    plt.plot(x, y, label=name, **kwds)
        #plt.show()

    best = good_coins[0]
    if best.gain < .03:
        print(f"{best.name} not good enough")
        return None

    return best


def hold_coin_while_gaining(coin):
    print(f"====== Holding {coin.name} ======")
    start_price = binance.fetch_ticker(coin.symbol)['last']
    start_time  = time.time()

    cell = lambda s: s.ljust(9)
    print(cell("y'"), cell("rate"), cell('gain'))

    while True:
        candles = Candles(coin.symbol, '3m', limit=20)
        deriv = np.polyder(candles.polyfit(2))
        now  = np.polyval(deriv, candles.end_time)      / start_price
        soon = np.polyval(deriv, candles.end_time+1/20) / start_price
        real = candles[-3:].rate / start_price
        gain = (binance.fetch_ticker(coin.symbol)['last'] - start_price) / start_price
        print(cell(f"{percentage(now)}/h"), cell(f"{percentage(real)}/h"), cell(percentage(gain)))

        if real < 0 and soon < 0:
            try:
                trade_coin(coin.name, 'BTC')
                break
            except TimeoutError as err:
                print(err)
        else:
            time.sleep(3*60)

    elapsed_time = time.time() - start_time
    candles = Candles(coin.symbol, '1m', limit=max(2, math.ceil(elapsed_time/60/1)))
    coin.plots['holding'] = *candles.prices, dict(linestyle='-')


def market_buy(symbol, fraction_of_btc=.95):
    price = binance.fetch_ticker(symbol)['last']
    free_btc = binance.fetch_balance()['BTC']['free']
    amount = binance.amount_to_lots(symbol, free_btc * fraction_of_btc / price)
    order = binance.create_market_buy_order(symbol, amount)
    _record_order(order)
    print(f"Bought {order['filled']} {symbol} at {order['price']}")
    return order


def market_sell(symbol, amount):
    order = binance.create_market_sell_order(symbol, amount)
    _record_order(order)
    print(f"Sold {order['filled']} {symbol} at {order['price']}")
    return order


trade_log = []


def _record_order(order):
    order['fill_time'] = order['timestamp'] / milli_seconds_in_hour
    btc = sum(float(fill['price']) * float(fill['qty']) for fill in order['info']['fills'])
    order['price'] = btc / order['filled']
    trade_log.append(order)


def trade_coin(from_coin, to_coin, max_change=None):
    assert from_coin != to_coin, to_coin
    print(f"Transferring {from_coin} to {to_coin}...")

    if from_coin == 'BTC':
        side = 'buy'
        symbol = f"{to_coin}/{from_coin}"

    else:
        side = 'sell'
        symbol = f"{from_coin}/{to_coin}"

    filled = 0
    for i in range(6):
        book = binance.fetch_order_book(symbol, limit=5)
        bid_price = book['bids'][0][0]
        ask_price = book['asks'][0][0]
        avg_price = np.average([bid_price, ask_price])

        holding_amount = binance.fetch_balance()[from_coin]['free']
        rate = Candles(symbol, '1m', limit=5).rate

        if side == 'buy':
            price  = round_price_down(symbol, min(ask_price*1.002, bid_price + rate/20))
            amount = binance.amount_to_lots(symbol, holding_amount / price)
        else:
            price  = round_price_up  (symbol, max(bid_price*.998, ask_price + rate/20))
            amount = holding_amount

        if max_change:
            start_price, allowed_change = max_change
            change = (price - start_price) / start_price
            if abs(change) > allowed_change:
                raise TimeoutError(f"{side} of {symbol} aborted due to price change of {percentage(change)}")

        m1x    = unmix(price, bid_price, ask_price)
        spread = (ask_price - bid_price) / avg_price
        rate   = rate / avg_price
        print(f"{side} {amount} {symbol} at {price} mix={round(m1x, 2)} <->={percentage(spread)} y'={percentage(rate)}/h")

        order = create_order_and_wait(symbol, side, amount, price)
        if order['status'] == 'closed':
            return order

        filled += order['filled']
        if filled == 0:
            break

    raise TimeoutError(f"{side} of {symbol} didn't get filled")


def create_order_and_wait(symbol, side, amount, price, timeout=5, poll=1):
    order = binance.create_order(symbol, 'limit', side, amount, price)
    del order['info']
    print(order)
    print(f"Wait for {timeout} minutes...")

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


def email_myself_plots(subject, start_balance, coins, log):
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
        now_hours = datetime.datetime.now().timestamp() / 3600
        for name, (x, y, kwds) in coin.plots.items():
            x = [t-now_hours for t in x]
            plt.plot(x, y, label=name, **kwds)

        for trade in trade_log:
            if trade['symbol'] == coin.symbol:
                x = trade['fill_time'] - now_hours
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


class Candles(list):
    cache = dict()

    def __init__(self, symbol, timeFrame, limit):
        super().__init__(binance.fetch_ohlcv(symbol, timeFrame, limit=limit))
        #key = symbol, timeFrame
        #ohlcv = self.cache.get(key)
        #if not ohlcv or limit > len(ohlcv):
        #    self.cache[key] = ohlcv = binance.fetch_ohlcv(symbol, timeFrame, limit=limit)
        #else:
        #    now = datetime.datetime.now().timestamp() * 1000
        #    dt = ohlcv[1][0] - ohlcv[0][0]
        #    keep_limit = max(len(ohlcv), limit)
        #    new_limit = min(keep_limit, math.floor((now - ohlcv[-1][0]) / dt) + 1)
        #    #print(new_limit, (now - ohlcv[-1][0]) / dt, (now - ohlcv[-1][0])/milli_seconds_in_minute)
        #    new_ohlcv = binance.fetch_ohlcv(symbol, timeFrame, limit=new_limit)
        #    ohlcv = ohlcv[:-1] + new_ohlcv
        #    self.cache[key] = ohlcv[-keep_limit:]
        #    ohlcv = ohlcv[-limit:]
        #    for i in range(len(ohlcv)-1):
        #        assert dt*.9 < ohlcv[i+1][0] - ohlcv[i][0] < dt*1.1

        #super().__init__(ohlcv)

    @property
    def prices(self):
        prices = [np.average(candle[2:4]) for candle in self]
        to_center = self.dt / 2
        times = [candle[0] / milli_seconds_in_hour + to_center for candle in self]
        #times.append(times[-1]+to_center)
        #prices.append(self[-1][-2])
        return times, prices

    @property
    def dt(self):
        return (self[1][0] - self[0][0]) / milli_seconds_in_hour

    def polyfit(self, deg, **kwds):
        return np.polyfit(*self.prices, deg, **kwds)

    class WaveFit(collections.namedtuple("Wave", "trend freq amp phase")): pass

    def wavefit(self):
        times, prices = self.prices
        trend_fit = np.polyfit(times, prices, deg=1)
        trend_prices = np.polyval(trend_fit, times)
        diffs = [price - trend_price for price, trend_price in zip(prices, trend_prices)]
        n = len(diffs)
        fft = np.fft.fft(diffs)[:int(n/2)] / n
        freq, wave = max(enumerate(fft), key=lambda x: abs(x[1]))
        val = lambda x: np.real(wave * (np.cos(x*freq*2*np.pi/n) + 1j * np.sin(x*freq*2*np.pi/n))) * 2
        wave_fit = self.WaveFit(trend_fit[0], freq, abs(wave), np.angle(wave) % (2*np.pi))
        wave_fit.trend_prices = times, trend_prices
        wave_fit.prices = times,  [trend_price + val(i) for i, trend_price in enumerate(trend_prices)]
        return wave_fit

    @property
    def avg_price(self):
        return self.polyfit(0)[0]

    @property
    def rate(self):
        return self.polyfit(1)[0]

    @property
    def acceleration(self):
        return self.polyfit(2)[0] * 2

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
    main()
