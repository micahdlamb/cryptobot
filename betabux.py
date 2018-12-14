"""
Ideas to try:
Simulate cost function to evaluate how good...
"""

import os, sys, time, math, collections, io, contextlib, traceback, datetime, platform
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid

import numpy as np
import matplotlib
if platform.system() == 'Linux':
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
                tickers = binance.fetch_tickers()
                symbols = [symbol for symbol, market in binance.markets.items() if market['active']
                           and market['quote'] == 'BTC'
                           and 10 ** -market['precision']['price'] / tickers[symbol]['last'] < .001
                           and tickers[symbol]['quoteVolume'] > 50]

                market_delta = np.average([tickers[symbol]['percentage'] for symbol in symbols])
                print(f"24 hour alt coin change: {market_delta}% ({len(symbols)} coins)")

                class Coin(collections.namedtuple("Coin", "name symbol plots")): pass
                holding = get_holding_coin()

                if holding.name == 'BTC':
                    while True:
                        coins = [Coin(symbol.split('/')[0], symbol, {}) for symbol in symbols]
                        best = get_best_coin(coins)

                        if not best:
                            time.sleep(5*60)
                            continue

                        try:
                            order = trade_coin('BTC', best.name, max_change=(best.price, .015))
                            hodl = best
                            break
                        except TimeoutError as error:
                            print(error)

                    result = f"BTC -> {best.name} -> BTC"

                else:
                    hodl = Coin(holding.name, f"{holding.name}/BTC", {})
                    result = f"{hodl.name} -> BTC"

                with record_plot(hodl, 'hold'):
                    hold_till_crest(hodl)

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


timeFrame = '5m'
candles_per_hour = 12

def get_best_coin(coins):
    print('Looking for best coin...')
    good_coins = []
    tickers = binance.fetch_tickers()
    for coin in coins:
        ticker = tickers[coin.symbol]
        coin.price  = ticker['last']
        coin.vol    = math.log10(ticker['quoteVolume'])
        candles = Candles(coin.symbol, timeFrame, limit=18*candles_per_hour)
        #coin.mix = unmix(coin.price, candles.max, candles.min)
        hours = [6, 12, 18]
        wave_fits = [candles[-h * candles_per_hour:].wavefit(slice(2, 4)) for h in hours]
        for fit, h in zip(wave_fits, hours): fit.hours = h
        fit = max(wave_fits, key=lambda fit: fit.amp * fit.freq)
        #coin.mix = unmix(coin.price, fit.candles.max, fit.candles.min)
        coin.hours = fit.hours
        coin.amp   = fit.amp / coin.price
        coin.freq  = fit.freq
        coin.phase = math.cos(fit.phase-math.pi-.5)
        coin.error = fit.rmse / coin.price
        coin.wave_length = fit.hours / fit.freq

        coin.gain = coin.vol * coin.amp * coin.freq * coin.phase / (1+(coin.error*1e2)**2)
        if coin.gain < 0: continue
        good_coins.append(coin)

        coin.plots["actual"] = *candles.prices, dict(linestyle='-')
        times, prices = fit.prices
        coin.plots[f"zero {fit.hours}"] = [times[0], times[-1]], [fit.zero, fit.zero], dict(linestyle='--')
        coin.plots[f"wave {fit.hours}"] = times, prices, dict(linestyle='--')

    if not good_coins: return None
    good_coins.sort(key=lambda coin: coin.gain, reverse=True)
    col  = lambda s,c=6: str(s).ljust(c)
    rcol = lambda n,c=6: str(round(n, 2)).ljust(c)
    pcol = lambda n: percentage(n).ljust(6)
    print(col(''), col('gain'), col('vol', 4), col('mix',4), col('amp'), col('freq',4), col('phase'), col('error'))
    for coin in good_coins[:5]:
        print(col(coin.name), pcol(coin.gain), rcol(coin.vol, 4), rcol(coin.mix,4), pcol(coin.amp), col(coin.freq,4), rcol(coin.phase), pcol(coin.error))
        #show_plots(coin)

    best = good_coins[0]
    if best.gain < .01:
        print(f"{best.name} not good enough")
        return None

    return best


def hold_till_crest(coin):
    print(f"====== Holding {coin.name} ======")
    start_price = binance.fetch_ticker(coin.symbol)['last']
    cell = lambda s: str(s).ljust(7)
    print(cell("phase"), cell('mix'), cell('gain'))
    while True:
        price = binance.fetch_ticker(coin.symbol)['last']
        gain = (price - start_price) / start_price
        wave_length = getattr(coin, 'wave_length', 2)
        candles = Candles(coin.symbol, timeFrame, limit=int(wave_length*candles_per_hour))
        fit = candles.wavefit(slice(1, 3))
        phase = math.cos(fit.phase)
        crest_mix = clamp(unmix(price, fit.zero-fit.amp, fit.zero+fit.amp) * 2 - 1, -1, 1)
        print(cell(round(phase, 2)), cell(round(crest_mix, 2)), cell(percentage(gain)))

        times, prices = fit.prices
        coin.plots["hold zero"] = [times[0], times[-1]], [fit.zero, fit.zero], dict(linestyle='--')
        coin.plots["hold wave"] = times, prices, dict(linestyle='--')
        #show_plots(coin)

        if np.average([phase, crest_mix]) > .75:
            try:
                trade_coin(coin.name, 'BTC', avoid_partial_fill=False)
                break
            except TimeoutError as err:
                print(err)
        else:
            time.sleep(5*60)


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


def trade_coin(from_coin, to_coin, max_change=None, avoid_partial_fill=True):
    assert from_coin != to_coin, to_coin
    print(f"Transferring {from_coin} to {to_coin}...")

    if from_coin == 'BTC':
        side = 'buy'
        symbol = f"{to_coin}/{from_coin}"

    else:
        side = 'sell'
        symbol = f"{from_coin}/{to_coin}"

    filled = 0
    for i in range(8):
        book = binance.fetch_order_book(symbol, limit=5)
        bid_price = book['bids'][0][0]
        ask_price = book['asks'][0][0]
        avg_price = np.average([bid_price, ask_price])

        holding_amount = binance.fetch_balance()[from_coin]['free']
        rate = Candles(symbol, '1m', limit=5).rate

        if side == 'buy':
            price  = round_price_down(symbol, min(ask_price*1.001, bid_price + rate/20))
            amount = binance.amount_to_lots(symbol, holding_amount / price)
        else:
            price  = round_price_up  (symbol, max(bid_price*.999, ask_price + rate/20))
            amount = holding_amount

        if max_change:
            start_price, allowed_change = max_change
            change = (price - start_price) / start_price
            if abs(change) > allowed_change:
                raise TimeoutError(f"{side} of {symbol} aborted due to price change of {percentage(change)}")

        m1x    = unmix(price, bid_price, ask_price)
        spread = (ask_price - bid_price) / avg_price
        rate   = rate / avg_price
        print(f"{side} {amount} {symbol} at {price} mix={round(m1x, 3)} <->={percentage(spread)} y'={percentage(rate)}/h")

        order = create_order_and_wait(symbol, side, amount, price)
        if order['status'] == 'closed':
            return order

        if not avoid_partial_fill:
            break

        filled += order['filled']
        if filled == 0:
            break

    raise TimeoutError(f"{side} of {symbol} didn't get filled")


def create_order_and_wait(symbol, side, amount, price, timeout=3, poll=1):
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


def show_plots(coin):
    plt.figure()
    plt.title(coin.name)
    now_hours = datetime.datetime.now().timestamp() / 3600
    for name, (x, y, kwds) in coin.plots.items():
        x = [t - now_hours for t in x]
        plt.plot(x, y, label=name, **kwds)
    plt.show()


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
        super().__init__(binance.fetch_ohlcv(symbol, timeFrame, limit=max(limit, 2)))
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
        to_center = self.delta_time / 2
        times = [candle[0] / milli_seconds_in_hour + to_center for candle in self]
        #times.append(times[-1]+to_center)
        #prices.append(self[-1][-2])
        return times, prices

    @property
    def delta_time(self):
        return (self[1][0] - self[0][0]) / milli_seconds_in_hour

    def polyfit(self, deg, **kwds):
        return np.polyfit(*self.prices, deg, **kwds)

    class WaveFit(collections.namedtuple("Wave", "zero freq amp phase rmse")): pass

    def wavefit(self, freq_slice):
        times, prices = self.prices
        n = len(prices)
        fft = np.fft.fft(prices)
        zero = fft[0].real / n
        freq, wave = max(enumerate(fft[freq_slice] / n), key=lambda x: abs(x[1]))
        freq += freq_slice.start
        val = lambda x: np.real(wave * (np.cos(x*freq*2*np.pi/n) + 1j * np.sin(x*freq*2*np.pi/n)))*2
        values = zero + np.array([val(i) for i in range(n)])
        rmse = np.sqrt(((values - prices) ** 2).mean())
        wave_fit = self.WaveFit(zero, freq, abs(wave)*2, np.angle(wave) % (2*np.pi), rmse)
        wave_fit.candles = self
        wave_fit.prices  = (times, values)
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
        return self[-1][0] / milli_seconds_in_hour + self.delta_time

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
    del balance['BNB'] # Used for fees not trading
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


@contextlib.contextmanager
def record_plot(coin, plot_name, style=dict(linestyle='-')):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    candles = Candles(coin.symbol, '1m', limit=math.ceil(elapsed_time/60))
    coin.plots[plot_name] = *candles.prices, style


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
