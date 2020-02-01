import time, traceback

import ccxt
binance = ccxt.binance()

symbols = "BTC/USDT ETH/USDT ETC/USDT XRP/USDT BCH/USDT LTC/USDT EOS/USDT".split()

def main():
    while True:
        try:
            for symbol in symbols:
                limit = {
                    'BTC/USDT': 5000,
                    'ETH/USDT': 1000,
                }.get(symbol, 500)
                price, balance, limit_volume = reduce_order_book(symbol, limit=limit)
                # ticker = binance.fetch_ticker(symbol)
                # trade_volume = ticker['quoteVolume']
                with open(f"data/{symbol.replace('/', '_')}.csv", 'a') as f:
                    print(int(time.time()), price, balance, round(limit_volume), sep=',', file=f)

            time.sleep(60*5)
        except Exception as error:
            print(traceback.format_exc())


def reduce_order_book(symbol, bound=.04, pow=4, limit=500):
    """Reduces order book to value between -1 -> 1.
       -1 means all orders are asks, 1 means all orders are bids.  Presumably -1 is bad and 1 is good.
       Volumes are weighted less the farther they are from the current price.
    """
    book = binance.fetch_order_book(symbol, limit=limit)
    ask_range = (book['asks'][-1][0] - book['asks'][0][0]) / book['asks'][0][0]
    if ask_range < bound:
        print(f"WARNING {symbol} ask range {round(ask_range, 3)} < {bound}. Increase limit")
    ask_price = book['asks'][0][0]
    ask_bound = ask_price * (1+bound)
    ask_volume = sum(max(0, unmix(price, ask_bound, ask_price))**pow * volume * price for price, volume in book['asks'])
    bid_price = book['bids'][0][0]
    bid_bound = bid_price * (1-bound)
    bid_volume = sum(max(0, unmix(price, bid_bound, bid_price))**pow * volume * price for price, volume in book['bids'])
    price = (ask_price + bid_price) / 2
    volume = bid_volume + ask_volume
    return price, (bid_volume / volume) * 2 - 1, volume


milli_seconds_in_hour   = 1000*60*60
milli_seconds_in_minute = 1000*60
clamp = lambda value, frm, to: max(frm, min(to, value))
mix   = lambda frm, to, factor: frm + (to - frm) * factor
unmix = lambda value, frm, to: (value - frm) / (to - frm)
percentage = lambda value: f"{round(value*100, 2)}%"


if __name__ == '__main__':
    main()
