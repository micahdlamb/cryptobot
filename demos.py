import math, collections, datetime
import numpy as np
import matplotlib.pyplot as plt
import ccxt

binance = ccxt.binance({
    #'apiKey': os.environ['binance_apiKey'],
    #'secret': os.environ['binance_secret']
})


milli_seconds_in_hour   = 1000*60*60
milli_seconds_in_minute = 1000*60
clamp = lambda value, frm, to: max(frm, min(to, value))
mix   = lambda frm, to, factor: frm + (to - frm) * factor
unmix = lambda value, frm, to: (value - frm) / (to - frm)
percentage = lambda value: f"{round(value*100, 2)}%"


def get_symbols(min_volume=50, max_tick_size=.001):
    """Use small tick_size to ignore coins like HOT whose price moves in large increments"""
    tickers = binance.fetch_tickers()
    return [symbol for symbol, market in binance.markets.items() if market['active']
            and market['quote'] == 'BTC'
            and 10 ** -market['precision']['price'] / tickers[symbol]['last'] < max_tick_size
            and tickers[symbol]['quoteVolume'] > min_volume]


class Candles(list):
    """Pull prices for a coin.  Times are in hours"""
    def __init__(self, symbol, timeFrame, limit):
        super().__init__(binance.fetch_ohlcv(symbol, timeFrame, limit=max(limit, 2)))
        self.symbol = symbol
        self.dt = int(timeFrame[:-1]) * {'m': 1/60, 'h': 1, 'd': 24}[timeFrame[-1]]

    @property
    def prices(self):
        "Candles reduced to single price by using avg(min, max)"
        prices = [np.average(candle[2:4]) for candle in self]
        times = [candle[0] / milli_seconds_in_hour + self.dt/2 for candle in self]
        return times, prices

    @property
    def start_price(self):
        return self[0][1]

    @property
    def end_price(self):
        return self[-1][-2]

    @property
    def end_time(self):
        self[-1][0] / milli_seconds_in_hour + self.dt / 2

    def polyfit(self, deg, **kwds):
        return np.polyfit(*self.prices, deg, **kwds)

    @property
    def avg_price(self):
        return self.polyfit(0)[0]

    @property
    def velocity(self):
        return self.polyfit(1)[0]

    @property
    def acceleration(self):
        return self.polyfit(2)[0] * 2

    @property
    def min(self):
        return min(candle[3] for candle in self)

    @property
    def max(self):
        return max(candle[2] for candle in self)

    class WaveFit(collections.namedtuple("Wave", "zero freq amp phase rmse")): pass

    def wavefit(self, freq_slice):
        times, prices = self.prices
        n = len(prices)
        fft = np.fft.fft(prices)
        zero = fft[0].real / n
        freq, wave = max(enumerate(fft[freq_slice] / n), key=lambda x: abs(x[1]))
        freq += freq_slice.start
        # amplitudes * 2 to account for imaginary component
        val = lambda x: np.real(wave * (np.cos(x*freq*2*np.pi/n) + 1j * np.sin(x*freq*2*np.pi/n)))*2
        values = zero + np.array([val(i) for i in range(n)])
        rmse = np.sqrt(((values - prices)**2).mean())
        wavefit = self.WaveFit(zero, freq, abs(wave)*2, np.angle(wave) % (2*np.pi), rmse)
        wavefit.candles = self
        wavefit.prices  = (times, values)
        return wavefit

    def __getitem__(self, item):
        if isinstance(item, slice):
            items = Candles.__new__(Candles)
            list.__init__(items, super().__getitem__(item))
            return items
        return super().__getitem__(item)


def reduce_order_book(symbol, bound=.04, pow=2, limit=500):
    """Reduces order book to value between -1 -> 1.
       -1 means all orders are asks, 1 means all orders are bids.  Presumably -1 is bad and 1 is good.
       Volumes are weighted less the farther they are from the current price.
       bound=.04 means orders more than +-4% of the current price have 0 weight.
       pow=2 gives orders closer to current price a lot more weight
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
    volume = bid_volume + ask_volume
    return (bid_volume / volume) * 2 - 1


def show_plot(title, curves):
    plt.figure()
    plt.title(title)
    now_hours = datetime.datetime.now().timestamp() / 3600
    for name, (x, y, kwds) in curves.items():
        x = [t - now_hours for t in x]
        plt.plot(x, y, label=name, **kwds)
    plt.show()


# Demos ################################################################################################################

def find_coin_with_highest_velocity():
    candles = [Candles(symbol, '5m', 12) for symbol in get_symbols()]
    best = max(candles, key=lambda coin: coin.velocity / coin.start_price)
    times, prices = best.prices
    show_plot(f"{best.symbol} velocity={percentage(best.velocity / best.start_price)}/hour", {
        "prices": (times, prices, dict(linestyle='-')),
        "fit": (times, np.polyval(best.polyfit(1), times), dict(linestyle='--'))
    })


def find_coin_with_highest_acceleration():
    candles = [Candles(symbol, '5m', 3*12) for symbol in get_symbols()]
    best = max(candles, key=lambda coin: coin.acceleration / coin.start_price)
    times, prices = best.prices
    show_plot(f"{best.symbol} acceleration={percentage(best.acceleration / best.start_price)}/hour^2", {
        "prices": (times, prices, dict(linestyle='-')),
        "fit": (times, np.polyval(best.polyfit(2), times), dict(linestyle='--'))
    })


def wave_fit():
    candles = Candles('MDA/BTC', '1h', 24)
    fit = candles.wavefit(slice(1,4))
    show_plot(f"MDA/BTC amp={percentage(fit.amp / fit.zero)} freq={fit.freq} phase={round(fit.phase, 1)}", {
        "prices": (*candles.prices, dict(linestyle='-')),
        "fit":    (*fit.prices, dict(linestyle='--'))
    })


def find_volatile_coin_near_min():
    # slice(2,6) makes it so we only consider coins with at least 2 wave cycles in the last 6 hours.
    # Hopefully that suggests a pattern that will continue?
    fit_hours = 6
    candles_per_hour = 12
    fits = [Candles(symbol, '5m', fit_hours*candles_per_hour).wavefit(slice(2, 6)) for symbol in get_symbols()]

    def is_at_min(fit):
        # This finds where the cos wave fit is at minimum but the real price might not be... Especially if freq is high
        phase = math.cos(fit.phase - math.pi)
        # So we'll also make sure price is at bottom of its last half wave
        wave_length = fit_hours / fit.freq
        last_half_wave = fit.candles[-int(wave_length * candles_per_hour / 2):]
        last_half_wave_mix = unmix(fit.candles.end_price, last_half_wave.max, last_half_wave.min)
        # Require both measures by multiplying them...
        return (phase*.5+.5) * last_half_wave_mix

    # Dividing by error makes sure prices fit the sin wave good but this is at the expense of finding the most volatile coin
    # You can try with and without error...
    #error = lambda fit: (1 + fit.rmse*1e2/fit.zero)

    # / fit.zero makes sure we are working in percentages across all coins
    best = max(fits, key=lambda fit: fit.amp * fit.freq * is_at_min(fit) / fit.zero)# / error(fit))
    show_plot(f"{best.candles.symbol} amp={percentage(best.amp / best.zero)} freq={best.freq}", {
        "prices": (*best.candles.prices, dict(linestyle='-')),
        "fit":    (*best.prices, dict(linestyle='--'))
    })


def find_order_book_with_most_bids_to_asks():
    orderbooks = [(symbol, reduce_order_book(symbol)) for symbol in get_symbols()]
    orderbooks.sort(key=lambda ob: ob[1], reverse=True)
    for symbol, ob in orderbooks[:5]:
        print(f"{symbol} bids to asks={round(ob, 2)}")
    return orderbooks[0][0]


if __name__ == "__main__":
    find_coin_with_highest_velocity()
    find_coin_with_highest_acceleration()
    wave_fit()
    find_volatile_coin_near_min()
    find_order_book_with_most_bids_to_asks()



