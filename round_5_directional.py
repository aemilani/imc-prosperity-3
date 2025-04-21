from dataclasses import dataclass
from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import List, Tuple


@dataclass
class Product:
    name: str
    limit: int
    bot: str
    fair_value: float = None
    position: int = 0
    posted_buy_volume: int = 0
    posted_sell_volume: int = 0
    best_bid: float = None
    best_ask: float = None
    best_bid_size: int = None
    best_ask_size: int = None


@dataclass
class CallOption(Product):
    bot: str
    strike_price: int = None
    time_to_expiry: float = None
    implied_vol: float = None
    delta: float = None
    vega: float = None
    moneyness: float = None


@dataclass
class RainforestResin(Product):
    name: str = 'RAINFOREST_RESIN'
    limit: int = 50
    bot: str = 'Penelope'
    fair_value: float = 10000
    take_thr: int = 1
    clear_thr: int = 0
    disregard_thr: int = 1
    join_thr: int = 2
    default_thr: int = 4
    soft_pos_limit: int = 40


@dataclass
class Kelp(Product):
    name: str = 'KELP'
    limit: int = 50
    bot: str = 'Charlie'
    take_thr: int = 1
    clear_thr: int = 0
    disregard_thr: int = 1
    join_thr: int = 0
    default_thr: int = 1
    volume_thr: int = 15


@dataclass
class SquidInk(Product):
    name: str = 'SQUID_INK'
    limit: int = 50
    bot: str = 'Charlie'


@dataclass
class Croissants(Product):
    name: str = 'CROISSANTS'
    limit: int = 250
    bot: str = 'Caesar'


@dataclass
class Jams(Product):
    name: str = 'JAMS'
    limit: int = 350
    bot: str = 'Caesar'


@dataclass
class Djembes(Product):
    name: str = 'DJEMBES'
    limit: int = 60
    bot: str = 'Caesar'


@dataclass
class Basket1(Product):
    name: str = 'PICNIC_BASKET1'
    limit: int = 60
    bot: str = 'Camilla'


@dataclass
class Basket2(Product):
    name: str = 'PICNIC_BASKET2'
    limit: int = 100
    bot: str = 'Camilla'


@dataclass
class Macarons(Product):
    name: str = 'MAGNIFICENT_MACARONS'
    limit: int = 75
    conversion_limit: int = 10
    bot: str = 'Caesar'


class VolcanicRock:
    def __init__(self):
        self.spot: Product = Product(name='VOLCANIC_ROCK', limit=400, bot = 'Caesar')
        self.strike_prices: List[int] = [9500, 9750, 10000, 10250, 10500]
        self.edge: float = 1
        self.vol_window = 30
        self.vol_spread_mean = 0.124
        self.vol_spread_std = 0.003
        self.vol_spread_z_score_thr = 1
        self.call_options: List[CallOption] = [
            CallOption(name=f'VOLCANIC_ROCK_VOUCHER_{strike}', limit=200, strike_price=strike,
                       time_to_expiry=5 / 250, bot = 'Camilla')
            for strike in self.strike_prices]


def directional_trade(state: TradingState, product: Product) -> List[Order]:
    order_depth: OrderDepth = state.order_depths[product.name]
    trades = state.market_trades.get(product.name, [])
    trades = [t for t in trades if t.timestamp == state.timestamp]

    orders: List[Order] = []
    if any(t.buyer == product.bot for t in trades) and order_depth.sell_orders:
        best_ask = min(order_depth.sell_orders.keys())
        buy_size = min(product.limit - product.position, product.limit)
        orders.append(Order(product.name, best_ask, buy_size))
    if any(t.seller == product.bot for t in trades) and order_depth.sell_orders:
        best_bid = max(order_depth.buy_orders.keys())
        sell_size = min(product.limit + product.position, product.limit)
        orders.append(Order(product.name, best_bid, -sell_size))

    return orders


class Trader:
    def run(self, state: TradingState):
        result = {}
        if 'SQUID_INK' in state.order_depths:
            position = state.position.get('SQUID_INK', 0)
            ink = SquidInk(position=position)
            result['SQUID_INK'] = directional_trade(state, ink)

        conversions = 0
        trader_data = None
        return result, conversions, trader_data