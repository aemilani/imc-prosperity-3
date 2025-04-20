from itertools import product

import jsonpickle
import numpy as np
from dataclasses import dataclass
from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import List, Dict, Tuple
from math import log, sqrt
from statistics import NormalDist


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
class SquidInk(Product):
    name: str = 'SQUID_INK'
    limit: int = 50
    bot: str = 'Charlie'


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