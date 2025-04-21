from dataclasses import dataclass
from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import List, Tuple


@dataclass
class Product:
    name: str
    limit: int
    bots: Tuple[str, str, str, str]
    fair_value: float = None
    position: int = 0
    posted_buy_volume: int = 0
    posted_sell_volume: int = 0
    best_bid: float = None
    best_ask: float = None
    best_bid_size: int = None
    best_ask_size: int = None


@dataclass
class Croissants(Product):
    name: str = 'CROISSANTS'
    limit: int = 250
    bots: Tuple[str] = ('Olivia', 'Caesar', 'Caesar', 'Olivia')


def directional_trade(state: TradingState, product: Product) -> List[Order]:
    order_depth: OrderDepth = state.order_depths[product.name]
    trades = state.market_trades.get(product.name, [])
    trades = [t for t in trades if t.timestamp == state.timestamp - 100]

    orders: List[Order] = []
    if any(t.buyer == product.bots[0] and t.seller == product.bots[1] for t in trades) and order_depth.sell_orders:
        print(f'Buy {product.name}')
        best_ask = min(order_depth.sell_orders.keys())
        buy_size = min(product.limit - product.position, product.limit)
        orders.append(Order(product.name, best_ask, buy_size))
    elif any(t.buyer == product.bots[2] and t.seller == product.bots[3] for t in trades) and order_depth.buy_orders:
        print(f'Sell {product.name}')
        best_bid = max(order_depth.buy_orders.keys())
        sell_size = min(product.limit + product.position, product.limit)
        orders.append(Order(product.name, best_bid, -sell_size))

    return orders


class Trader:
    def run(self, state: TradingState):
        result = {}
        if 'CROISSANTS' in state.order_depths:
            position = state.position.get('CROISSANTS', 0)
            cro = Croissants(position=position)
            result['CROISSANTS'] = directional_trade(state, cro)

        conversions = 0
        trader_data = None
        return result, conversions, trader_data