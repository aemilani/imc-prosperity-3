import jsonpickle
import numpy as np
from datamodel import OrderDepth, TradingState, Order
from typing import List


def buy(product: str, buy_price: int, buy_size: int) -> List[Order]:
    orders = []
    if buy_size > 0:
        print(f'BUY: {buy_size} @ {buy_price}')
        orders.append(Order(product, buy_price, buy_size))
    return orders


def sell(product: str, sell_price: int, sell_size: int) -> List[Order]:
    orders = []
    if sell_size < 0:
        print(f'SELL: {sell_size} @ {sell_price}')
        orders.append(Order(product, sell_price, sell_size))
    return orders


def market_making(product: str, buy_price: int, buy_size: int,
                  sell_price: int, sell_size: int) -> List[Order]:
    orders = []
    orders.extend(buy(product, buy_price, buy_size))
    orders.extend(sell(product, sell_price, sell_size))
    return orders


def calc_kelp_fair_value(state: TradingState) -> float:
    previous_price = None
    if state.traderData:
        previous_state = jsonpickle.decode(state.traderData)
        previous_price = previous_state.get('kelp_last_price', None)

    order_depth: OrderDepth = state.order_depths['KELP']

    best_ask = min(order_depth.sell_orders.keys())
    best_bid = max(order_depth.buy_orders.keys())

    filtered_asks = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
    filtered_bids = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
    best_filtered_ask = min(filtered_asks) if len(filtered_asks) > 0 else None
    best_filtered_bid = max(filtered_bids) if len(filtered_bids) > 0 else None

    if best_filtered_ask and best_filtered_bid:
        fair_value = (best_filtered_ask + best_filtered_bid) / 2
    else:
        fair_value = (best_ask + best_bid) / 2

    if not previous_price:
        return fair_value
    else:
        previous_logr = np.log(fair_value / previous_price)
        next_logr = previous_logr * -0.27
        return fair_value * np.exp(next_logr)


def calc_ink_fair_value(state: TradingState) -> float:
    order_depth: OrderDepth = state.order_depths['SQUID_INK']

    buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
    sell_orders = sorted(order_depth.sell_orders.items())

    popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
    popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

    fair_value = (popular_buy_price + popular_sell_price) / 2

    return fair_value


def trade_resin(state: TradingState) -> List[Order]:
    curr_position = state.position.get('RAINFOREST_RESIN', 0)
    print(f'RAINFOREST_RESIN position: {curr_position}')

    max_position = 50
    max_buy_size = min(max_position, max_position - curr_position)
    max_sell_size = max(-max_position, -max_position - curr_position)

    fair_value = 10000
    print(f'RAINFOREST_RESIN fair value: {fair_value}')

    thr_l = fair_value - 2
    thr_h = fair_value + 2
    buy_price = thr_l
    sell_price = thr_h

    orders = []
    orders.extend(market_making('RAINFOREST_RESIN', buy_price, max_buy_size, sell_price, max_sell_size))

    return orders


def trade_kelp(state: TradingState, fair_value: float) -> List[Order]:
    curr_position = state.position.get('KELP', 0)
    print(f'KELP position: {curr_position}')

    order_depth: OrderDepth = state.order_depths['KELP']
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    best_bid_size = order_depth.buy_orders[best_bid]
    best_ask_size = order_depth.sell_orders[best_ask]

    max_position = 50
    max_buy_size = min(max_position, max_position - curr_position)
    max_sell_size = max(-max_position, -max_position - curr_position)

    orders = []

    # Market taking
    buy_size = min(max_buy_size, -best_ask_size)
    sell_size = max(max_sell_size, -best_bid_size)
    if best_ask <= (fair_value - 1) and abs(best_ask_size) < 15:
        orders.extend(buy('KELP', best_ask, buy_size))
        max_buy_size -= buy_size

    if best_bid >= (fair_value + 1) and abs(best_bid_size) < 15:
        orders.extend(sell('KELP', best_bid, sell_size))
        max_sell_size -= sell_size

    # Market making
    buy_price = int(np.floor(fair_value))
    sell_price = int(np.ceil(fair_value))
    orders.extend(market_making('KELP', buy_price, max_buy_size, sell_price, max_sell_size))

    return orders


def trade_ink(state: TradingState, fair_value: float) -> List[Order]:
    curr_position = state.position.get('SQUID_INK', 0)
    print(f'SQUID_INK position: {curr_position}')

    max_position = 50
    max_buy_size = min(max_position, max_position - curr_position)
    max_sell_size = max(-max_position, -max_position - curr_position)

    orders = []
    if max_buy_size > 0:
        # buy_price = int(np.floor(fair_value)) - 1
        buy_price = int(np.floor(fair_value))
        print(f'BUY: {max_buy_size} @ {buy_price}')
        orders.append(Order('SQUID_INK', buy_price, max_buy_size))
    if max_sell_size < 0:
        # sell_price = int(np.ceil(fair_value)) + 1
        sell_price = int(np.ceil(fair_value))
        print(f'SELL: {max_sell_size} @ {sell_price}')
        orders.append(Order('SQUID_INK', sell_price, max_sell_size))

    return orders


class Trader:
    def run(self, state: TradingState):
        conversions = 0
        kelp_fair_value = None

        result = {}
        for product in state.order_depths:
            orders: List[Order] = []
            # if product == 'RAINFOREST_RESIN':
            #     orders.extend(trade_resin(state))
            if product == 'KELP':
                kelp_fair_value = calc_kelp_fair_value(state)
                print(f'KELP fair value: {kelp_fair_value}')
                orders.extend(trade_kelp(state, kelp_fair_value))
            # if product == 'SQUID_INK':
            #     ink_fair_value = calc_ink_fair_value(state)
            #     print(f'SQUID_INK fair value: {ink_fair_value}')
            #     orders.extend(trade_ink(state, ink_fair_value))
            result[product] = orders
            # print('---')

        trader_data = jsonpickle.encode({'kelp_last_price': kelp_fair_value})
        return result, conversions, trader_data