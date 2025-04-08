import jsonpickle
import numpy as np
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List


def calc_kelp_fair_value_reg(state: TradingState) -> int:
    previous_price = None
    if state.traderData:
        previous_state = jsonpickle.decode(state.traderData)
        previous_price = previous_state.get('kelp_last_price', None)

    order_depth: OrderDepth = state.order_depths['KELP']
    buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
    sell_orders = sorted(order_depth.sell_orders.items())

    popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
    popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

    fair_value = round((popular_buy_price + popular_sell_price) / 2)

    if not previous_price:
        return fair_value
    else:
        # previous_logr = np.log(fair_value / previous_price)
        # next_logr = previous_logr * -0.262
        # return fair_value * np.exp(next_logr)
        previous_return = (fair_value - previous_price) / previous_price
        next_return = -0.229 * previous_return
        return fair_value + (fair_value * next_return)


def calc_kelp_fair_value(state: TradingState) -> int:
    order_depth: OrderDepth = state.order_depths['KELP']
    buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
    sell_orders = sorted(order_depth.sell_orders.items())

    popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
    popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

    fair_value = round((popular_buy_price + popular_sell_price) / 2)

    return fair_value


def trade_resin(state: TradingState) -> List[Order]:
    curr_position = state.position.get('RAINFOREST_RESIN', 0)
    # print(f'RAINFOREST_RESIN position: {curr_position}')

    max_position = 50
    max_buy_size = min(max_position, max_position - curr_position)
    max_sell_size = max(-max_position, -max_position - curr_position)

    fair_value = 10000
    # print(f'RAINFOREST_RESIN fair value: {fair_value}')

    thr_l = fair_value - 2
    thr_h = fair_value + 2
    buy_price = thr_l
    sell_price = thr_h

    orders = []
    if max_buy_size > 0:
        # print(f'BUY: {max_buy_size} @ {buy_price}')
        orders.append(Order('RAINFOREST_RESIN', buy_price, max_buy_size))
    if max_sell_size < 0:
        # print(f'SELL: {max_sell_size} @ {sell_price}')
        orders.append(Order('RAINFOREST_RESIN', sell_price, max_sell_size))

    return orders


def trade_kelp(state: TradingState, fair_value: float) -> List[Order]:
    curr_position = state.position.get('KELP', 0)
    # print(f'KELP position: {curr_position}')

    max_position = 50
    max_buy_size = min(max_position, max_position - curr_position)
    max_sell_size = max(-max_position, -max_position - curr_position)

    orders = []
    if max_buy_size > 0:
        buy_price = int(np.floor(fair_value)) - 1
        # print(f'BUY: {max_buy_size} @ {buy_price}')
        orders.append(Order('KELP', buy_price, max_buy_size))
    if max_sell_size < 0:
        sell_price = int(np.ceil(fair_value)) + 1
        # print(f'SELL: {max_sell_size} @ {sell_price}')
        orders.append(Order('KELP', sell_price, max_sell_size))

    return orders


def trade_ink(state: TradingState) -> List[Order]:
    curr_position = state.position.get('SQUID_INK', 0)
    print(f'SQUID_INK position: {curr_position}')

    order_depth: OrderDepth = state.order_depths['SQUID_INK']

    max_position = 50
    max_buy_size = min(max_position, max_position - curr_position)
    max_sell_size = max(-max_position, -max_position - curr_position)

    buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
    sell_orders = sorted(order_depth.sell_orders.items())

    popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
    popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

    fair_value = round((popular_buy_price + popular_sell_price) / 2)
    print(f'SQUID_INK fair value: {fair_value}')

    orders = []
    if max_buy_size > 0:
        buy_price = int(np.floor(fair_value)) - 1
        print(f'BUY: {max_buy_size} @ {buy_price}')
        orders.append(Order('SQUID_INK', buy_price, max_buy_size))
    if max_sell_size < 0:
        sell_price = int(np.ceil(fair_value)) + 1
        print(f'SELL: {max_sell_size} @ {sell_price}')
        orders.append(Order('SQUID_INK', sell_price, max_sell_size))

    return orders


class Trader:
    def run(self, state: TradingState):
        conversions = 0
        trader_data = None

        result = {}
        for product in state.order_depths:
            orders: List[Order] = []
            if product == 'RAINFOREST_RESIN':
                orders.extend(trade_resin(state))
            elif product == 'KELP':
                kelp_fair_value = calc_kelp_fair_value(state)
                # print(f'KELP fair value: {kelp_fair_value}')
                orders.extend(trade_kelp(state, kelp_fair_value))
            # elif product == 'SQUID_INK':
            #     orders.extend(trade_ink(state))
            result[product] = orders
            # print('---')

        # trader_data = jsonpickle.encode({'kelp_last_price': kelp_fair_value})
        # print(trader_data)
        return result, conversions, trader_data