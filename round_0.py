import numpy as np
from datamodel import OrderDepth, TradingState, Order
from typing import List


def trade_resin(curr_position: int) -> List[Order]:
    fair_value = 10000

    max_position = 50
    max_buy_size = min(max_position, max_position - curr_position)
    max_sell_size = max(-max_position, -max_position - curr_position)

    thr_l = fair_value - 2
    thr_h = fair_value + 2
    buy_price = thr_l
    sell_price = thr_h

    resin_orders = []
    if max_buy_size > 0:
        print(f'BUY: {max_buy_size} @ {buy_price}')
        resin_orders.append(Order('RAINFOREST_RESIN', buy_price, max_buy_size))
    if max_sell_size < 0:
        print(f'SELL: {max_sell_size} @ {sell_price}')
        resin_orders.append(Order('RAINFOREST_RESIN', sell_price, max_sell_size))

    return resin_orders


def trade_kelp(curr_position: int, order_depth: OrderDepth) -> List[Order]:
    max_position = 50
    max_buy_size = min(max_position, max_position - curr_position)
    max_sell_size = max(-max_position, -max_position - curr_position)

    buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
    sell_orders = sorted(order_depth.sell_orders.items())

    popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
    popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

    fair_value = round((popular_buy_price + popular_sell_price) / 2)
    print(f'KELP fair value: {fair_value}')

    kelp_orders = []
    if max_buy_size > 0:
        buy_price = int(np.floor(fair_value)) - 1
        print(f'BUY: {max_buy_size} @ {buy_price}')
        kelp_orders.append(Order('KELP', buy_price, max_buy_size))
    if max_sell_size < 0:
        sell_price = int(np.ceil(fair_value)) + 1
        print(f'SELL: {max_sell_size} @ {sell_price}')
        kelp_orders.append(Order('KELP', sell_price, max_sell_size))

    return kelp_orders


class Trader:
    def run(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]

            if product in state.position:
                if state.position[product]:
                    curr_position: int = state.position[product]
                else:
                    curr_position = 0
            else:
                curr_position = 0
            print(f'{product} position: {curr_position}')

            orders: List[Order] = []
            if product == 'RAINFOREST_RESIN':
                orders.extend(trade_resin(curr_position))
            elif product == 'KELP':
                orders.extend(trade_kelp(curr_position, order_depth))

            result[product] = orders

        trader_data = None
        conversions = 1
        return result, conversions, trader_data