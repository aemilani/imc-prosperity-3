import jsonpickle
import numpy as np
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List


class Trader:
    def run(self, state: TradingState):

        result = {}
        for product in state.order_depths:

            order_depth: OrderDepth = state.order_depths[product]
            print(f'{product} bids: {order_depth.buy_orders}')
            print(f'{product} asks: {order_depth.sell_orders}')

            if product in state.position:
                if state.position[product]:
                    curr_position: int = state.position[product]
                else:
                    curr_position = 0
            else:
                curr_position = 0
            print(f'{product} position: {curr_position}')

            orders: List[Order] = []

            # RAINFOREST_RESIN ----------------------------------------------------------------------------------------
            if product == 'RAINFOREST_RESIN':
                fair_value = 10000
                max_position = 50

                max_buy_size = min(max_position, max_position - curr_position)
                max_sell_size = max(-max_position, -max_position - curr_position)

                thr_l = fair_value - 2
                thr_h = fair_value + 2
                buy_price = thr_l
                sell_price = thr_h

                if max_buy_size > 0:
                    print(f'BUY: {max_buy_size} @ {buy_price}')
                    orders.append(Order(product, buy_price, max_buy_size))

                if max_sell_size < 0:
                    print(f'SELL: {max_sell_size} @ {sell_price}')
                    orders.append(Order(product, sell_price, max_sell_size))

            # KELP ----------------------------------------------------------------------------------------------------
            elif product == 'KELP':
                max_position = 50
                max_buy_size = min(max_position, max_position - curr_position)
                max_sell_size = max(-max_position, -max_position - curr_position)

                buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
                sell_orders = sorted(order_depth.sell_orders.items())

                popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
                popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

                fair_value = round((popular_buy_price + popular_sell_price) / 2)
                print(f'{product} fair value: {fair_value}')

                if max_buy_size > 0:
                    buy_price = int(np.floor(fair_value)) - 1
                    print(f'BUY: {max_buy_size} @ {buy_price}')
                    orders.append(Order(product, buy_price, max_buy_size))

                if max_sell_size < 0:
                    sell_price = int(np.ceil(fair_value)) + 1
                    print(f'SELL: {max_sell_size} @ {sell_price}')
                    orders.append(Order(product, sell_price, max_sell_size))

            result[product] = orders

        traderData = None

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData