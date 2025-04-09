import jsonpickle
import numpy as np
from datamodel import OrderDepth, TradingState, Order
from typing import List


class Product:
    def __init__(self, name: str, position: int, limit: int, fair_value: float, take_thr: int, clear_thr: int,
                 disregard_thr: int, join_thr: int, default_thr: int, soft_pos_limit: int, volume_thr: int,
                 posted_buy_volume: int, posted_sell_volume: int):
        self.name = name
        self.position = position
        self.limit = limit
        self.fair_value = fair_value
        self.take_thr = take_thr
        self.clear_thr = clear_thr
        self.disregard_thr = disregard_thr
        self.join_thr = join_thr
        self.default_thr = default_thr
        self.soft_pos_limit = soft_pos_limit
        self.volume_thr = volume_thr
        self.posted_buy_volume = posted_buy_volume
        self.posted_sell_volume = posted_sell_volume


class RainforestResin(Product):
    def __init__(self, name='RAINFOREST_RESIN', position=0, limit=50, fair_value=10000, take_thr=1, clear_thr=0,
                 disregard_thr=1, join_thr=2, default_thr=4, soft_pos_limit=40, volume_thr=None,
                 posted_buy_volume=0, posted_sell_volume=0):
        super().__init__(name, position, limit, fair_value, take_thr, clear_thr, disregard_thr, join_thr, default_thr,
                         soft_pos_limit, volume_thr, posted_buy_volume, posted_sell_volume)


class Kelp(Product):
    def __init__(self, name='KELP', position=0, limit=50, fair_value=2000, take_thr=1, clear_thr=0, disregard_thr=1,
                 join_thr=0, default_thr=1, soft_pos_limit=None, volume_thr=15,
                 posted_buy_volume=0, posted_sell_volume=0):
        super().__init__(name, position, limit, fair_value, take_thr, clear_thr, disregard_thr, join_thr, default_thr,
                         soft_pos_limit, volume_thr, posted_buy_volume, posted_sell_volume)


class SquidInk(Product):
    def __init__(self, name='SQUID_INK', position=0, limit=50, fair_value=2000, take_thr=0, clear_thr=0, disregard_thr=0,
                 join_thr=0, default_thr=0, soft_pos_limit=30, volume_thr=15, posted_buy_volume=0, posted_sell_volume=0):
        super().__init__(name, position, limit, fair_value, take_thr, clear_thr, disregard_thr,
                         join_thr, default_thr, soft_pos_limit, volume_thr,
                         posted_buy_volume, posted_sell_volume)


def update_volatility(prices: List[float], lambda_: float = 0.94, vol_floor=0.5) -> float:
    if len(prices) < 2:
        return vol_floor
    log_returns = np.diff(np.log(prices))
    squared_returns = log_returns**2
    weights = (1 - lambda_) * lambda_**np.arange(len(squared_returns))[::-1]
    ewma_vol = np.sqrt(np.sum(weights * squared_returns))
    return max(ewma_vol, vol_floor)


def avellaneda_stoikov_mm(product: Product, volatility: float) -> List[Order]:
    gamma = 0.2  # risk aversion
    spread = 1

    reservation_price = product.fair_value - product.position * gamma * volatility**2
    print(f'SQUID_INK reservation price: {reservation_price}')

    bid = np.floor(reservation_price) - spread
    ask = np.ceil(reservation_price) + spread

    buy_qty = product.limit - product.position
    sell_qty = product.limit + product.position

    return [
        Order(product.name, round(bid), buy_qty),
        Order(product.name, round(ask), -sell_qty)
    ]


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

    return fair_value


def market_taking(product: Product, state: TradingState) -> List[Order]:
    order_depth: OrderDepth = state.order_depths[product.name]

    orders = []
    if len(order_depth.sell_orders) != 0:
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -1 * order_depth.sell_orders[best_ask]

        if not product.volume_thr or abs(best_ask_amount) <= product.volume_thr:
            if best_ask <= product.fair_value - product.take_thr:
                quantity = min(
                    best_ask_amount, product.limit - product.position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(product.name, best_ask, quantity))
                    product.posted_buy_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

    if len(order_depth.buy_orders) != 0:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]

        if not product.volume_thr or abs(best_bid_amount) <= product.volume_thr:
            if best_bid >= product.fair_value + product.take_thr:
                quantity = min(
                    best_bid_amount, product.limit + product.position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(product.name, best_bid, -1 * quantity))
                    product.posted_sell_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

    return orders


def position_clearance(product: Product, state: TradingState) -> List[Order]:
    order_depth: OrderDepth = state.order_depths[product.name]
    position_after_take = product.position + product.posted_buy_volume - product.posted_sell_volume

    fair_for_bid = round(product.fair_value - product.clear_thr)
    fair_for_ask = round(product.fair_value + product.clear_thr)

    buy_quantity = product.limit - (product.position + product.posted_buy_volume)
    sell_quantity = product.limit + (product.position - product.posted_sell_volume)

    orders = []
    if position_after_take > 0:
        # Aggregate volume from all buy orders with price greater than fair_for_ask
        clear_quantity = sum(
            volume
            for price, volume in order_depth.buy_orders.items()
            if price >= fair_for_ask
        )
        clear_quantity = min(clear_quantity, position_after_take)
        sent_quantity = min(sell_quantity, clear_quantity)
        if sent_quantity > 0:
            orders.append(Order(product.name, fair_for_ask, -abs(sent_quantity)))
            product.posted_sell_volume += abs(sent_quantity)

    if position_after_take < 0:
        # Aggregate volume from all sell orders with price lower than fair_for_bid
        clear_quantity = sum(
            abs(volume)
            for price, volume in order_depth.sell_orders.items()
            if price <= fair_for_bid
        )
        clear_quantity = min(clear_quantity, abs(position_after_take))
        sent_quantity = min(buy_quantity, clear_quantity)
        if sent_quantity > 0:
            orders.append(Order(product.name, fair_for_bid, abs(sent_quantity)))
            product.posted_buy_volume += abs(sent_quantity)

    return orders


def market_making(product: Product, state: TradingState) -> List[Order]:
    order_depth: OrderDepth = state.order_depths[product.name]

    asks_above_fair = [
        price
        for price in order_depth.sell_orders.keys()
        if price > product.fair_value + product.disregard_thr
    ]
    bids_below_fair = [
        price
        for price in order_depth.buy_orders.keys()
        if price < product.fair_value - product.disregard_thr
    ]

    best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
    best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

    ask = round(product.fair_value + product.default_thr)
    if best_ask_above_fair is not None:
        if abs(best_ask_above_fair - product.fair_value) <= product.join_thr:
            ask = best_ask_above_fair  # join
        else:
            ask = best_ask_above_fair - 1  # penny

    bid = round(product.fair_value - product.default_thr)
    if best_bid_below_fair is not None:
        if abs(product.fair_value - best_bid_below_fair) <= product.join_thr:
            bid = best_bid_below_fair
        else:
            bid = best_bid_below_fair + 1

    if product.soft_pos_limit:
        if product.position > product.soft_pos_limit:
            ask -= 1
        elif product.position < -1 * product.soft_pos_limit:
            bid += 1

    orders: List[Order] = []
    buy_quantity = product.limit - (product.position + product.posted_buy_volume)
    if buy_quantity > 0:
        orders.append(Order(product.name, round(bid), buy_quantity))  # Buy order

    sell_quantity = product.limit + (product.position - product.posted_sell_volume)
    if sell_quantity > 0:
        orders.append(Order(product.name, round(ask), -sell_quantity))  # Sell order

    return orders


class Trader:
    def run(self, state: TradingState):
        conversions = 0
        kelp_fair_value = None

        squid_ink_prices = []
        if state.traderData:
            previous_state = jsonpickle.decode(state.traderData)
            squid_ink_prices = previous_state.get('squid_ink_prices', [])

        result = {}
        for product_name in state.order_depths:
            position = state.position.get(product_name, 0)
            print(f'{product_name} position: {position}')
            orders: List[Order] = []
            if product_name == 'RAINFOREST_RESIN':
                product = RainforestResin(position=position)
                orders.extend(market_taking(product, state))
                orders.extend(position_clearance(product, state))
                orders.extend(market_making(product, state))
            if product_name == 'KELP':
                kelp_fair_value = calc_kelp_fair_value(state)
                print(f'KELP fair value: {kelp_fair_value}')
                product = Kelp(position=position, fair_value=kelp_fair_value)
                orders.extend(market_taking(product, state))
                orders.extend(position_clearance(product, state))
                orders.extend(market_making(product, state))
            if product_name == 'SQUID_INK':
                ink_fair_value = calc_ink_fair_value(state)
                print(f'SQUID_INK fair value: {ink_fair_value}')
                product = SquidInk(position=position, fair_value=ink_fair_value)

                squid_ink_prices.append(ink_fair_value)
                if len(squid_ink_prices) > 100:
                    squid_ink_prices = squid_ink_prices[-100:]

                volatility = update_volatility(squid_ink_prices)
                orders.extend(avellaneda_stoikov_mm(product, volatility))

            result[product_name] = orders
            print('---')

        trader_data = jsonpickle.encode({
            'kelp_last_price': kelp_fair_value,
            'squid_ink_prices': squid_ink_prices
        })
        return result, conversions, trader_data