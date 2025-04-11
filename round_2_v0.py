import jsonpickle
import numpy as np
from dataclasses import dataclass
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple


@dataclass
class Product:
    name: str
    limit: int
    fair_value: float = None
    position: int = 0
    posted_buy_volume: int = 0
    posted_sell_volume: int = 0


@dataclass
class RainforestResin(Product):
    name: str = 'RAINFOREST_RESIN'
    limit: int = 50
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


@dataclass
class Croissants(Product):
    name: str = 'CROISSANTS'
    limit: int = 250


@dataclass
class Jams(Product):
    name: str = 'JAMS'
    limit: int = 350


@dataclass
class Djembes(Product):
    name: str = 'DJEMBES'
    limit: int = 60


@dataclass
class Basket1(Product):
    name: str = 'PICNIC_BASKET1'
    limit: int = 60


@dataclass
class Basket2(Product):
    name: str = 'PICNIC_BASKET2'
    limit: int = 100


@dataclass
class Spread(Product):
    name: str = 'SPREAD'
    limit: int = 60
    weights: Tuple[int] = (1, -1, -2, -1, -1)  # Basket1, Basket2, Croissants, Jams, Djembes


def buy(product_name: str, buy_price: int, buy_size: int) -> List[Order]:
    orders = []
    if buy_size > 0:
        print(f'BUY: {buy_size} @ {buy_price}')
        orders.append(Order(product_name, buy_price, buy_size))
    return orders


def sell(product_name: str, sell_price: int, sell_size: int) -> List[Order]:
    orders = []
    if sell_size < 0:
        print(f'SELL: {sell_size} @ {sell_price}')
        orders.append(Order(product_name, sell_price, sell_size))
    return orders


def get_spread_order_depth(state: TradingState) -> OrderDepth:
    order_depths: Dict[str, OrderDepth] = state.order_depths
    spread = Spread()
    products = [Basket1(), Basket2(), Croissants(), Jams(), Djembes()]
    product_weights = spread.weights

    spread_order_depth = OrderDepth()
    best_bids, best_asks, best_bid_volumes, best_ask_volumes = [], [], [], []
    for product in products:
        best_bid = (max(order_depths[product.name].buy_orders.keys())
                    if order_depths[product.name].buy_orders else 0)
        best_ask = (min(order_depths[product.name].sell_orders.keys())
                    if order_depths[product.name].sell_orders else float("inf"))
        best_bid_volume = order_depths[product.name].buy_orders[best_bid]
        best_ask_volume = -order_depths[product.name].sell_orders[best_ask]
        best_bids.append(best_bid)
        best_asks.append(best_ask)
        best_bid_volumes.append(best_bid_volume)
        best_ask_volumes.append(best_ask_volume)

    spread_bid, spread_ask = 0, 0
    spread_bid_volumes, spread_ask_volumes = [], []
    for bid, ask, bid_vol, ask_vol, w in zip(best_bids, best_asks, best_bid_volumes, best_ask_volumes, product_weights):
        if w > 0:
            spread_bid += bid * w
            spread_ask += ask * w
            spread_bid_volumes.append(bid_vol // w)
            spread_ask_volumes.append(ask_vol // w)
        if w < 0:
            spread_bid += ask * w
            spread_ask += bid * w
            spread_bid_volumes.append(ask_vol // w)
            spread_ask_volumes.append(bid_vol // w)
    spread_bid_volume = min(spread_bid_volumes)
    spread_ask_volume = min(spread_ask_volumes)
    spread_order_depth.buy_orders[spread_bid] = spread_bid_volume
    spread_order_depth.sell_orders[spread_ask] = -spread_ask_volume

    return spread_order_depth


def buy_spread(state: TradingState, spread: Spread) -> List[Order]:
    ...


def sell_spread(state: TradingState, spread: Spread) -> List[Order]:
    ...


def calc_kelp_fair_value(state: TradingState) -> float:
    previous_price = None
    if state.traderData:
        previous_state = jsonpickle.decode(state.traderData)
        previous_price = previous_state.get('kelp_last_price', None)

    order_depth: OrderDepth = state.order_depths['KELP']

    if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
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
            curr_logr = np.log(fair_value / previous_price)
            next_logr = curr_logr * -0.27  # mean-reversion param
            return fair_value * np.exp(next_logr)
    else:
        return previous_price


def calc_ink_fair_value(state: TradingState) -> float:
    order_depth: OrderDepth = state.order_depths['SQUID_INK']

    if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
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

    else:
        previous_price = None
        if state.traderData:
            previous_state = jsonpickle.decode(state.traderData)
            previous_price = previous_state.get('squid_ink_prices', [None])[-1]

        return previous_price



def trade_resin(state: TradingState, resin: RainforestResin) -> List[Order]:
    order_depth: OrderDepth = state.order_depths['RAINFOREST_RESIN']
    orders: List[Order] = []

    # Market taking
    if len(order_depth.sell_orders) != 0:
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -1 * order_depth.sell_orders[best_ask]

        if best_ask <= resin.fair_value - resin.take_thr:
            quantity = min(
                best_ask_amount, resin.limit - resin.position
            )  # max amt to buy
            if quantity > 0:
                orders.append(Order(resin.name, best_ask, quantity))
                resin.posted_buy_volume += quantity
                order_depth.sell_orders[best_ask] += quantity
                if order_depth.sell_orders[best_ask] == 0:
                    del order_depth.sell_orders[best_ask]
    if len(order_depth.buy_orders) != 0:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]

        if best_bid >= resin.fair_value + resin.take_thr:
            quantity = min(
                best_bid_amount, resin.limit + resin.position
            )  # should be the max we can sell
            if quantity > 0:
                orders.append(Order(resin.name, best_bid, -1 * quantity))
                resin.posted_sell_volume += quantity
                order_depth.buy_orders[best_bid] -= quantity
                if order_depth.buy_orders[best_bid] == 0:
                    del order_depth.buy_orders[best_bid]

    # Position clearance
    position_after_take = resin.position + resin.posted_buy_volume - resin.posted_sell_volume
    fair_for_bid = round(resin.fair_value - resin.clear_thr)
    fair_for_ask = round(resin.fair_value + resin.clear_thr)
    buy_quantity = resin.limit - (resin.position + resin.posted_buy_volume)
    sell_quantity = resin.limit + (resin.position - resin.posted_sell_volume)

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
            orders.append(Order(resin.name, fair_for_ask, -abs(sent_quantity)))
            resin.posted_sell_volume += abs(sent_quantity)

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
            orders.append(Order(resin.name, fair_for_bid, abs(sent_quantity)))
            resin.posted_buy_volume += abs(sent_quantity)

    # Market making
    asks_above_fair = [
        price
        for price in order_depth.sell_orders.keys()
        if price > resin.fair_value + resin.disregard_thr
    ]
    bids_below_fair = [
        price
        for price in order_depth.buy_orders.keys()
        if price < resin.fair_value - resin.disregard_thr
    ]
    best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
    best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

    ask = round(resin.fair_value + resin.default_thr)
    if best_ask_above_fair is not None:
        if abs(best_ask_above_fair - resin.fair_value) <= resin.join_thr:
            ask = best_ask_above_fair  # join
        else:
            ask = best_ask_above_fair - 1  # penny

    bid = round(resin.fair_value - resin.default_thr)
    if best_bid_below_fair is not None:
        if abs(resin.fair_value - best_bid_below_fair) <= resin.join_thr:
            bid = best_bid_below_fair
        else:
            bid = best_bid_below_fair + 1

    if resin.position > resin.soft_pos_limit:
        ask -= 1
    elif resin.position < -1 * resin.soft_pos_limit:
        bid += 1

    buy_quantity = resin.limit - (resin.position + resin.posted_buy_volume)
    if buy_quantity > 0:
        orders.append(Order(resin.name, round(bid), buy_quantity))  # Buy order

    sell_quantity = resin.limit + (resin.position - resin.posted_sell_volume)
    if sell_quantity > 0:
        orders.append(Order(resin.name, round(ask), -sell_quantity))  # Sell order

    return orders


def trade_kelp(state: TradingState, kelp:Kelp) -> List[Order]:
    order_depth: OrderDepth = state.order_depths['KELP']
    orders: List[Order] = []

    # Market taking
    if len(order_depth.sell_orders) != 0:
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -1 * order_depth.sell_orders[best_ask]

        if abs(best_ask_amount) <= kelp.volume_thr:
            if best_ask <= kelp.fair_value - kelp.take_thr:
                quantity = min(
                    best_ask_amount, kelp.limit - kelp.position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(kelp.name, best_ask, quantity))
                    kelp.posted_buy_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]
    if len(order_depth.buy_orders) != 0:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]

        if abs(best_bid_amount) <= kelp.volume_thr:
            if best_bid >= kelp.fair_value + kelp.take_thr:
                quantity = min(
                    best_bid_amount, kelp.limit + kelp.position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(kelp.name, best_bid, -1 * quantity))
                    kelp.posted_sell_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

    # Position clearance
    position_after_take = kelp.position + kelp.posted_buy_volume - kelp.posted_sell_volume
    fair_for_bid = round(kelp.fair_value - kelp.clear_thr)
    fair_for_ask = round(kelp.fair_value + kelp.clear_thr)
    buy_quantity = kelp.limit - (kelp.position + kelp.posted_buy_volume)
    sell_quantity = kelp.limit + (kelp.position - kelp.posted_sell_volume)

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
            orders.append(Order(kelp.name, fair_for_ask, -abs(sent_quantity)))
            kelp.posted_sell_volume += abs(sent_quantity)

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
            orders.append(Order(kelp.name, fair_for_bid, abs(sent_quantity)))
            kelp.posted_buy_volume += abs(sent_quantity)

    # Market making
    asks_above_fair = [
        price
        for price in order_depth.sell_orders.keys()
        if price > kelp.fair_value + kelp.disregard_thr
    ]
    bids_below_fair = [
        price
        for price in order_depth.buy_orders.keys()
        if price < kelp.fair_value - kelp.disregard_thr
    ]
    best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
    best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

    ask = round(kelp.fair_value + kelp.default_thr)
    if best_ask_above_fair is not None:
        if abs(best_ask_above_fair - kelp.fair_value) <= kelp.join_thr:
            ask = best_ask_above_fair  # join
        else:
            ask = best_ask_above_fair - 1  # penny

    bid = round(kelp.fair_value - kelp.default_thr)
    if best_bid_below_fair is not None:
        if abs(kelp.fair_value - best_bid_below_fair) <= kelp.join_thr:
            bid = best_bid_below_fair
        else:
            bid = best_bid_below_fair + 1

    buy_quantity = kelp.limit - (kelp.position + kelp.posted_buy_volume)
    if buy_quantity > 0:
        orders.append(Order(kelp.name, round(bid), buy_quantity))  # Buy order

    sell_quantity = kelp.limit + (kelp.position - kelp.posted_sell_volume)
    if sell_quantity > 0:
        orders.append(Order(kelp.name, round(ask), -sell_quantity))  # Sell order

    return orders


def trade_ink(ink: SquidInk) -> List[Order]:
    orders: List[Order] = []

    # Market making (Avellaneda-Stoikov)
    gamma = 0.2  # risk aversion
    spread = 1
    volatility = 0.5

    reservation_price = ink.fair_value - ink.position * gamma * volatility ** 2
    print(f'SQUID_INK reservation price: {reservation_price}')

    buy_qty = ink.limit - ink.position
    sell_qty = ink.limit + ink.position
    bid = np.floor(reservation_price) - spread
    ask = np.ceil(reservation_price) + spread

    orders.append(Order(ink.name, round(bid), buy_qty))
    orders.append(Order(ink.name, round(ask), -sell_qty))

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
                orders.extend(trade_resin(state, product))
            if product_name == 'KELP':
                kelp_fair_value = calc_kelp_fair_value(state)
                print(f'KELP fair value: {kelp_fair_value}')
                product = Kelp(position=position, fair_value=kelp_fair_value)
                orders.extend(trade_kelp(state, product))
            if product_name == 'SQUID_INK':
                ink_fair_value = calc_ink_fair_value(state)
                print(f'SQUID_INK fair value: {ink_fair_value}')
                product = SquidInk(position=position, fair_value=ink_fair_value)

                squid_ink_prices.append(ink_fair_value)
                if len(squid_ink_prices) > 100:
                    squid_ink_prices = squid_ink_prices[-100:]

                orders.extend(trade_ink(product))

            result[product_name] = orders
            print('---')

        trader_data = jsonpickle.encode({
            'kelp_last_price': kelp_fair_value,
            'squid_ink_prices': squid_ink_prices
        })
        return result, conversions, trader_data