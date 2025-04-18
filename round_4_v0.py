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
    strike_price: int = None
    time_to_expiry: float = None
    implied_vol: float = None
    delta: float = None
    moneyness: float = None


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
    product_names: Tuple[str] = ('PICNIC_BASKET1', 'PICNIC_BASKET2', 'CROISSANTS', 'JAMS', 'DJEMBES')
    product_weights: Tuple[int] = (1, -1, -2, -1, -1)  # Basket1, Basket2, Croissants, Jams, Djembes
    mean: float = 32.9
    std: float = 96.4


@dataclass
class Macarons(Product):
    name: str = 'MAGNIFICENT_MACARONS'
    limit: int = 75
    conversion_limit: int = 10
    make_min_edge: int = 1
    make_probability: float = 0.5
    init_make_edge: int = 2
    min_edge: float = 0.5
    volume_avg_timestamp: int = 5
    volume_bar: int = 5
    dec_edge_discount: float = 0.8
    step_size: float = 0.5


class VolcanicRock:
    def __init__(self):
        self.spot: Product = Product(name='VOLCANIC_ROCK', limit=400)
        self.strike_prices: List[int] = [9500, 9750, 10000, 10250, 10500]
        self.vol_spread_window = 100
        self.vol_spread_mean = 0.65
        self.vol_spread_std = 0.16
        self.vol_spread_z_score_thr = 1
        self.call_options: List[CallOption] = [
            CallOption(name=f'VOLCANIC_ROCK_VOUCHER_{strike}', limit=200, strike_price=strike, time_to_expiry=5 / 250)
            for strike in self.strike_prices]


class BlackScholes:
    def __init__(self, spot, strike, time_to_expiry):
        self.spot = spot
        self.strike = strike
        self.time_to_expiry = time_to_expiry

    def call_price(self, volatility):
        d1 = (
            log(self.spot) - log(self.strike) + (0.5 * volatility * volatility) * self.time_to_expiry
        ) / (volatility * sqrt(self.time_to_expiry))
        d2 = d1 - volatility * sqrt(self.time_to_expiry)
        call_price = self.spot * NormalDist().cdf(d1) - self.strike * NormalDist().cdf(d2)
        return call_price

    def delta(self, volatility):
        d1 = (
            log(self.spot) - log(self.strike) + (0.5 * volatility * volatility) * self.time_to_expiry
        ) / (volatility * sqrt(self.time_to_expiry))
        return NormalDist().cdf(d1)

    def gamma(self, volatility):
        d1 = (
            log(self.spot) - log(self.strike) + (0.5 * volatility * volatility) * self.time_to_expiry
        ) / (volatility * sqrt(self.time_to_expiry))
        return NormalDist().pdf(d1) / (self.spot * volatility * sqrt(self.time_to_expiry))

    def vega(self, volatility):
        d1 = (
            log(self.spot) - log(self.strike) + (0.5 * volatility * volatility) * self.time_to_expiry
        ) / (volatility * sqrt(self.time_to_expiry))
        return NormalDist().pdf(d1) * (self.spot * sqrt(self.time_to_expiry)) / 100

    def implied_volatility(self, market_call_price, max_iterations=200, tolerance=1e-10):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = self.call_price(volatility)
            diff = estimated_price - market_call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


def macarons_implied_bid(observation: ConversionObservation) -> float:
    return observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1


def macarons_implied_ask(observation: ConversionObservation) -> float:
    return observation.askPrice + observation.importTariff + observation.transportFees


def macarons_adap_edge(mac: Macarons, timestamp: int, curr_edge: float, trader_object: dict) -> float:
    if timestamp == 0:
        trader_object["ORCHIDS"]["curr_edge"] = mac.init_make_edge
        return mac.init_make_edge

    # Timestamp not 0
    trader_object["ORCHIDS"]["volume_history"].append(abs(mac.position))
    if len(trader_object["ORCHIDS"]["volume_history"]) > mac.volume_avg_timestamp:
        trader_object["ORCHIDS"]["volume_history"].pop(0)

    if len(trader_object["ORCHIDS"]["volume_history"]) < mac.volume_avg_timestamp:
        return curr_edge
    elif not trader_object["ORCHIDS"]["optimized"]:
        volume_avg = np.mean(trader_object["ORCHIDS"]["volume_history"])

        # Bump up edge if consistently getting lifted full size
        if volume_avg >= mac.volume_bar:
            trader_object["ORCHIDS"]["volume_history"] = [] # clear volume history if edge changed
            trader_object["ORCHIDS"]["curr_edge"] = curr_edge + mac.step_size
            return curr_edge + mac.step_size

        # Decrease edge if more cash with less edge, included discount
        elif mac.dec_edge_discount * mac.volume_bar * (curr_edge - mac.step_size) > volume_avg * curr_edge:
            if curr_edge - mac.step_size > mac.min_edge:
                trader_object["ORCHIDS"]["volume_history"] = [] # clear volume history if edge changed
                trader_object["ORCHIDS"]["curr_edge"] = curr_edge - mac.step_size
                trader_object["ORCHIDS"]["optimized"] = True
                return curr_edge - mac.step_size
            else:
                trader_object["ORCHIDS"]["curr_edge"] = mac.min_edge
                return mac.min_edge

    trader_object["ORCHIDS"]["curr_edge"] = curr_edge
    return curr_edge


def macarons_arb_take(mac: Macarons, order_depth: OrderDepth, observation: ConversionObservation, adap_edge: float
                     ) -> List[Order]:
    orders: List[Order] = []
    position_limit = mac.limit
    buy_order_volume = 0
    sell_order_volume = 0

    implied_bid = macarons_implied_bid(observation)
    implied_ask = macarons_implied_ask(observation)

    buy_quantity = min(position_limit - mac.position, position_limit)
    sell_quantity = min(position_limit + mac.position, position_limit)

    ask = implied_ask + adap_edge

    foreign_mid = (observation.askPrice + observation.bidPrice) / 2
    aggressive_ask = foreign_mid - 1.6

    if aggressive_ask > implied_ask:
        ask = aggressive_ask

    edge = (ask - implied_ask) * mac.make_probability

    for price in sorted(list(order_depth.sell_orders.keys())):
        if price > implied_bid - edge:
            break

        if price < implied_bid - edge:
            quantity = min(abs(order_depth.sell_orders[price]), buy_quantity) # max amount to buy
            if quantity > 0:
                orders.append(Order(mac.name, round(price), quantity))
                buy_order_volume += quantity

    for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
        if price < implied_ask + edge:
            break

        if price > implied_ask + edge:
            quantity = min(abs(order_depth.buy_orders[price]), sell_quantity) # max amount to sell
            if quantity > 0:
                orders.append(Order(mac.name, round(price), -quantity))
                sell_order_volume += quantity

    mac.posted_buy_volume += buy_order_volume
    mac.posted_sell_volume += sell_order_volume

    return orders


def macarons_arb_clear(mac: Macarons) -> int:
    conversion_size = min(mac.conversion_limit, abs(mac.position))
    if mac.position > 0:
        return -conversion_size
    else:
        return conversion_size


def macarons_arb_make(mac: Macarons, order_depth: OrderDepth, observation: ConversionObservation, edge: float
                      ) -> List[Order]:
    orders: List[Order] = []
    position_limit = mac.limit

    implied_bid = macarons_implied_bid(observation)
    implied_ask = macarons_implied_ask(observation)

    bid = implied_bid - edge
    ask = implied_ask + edge

    # ask = foreign_mid - 1.6 best performance so far
    foreign_mid = (observation.askPrice + observation.bidPrice) / 2
    aggressive_ask = foreign_mid - 1.6 # Aggressive ask

    # don't lose money
    if aggressive_ask >= implied_ask + mac.min_edge:
        ask = aggressive_ask
        # print("AGGRESSIVE")
        # print(f"ALGO ASK: {round(ask)}")
        # print(f"ALGO BID: {round(bid)}")
    else:
        pass
        # print(f"ALGO ASK: {round(ask)}")
        # print(f"ALGO BID: {round(bid)}")

    filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 10]
    filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 20]

    # If we're not best level, penny until min edge
    if len(filtered_ask) > 0 and ask > filtered_ask[0]:
        if filtered_ask[0] - 1 > implied_ask:
            ask = filtered_ask[0] - 1
        else:
            ask = implied_ask + edge
    if len(filtered_bid) > 0 and  bid < filtered_bid[0]:
        if filtered_bid[0] + 1 < implied_bid:
            bid = filtered_bid[0] + 1
        else:
            bid = implied_bid - edge

    buy_quantity = min(position_limit - (mac.position + mac.posted_buy_volume), position_limit)
    if buy_quantity > 0:
        orders.append(Order(mac.name, round(bid), buy_quantity))  # Buy order

    sell_quantity = min(position_limit + (mac.position - mac.posted_sell_volume), position_limit)
    if sell_quantity > 0:
        orders.append(Order(mac.name, round(ask), -sell_quantity))  # Sell order

    return orders


def calc_time_to_expiry(day, ts):
    return (8 - day) / 250 - ts / 1_000_000 / 250


def calc_implied_vol(spot_price: float, strike_price: float, time_to_expiry: float, call_price: float) -> float:
    bs = BlackScholes(spot=spot_price, strike=strike_price, time_to_expiry=time_to_expiry)
    return bs.implied_volatility(call_price)


def calc_delta(spot_price: float, strike_price: float, time_to_expiry: float, implied_vol: float) -> float:
    bs = BlackScholes(spot=spot_price, strike=strike_price, time_to_expiry=time_to_expiry)
    return bs.delta(implied_vol)


def calc_moneyness(spot_price: float, strike_price: float, time_to_expiry: float) -> float:
    return np.log(strike_price / spot_price) / np.sqrt(time_to_expiry)


def calc_base_iv(rock: VolcanicRock) -> float:
    m_list = []
    v_list = []
    for call in rock.call_options:
        m_list.append(call.moneyness)
        v_list.append(call.implied_vol)
    params = np.polyfit(m_list, v_list, 2)  # params = [c, b, a]
    base_iv = float(params[0])
    if base_iv > 0:
        return base_iv
    else:
        return v_list[list(np.argsort(np.abs(np.array(m_list))))[0]]


def sort_call_by_delta(rock: VolcanicRock) -> List[int]:
    deltas = [call.delta for call in rock.call_options]
    sorted_idx = list(np.argsort(np.abs(np.array(deltas) - 0.5)))
    return sorted_idx


def sort_call_by_moneyness(rock: VolcanicRock) -> List[int]:
    calls_moneyness = [call.moneyness for call in rock.call_options]
    sorted_idx = list(np.argsort(np.abs(np.array(calls_moneyness))))
    return sorted_idx


def set_rock_call_greeks(state: TradingState, rock: VolcanicRock) -> VolcanicRock:
    time_to_expiry = calc_time_to_expiry(day=3, ts=state.timestamp)
    for call in rock.call_options:
        call.implied_vol = calc_implied_vol(rock.spot.fair_value, call.strike_price, time_to_expiry, call.fair_value)
        call.delta = calc_delta(rock.spot.fair_value, call.strike_price, time_to_expiry, call.implied_vol)
        call.moneyness = calc_moneyness(rock.spot.fair_value, call.strike_price, time_to_expiry)
        call.time_to_expiry = time_to_expiry
    return rock


def set_rock_positions(state: TradingState, rock: VolcanicRock) -> VolcanicRock:
    rock.spot.position = state.position.get(rock.spot.name, 0)
    for call in rock.call_options:
        call.position = state.position.get(call.name, 0)
    return rock


def set_rock_orderbook_data(state: TradingState, rock: VolcanicRock) -> VolcanicRock:
    order_depths: Dict[str, OrderDepth] = state.order_depths
    products = [rock.spot] + rock.call_options

    for product in products:
        if order_depths[product.name].buy_orders and order_depths[product.name].sell_orders:
            best_bid = max(order_depths[product.name].buy_orders.keys())
            best_ask = min(order_depths[product.name].sell_orders.keys())
            best_bid_size = order_depths[product.name].buy_orders[best_bid]
            best_ask_size = -order_depths[product.name].sell_orders[best_ask]
            product.best_bid = best_bid
            product.best_ask = best_ask
            product.best_bid_size = best_bid_size
            product.best_ask_size = best_ask_size

    return rock


def get_rock_mid_prices(state: TradingState, rock: VolcanicRock) -> List[float]:
    order_depths: Dict[str, OrderDepth] = state.order_depths
    products = [rock.spot] + rock.call_options

    mid_prices = []
    for product in products:
        best_bid = (max(order_depths[product.name].buy_orders.keys())
                    if order_depths[product.name].buy_orders else None)
        best_ask = (min(order_depths[product.name].sell_orders.keys())
                    if order_depths[product.name].sell_orders else None)
        if best_bid and best_ask:
            mid_prices.append((best_bid + best_ask) / 2)
        else:
            mid_prices.append(None)
    return mid_prices


def get_spread_position(state: TradingState) -> int:
    return state.position.get('PICNIC_BASKET1', 0)


def get_target_spread_position_size(spread: Spread) -> int:
    zscore = (spread.fair_value - spread.mean) / spread.std

    thr = 0.8
    if zscore < -thr:
        target_position = spread.limit
    elif zscore > thr:
        target_position = -spread.limit
    else:
        target_position = spread.position

    return target_position


def get_spread_products_orders(state: TradingState) -> Tuple[List[int], List[int], List[int], List[int]]:
    order_depths: Dict[str, OrderDepth] = state.order_depths
    products = [Basket1(), Basket2(), Croissants(), Jams(), Djembes()]

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

    return best_bids, best_asks, best_bid_volumes, best_ask_volumes


def get_spread_order_depth(state: TradingState) -> OrderDepth:
    best_bids, best_asks, best_bid_volumes, best_ask_volumes = get_spread_products_orders(state)

    spread = Spread()
    spread_order_depth = OrderDepth()
    product_weights = spread.product_weights
    spread_bid, spread_ask = 0, 0
    spread_bid_volumes, spread_ask_volumes = [], []
    for bid, ask, bid_vol, ask_vol, w in zip(best_bids, best_asks, best_bid_volumes, best_ask_volumes, product_weights):
        if w > 0:
            spread_bid += bid * w
            spread_ask += ask * w
            spread_bid_volumes.append(abs(bid_vol // w))
            spread_ask_volumes.append(abs(ask_vol // w))
        if w < 0:
            spread_bid += ask * w
            spread_ask += bid * w
            spread_bid_volumes.append(abs(ask_vol // w))
            spread_ask_volumes.append(abs(bid_vol // w))
    spread_bid_volume = min(spread_bid_volumes)
    spread_ask_volume = min(spread_ask_volumes)
    spread_order_depth.buy_orders[spread_bid] = spread_bid_volume
    spread_order_depth.sell_orders[spread_ask] = -spread_ask_volume

    return spread_order_depth


def get_spread_mid_price(state: TradingState) -> float:
    spread_order_depth: OrderDepth = get_spread_order_depth(state)
    return (max(spread_order_depth.buy_orders.keys()) + min(spread_order_depth.sell_orders.keys())) / 2


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


def trade_spread(state: TradingState, spread: Spread) -> Dict[str, List[Order]]:
    order_depth = get_spread_order_depth(state)
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    best_bid_size = abs(order_depth.buy_orders[best_bid])
    best_ask_size = abs(order_depth.sell_orders[best_ask])
    current_position = spread.position
    target_position = get_target_spread_position_size(spread)
    position_diff = round(current_position - target_position)
    print(f'Spread bid size: {best_bid_size}')
    print(f'Spread ask size: {best_ask_size}')
    print(f'Current spread position: {current_position}')
    print(f'Target spread position: {target_position}')
    print(f'Spread position diff: {position_diff}')

    products = [Basket1(), Basket2(), Croissants(), Jams(), Djembes()]
    best_bids, best_asks, _, _ = get_spread_products_orders(state)

    orders: Dict[str, List[Order]] = {key: [] for key in spread.product_names}
    if position_diff > 0:  # sell spread
        size = min(position_diff, best_bid_size)
        for product, w, bid, ask in zip(products, spread.product_weights, best_bids, best_asks):
            if w > 0:
                orders[product.name].append(Order(product.name, round(bid), -abs(size * w)))  # sell product
            else:
                orders[product.name].append(Order(product.name, round(ask), abs(size * w)))  # buy product
    elif position_diff < 0:  # buy spread
        size = min(-position_diff, best_ask_size)
        for product, w, bid, ask in zip(products, spread.product_weights, best_bids, best_asks):
            if w > 0:
                orders[product.name].append(Order(product.name, round(ask), abs(size * w)))  # buy product
            else:
                orders[product.name].append(Order(product.name, round(bid), -abs(size * w)))  # sell product

    return orders


def hedge_rock(rock: VolcanicRock) -> List[Order]:
    orders: List[Order] = []

    # Calculate net delta from all option positions
    net_delta = sum(call.delta * call.position for call in rock.call_options)

    # Desired spot position to neutralize delta
    target_spot_position = -net_delta
    current_spot_position = rock.spot.position

    # Compute desired hedge quantity
    hedge_qty = int(round(target_spot_position - current_spot_position))

    # Respect position limits
    max_buy = rock.spot.limit - current_spot_position
    max_sell = rock.spot.limit + current_spot_position  # since position can be negative

    if hedge_qty > 0 and rock.spot.best_ask is not None:
        hedge_qty = min(hedge_qty, max_buy)
        if hedge_qty > 0:
            orders.append(Order(rock.spot.name, round(rock.spot.best_ask), hedge_qty))

    elif hedge_qty < 0 and rock.spot.best_bid is not None:
        hedge_qty = min(-hedge_qty, max_sell)
        if hedge_qty > 0:
            orders.append(Order(rock.spot.name, round(rock.spot.best_bid), -hedge_qty))

    return orders


def trade_rock_vouchers(rock: VolcanicRock, vol_spread: float) -> Dict[str, List[Order]]:
    orders: Dict[str, List[Order]] = {k: [] for k in [call.name for call in rock.call_options]}

    # sorted_call_idx = sort_call_by_delta(rock)[:2]
    sorted_call_idx = sort_call_by_moneyness(rock)
    for idx in sorted_call_idx[:2]:
        call = rock.call_options[idx]
        rock_vol_spread_z_score = (vol_spread - rock.vol_spread_mean) / rock.vol_spread_std

        if rock_vol_spread_z_score > rock.vol_spread_z_score_thr and call.best_bid:  # short call
            call_size = min(abs(call.limit + call.position), call.best_bid_size, call.limit)
            if call_size > 0:
                orders[call.name].append(Order(call.name, round(call.best_bid), -round(call_size)))

        if rock_vol_spread_z_score < - rock.vol_spread_z_score_thr and call.best_ask:  # long call
            call_size = min(abs(call.limit - call.position), call.best_ask_size, call.limit)
            if call_size > 0:
                orders[call.name].append(Order(call.name, round(call.best_ask), round(call_size)))

    return orders


def trade_macarons(state: TradingState, mac: Macarons) -> List[Order]:
    order_depth = state.order_depths[mac.name]
    observations = state.observations.conversionObservations[mac.name]

    orders: List[Order] = []

    max_sell_amount = min(abs(mac.limit + mac.position), mac.conversion_limit)
    implied_ask = macarons_implied_ask(observations)

    if len(order_depth.buy_orders) != 0:
        sell_price = max(int(observations.bidPrice - 0.5), int(implied_ask) + 1)
        orders.append(Order(mac.name, sell_price, -max_sell_amount))

        # bids = [bid for bid in order_depth.buy_orders.keys() if bid > implied_ask]
        # bids = sorted(bids, reverse=True)
        # if len(bids) > 0:  # sell
        #     for bid in bids:
        #         bid_size = order_depth.buy_orders[bid]
        #         sell_size = min(bid_size, max_sell_amount, (mac.limit - mac.posted_sell_volume))
        #         if sell_size > 0:
        #             orders.append(Order(mac.name, bid, -sell_size))
        #             mac.posted_sell_volume += sell_size

    return orders


class Trader:
    def run(self, state: TradingState):
        conversions = 0
        kelp_fair_value = None
        previous_state = {}
        squid_ink_prices = []
        previous_rock_prices = []
        rock_vol_spreads = []
        spot_prices = []

        if state.traderData:
            previous_state = jsonpickle.decode(state.traderData)
            squid_ink_prices = previous_state.get('squid_ink_prices', [])
            previous_rock_prices = previous_state.get('previous_rock_prices', [])
            rock_vol_spreads = previous_state.get('rock_vol_spreads', [])
            spot_prices = previous_state.get('spot_prices', [])

        result = {}
        # for product_name in state.order_depths:
            # position = state.position.get(product_name, 0)
            # print(f'{product_name} position: {position}')
            # orders: List[Order] = []
            # if product_name == 'RAINFOREST_RESIN':
            #     product = RainforestResin(position=position)
            #     orders.extend(trade_resin(state, product))
            # if product_name == 'KELP':
            #     kelp_fair_value = calc_kelp_fair_value(state)
            #     print(f'KELP fair value: {kelp_fair_value}')
            #     product = Kelp(position=position, fair_value=kelp_fair_value)
            #     orders.extend(trade_kelp(state, product))
            # if product_name == 'SQUID_INK':
            #     ink_fair_value = calc_ink_fair_value(state)
            #     print(f'SQUID_INK fair value: {ink_fair_value}')
            #     product = SquidInk(position=position, fair_value=ink_fair_value)
            #
            #     squid_ink_prices.append(ink_fair_value)
            #     if len(squid_ink_prices) > 100:
            #         squid_ink_prices = squid_ink_prices[-100:]
            #
            #     orders.extend(trade_ink(product))

            # if product_name == 'MAGNIFICENT_MACARONS':
            #     product = Macarons(position=position)
            #     print(f'{product_name} position: {position}')
            #
            #     conversions = macarons_arb_clear(product)
            #     print(f'Conversion: {conversions}')
            #
            #     # orders.extend(trade_macarons(state, product))
            #
            #     if "ORCHIDS" not in previous_state:
            #         previous_state["ORCHIDS"] = {
            #             "curr_edge": product.init_make_edge, "volume_history": [], "optimized": False}
            #
            #     adap_edge = macarons_adap_edge(product, state.timestamp,
            #                                    previous_state["ORCHIDS"]["curr_edge"], previous_state)
            #
            #     orders.extend(macarons_arb_take(product, state.order_depths[product.name],
            #         state.observations.conversionObservations[product.name], adap_edge))
            #     orders.extend(macarons_arb_make(product, state.order_depths[product.name],
            #         state.observations.conversionObservations[product.name], adap_edge))
            #
            # result[product_name] = orders
            # print('---')
        #
        # if 'PICNIC_BASKET1' in state.order_depths:
        #     spread_position = get_spread_position(state)
        #     spread_mid_price = get_spread_mid_price(state)
        #     print(f'Spread mid-price: {spread_mid_price}')
        #
        #     spread = Spread(position=spread_position, fair_value=spread_mid_price)
        #     spread_orders: Dict[str, List[Order]] = trade_spread(state, spread)
        #     for product_name, orders in spread_orders.items():
        #         result[product_name] = orders
        #         print(orders)

        if 'VOLCANIC_ROCK' in state.order_depths:
            rock = VolcanicRock()
            rock_prices = get_rock_mid_prices(state, rock)

            for i in range(len(rock.call_options) + 1):  # First one is the underlying and the rest are call options
                if rock_prices[i] is None:
                    rock_prices[i] = previous_rock_prices[i]

            # set current prices
            products = [rock.spot] + rock.call_options
            for i, product in enumerate(products):
                product.fair_value = rock_prices[i]

            # set greeks, positions, and orderbook data
            rock = set_rock_call_greeks(state, rock)
            rock = set_rock_positions(state, rock)
            rock = set_rock_orderbook_data(state, rock)

            vol_spread = calc_base_iv(rock)
            rock_vol_spreads.append(vol_spread)
            if len(rock_vol_spreads) > rock.vol_spread_window:
                rock_vol_spreads.pop(0)

            spot_prices.append(rock.spot.fair_value)
            if len(spot_prices) > rock.vol_spread_window:
                spot_prices.pop(0)

            if len(spot_prices) == rock.vol_spread_window:
                realized_vol = (
                        np.std(np.log(np.array(spot_prices[1:]) / np.array(spot_prices[:-1]))) * np.sqrt(10000 * 250))
                vol_spread = np.mean(rock_vol_spreads)
                vol_spread = vol_spread - realized_vol

                rock_orders: List[Order] = hedge_rock(rock)
                result[rock.spot.name] = rock_orders

                voucher_orders: Dict[str, List[Order]] = trade_rock_vouchers(rock, vol_spread)

                for product_name, orders in voucher_orders.items():
                    result[product_name] = orders

            previous_rock_prices = rock_prices

        trader_data = jsonpickle.encode({
            'kelp_last_price': kelp_fair_value,
            'squid_ink_prices': squid_ink_prices,
            'previous_rock_prices': previous_rock_prices,
            'rock_vol_spreads': rock_vol_spreads,
            'spot_prices': spot_prices
        })
        return result, conversions, trader_data