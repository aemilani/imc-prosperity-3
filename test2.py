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
    vega: float = None
    moneyness: float = None


class VolcanicRock:
    def __init__(self):
        self.spot: Product = Product(name='VOLCANIC_ROCK', limit=400)
        self.strike_prices: List[int] = [9500, 9750, 10000, 10250, 10500]
        self.vol_window = 10
        self.vol_spread_mean = -0.24
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


def calc_time_to_expiry(day, ts):
    return (8 - day) / 250 - ts / 1_000_000 / 250


def calc_implied_vol(spot_price: float, strike_price: float, time_to_expiry: float, call_price: float) -> float:
    bs = BlackScholes(spot=spot_price, strike=strike_price, time_to_expiry=time_to_expiry)
    return bs.implied_volatility(call_price)


def calc_delta(spot_price: float, strike_price: float, time_to_expiry: float, implied_vol: float) -> float:
    bs = BlackScholes(spot=spot_price, strike=strike_price, time_to_expiry=time_to_expiry)
    return bs.delta(implied_vol)


def calc_vega(spot_price: float, strike_price: float, time_to_expiry: float, implied_vol: float) -> float:
    bs = BlackScholes(spot=spot_price, strike=strike_price, time_to_expiry=time_to_expiry)
    return bs.vega(implied_vol)


def calc_moneyness(spot_price: float, strike_price: float, time_to_expiry: float) -> float:
    return np.log(strike_price / spot_price) / np.sqrt(time_to_expiry)


def calc_base_iv(m_list: List[float], v_list: List[float]) -> float:
    params = np.polyfit(m_list, v_list, 2)  # ax^2 + bx + c --- params = [a, b, c]
    base_iv = float(params[2])
    if base_iv > 0:
        return base_iv
    else:
        return v_list[list(np.argsort(np.abs(np.array(m_list))))[0]]


def sort_call_by_moneyness(rock: VolcanicRock) -> List[int]:
    calls_moneyness = [call.moneyness for call in rock.call_options]
    sorted_idx = list(np.argsort(np.abs(np.array(calls_moneyness))))
    return sorted_idx


def set_rock_call_greeks(state: TradingState, rock: VolcanicRock) -> VolcanicRock:
    time_to_expiry = calc_time_to_expiry(day=3, ts=state.timestamp)
    for call in rock.call_options:
        intrinsic = max(rock.spot.fair_value - call.strike_price, 0)
        if call.fair_value < intrinsic:
            call.fair_value = intrinsic
        call.implied_vol = calc_implied_vol(rock.spot.fair_value, call.strike_price, time_to_expiry, call.fair_value)
        call.delta = calc_delta(rock.spot.fair_value, call.strike_price, time_to_expiry, call.implied_vol)
        call.vega = calc_vega(rock.spot.fair_value, call.strike_price, time_to_expiry, call.implied_vol)
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
        # rock_vol_spread_z_score = (vol_spread - rock.vol_spread_mean) / rock.vol_spread_std
        rock_vol_spread_z_score = vol_spread
        if rock_vol_spread_z_score > rock.vol_spread_z_score_thr and call.best_bid:  # short call
            call_size = min(abs(call.limit + call.position), call.best_bid_size, call.limit)
            if call_size > 0:
                orders[call.name].append(Order(call.name, round(call.best_bid), -round(call_size)))

        if rock_vol_spread_z_score < - rock.vol_spread_z_score_thr and call.best_ask:  # long call
            call_size = min(abs(call.limit - call.position), call.best_ask_size, call.limit)
            if call_size > 0:
                orders[call.name].append(Order(call.name, round(call.best_ask), round(call_size)))

    return orders


class Trader:
    def run(self, state: TradingState):
        conversions = 0
        previous_rock_prices = []
        rock_vol_spreads = []
        m_list = []
        v_list = []

        if state.traderData:
            previous_state = jsonpickle.decode(state.traderData)
            previous_rock_prices = previous_state.get('previous_rock_prices', [])
            rock_vol_spreads = previous_state.get('rock_vol_spreads', [])
            m_list = previous_state.get('m_list', [])
            v_list = previous_state.get('v_list', [])

        result = {}

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

            for call in rock.call_options:
                if call.vega > 1e-6:
                    m_list.append(call.moneyness)
                    v_list.append(call.implied_vol)

            m_list = m_list[-rock.vol_window * len(rock.call_options):]
            v_list = v_list[-rock.vol_window * len(rock.call_options):]

            if len(m_list) == rock.vol_window * len(rock.call_options) and len(v_list) == rock.vol_window * len(rock.call_options):
                base_iv = calc_base_iv(m_list, v_list)
                vol_spread = base_iv

                rock_vol_spreads.append(vol_spread)
                if len(rock_vol_spreads) > rock.vol_window:
                    rock_vol_spreads.pop(0)

                if len(rock_vol_spreads) == rock.vol_window:
                    vol_spread_z = (vol_spread - np.mean(rock_vol_spreads)) / np.std(rock_vol_spreads)
                    # vol_spread_z = (rock_vol_spreads[-1] - rock.vol_spread_mean) / rock.vol_spread_std

                    rock_orders: List[Order] = hedge_rock(rock)
                    result[rock.spot.name] = rock_orders

                    voucher_orders: Dict[str, List[Order]] = trade_rock_vouchers(rock, vol_spread_z)

                    for product_name, orders in voucher_orders.items():
                        result[product_name] = orders

            previous_rock_prices = rock_prices

        trader_data = jsonpickle.encode({
            'previous_rock_prices': previous_rock_prices,
            'rock_vol_spreads': rock_vol_spreads,
            'm_list': m_list,
            'v_list': v_list
        })
        return result, conversions, trader_data