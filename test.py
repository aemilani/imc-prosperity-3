import jsonpickle
import numpy as np
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any
from math import log, sqrt
from statistics import NormalDist


class Product:
    COCONUT = "COCONUT"
    COCONUT_COUPON = "COCONUT_COUPON"


PARAMS = {
    Product.COCONUT_COUPON: {
        "mean_volatility": 0.15959997370608378,
        "threshold": 0.00163,
        "strike": 10000,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 6,
        "zscore_threshold": 21,
    },
}


class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.COCONUT: 300,
            Product.COCONUT_COUPON: 600,
        }

    def get_coconut_coupon_mid_price(
        self, coconut_coupon_order_depth: OrderDepth, traderData: Dict[str, Any]
    ):
        if (
            len(coconut_coupon_order_depth.buy_orders) > 0
            and len(coconut_coupon_order_depth.sell_orders) > 0
        ):
            best_bid = max(coconut_coupon_order_depth.buy_orders.keys())
            best_ask = min(coconut_coupon_order_depth.sell_orders.keys())
            traderData["prev_coupon_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_coupon_price"]

    def coconut_hedge_orders(
        self,
        coconut_order_depth: OrderDepth,
        coconut_coupon_orders: List[Order],
        coconut_position: int,
        coconut_coupon_position: int,
        delta: float,
    ) -> list[Order] | None:
        if coconut_coupon_orders is None or len(coconut_coupon_orders) == 0:
            coconut_coupon_position_after_trade = coconut_coupon_position
        else:
            coconut_coupon_position_after_trade = coconut_coupon_position + sum(
                order.quantity for order in coconut_coupon_orders
            )

        target_coconut_position = -delta * coconut_coupon_position_after_trade

        if target_coconut_position == coconut_position:
            return None

        target_coconut_quantity = target_coconut_position - coconut_position

        orders: List[Order] = []
        if target_coconut_quantity > 0:
            # Buy COCONUT
            best_ask = min(coconut_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_coconut_quantity),
                self.LIMIT[Product.COCONUT] - coconut_position,
            )
            if quantity > 0:
                orders.append(Order(Product.COCONUT, best_ask, round(quantity)))

        elif target_coconut_quantity < 0:
            # Sell COCONUT
            best_bid = max(coconut_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_coconut_quantity),
                self.LIMIT[Product.COCONUT] + coconut_position,
            )
            if quantity > 0:
                orders.append(Order(Product.COCONUT, best_bid, -round(quantity)))

        return orders

    def coconut_coupon_orders(
        self,
        coconut_coupon_order_depth: OrderDepth,
        coconut_coupon_position: int,
        traderData: Dict[str, Any],
        volatility: float,
    ) -> tuple[None, None] | tuple[list[Order], list[Any]] | tuple[list[Order], list[Order]]:
        traderData["past_coupon_vol"].append(volatility)
        if (
            len(traderData["past_coupon_vol"])
            < self.params[Product.COCONUT_COUPON]["std_window"]
        ):
            return None, None

        if (
            len(traderData["past_coupon_vol"])
            > self.params[Product.COCONUT_COUPON]["std_window"]
        ):
            traderData["past_coupon_vol"].pop(0)

        vol_z_score = (
            volatility - self.params[Product.COCONUT_COUPON]["mean_volatility"]
        ) / np.std(traderData["past_coupon_vol"])
        if vol_z_score >= self.params[Product.COCONUT_COUPON]["zscore_threshold"]:
            if coconut_coupon_position != -self.LIMIT[Product.COCONUT_COUPON]:
                target_coconut_coupon_position = -self.LIMIT[Product.COCONUT_COUPON]
                if len(coconut_coupon_order_depth.buy_orders) > 0:
                    best_bid = max(coconut_coupon_order_depth.buy_orders.keys())
                    target_quantity = abs(
                        target_coconut_coupon_position - coconut_coupon_position
                    )
                    quantity = min(
                        target_quantity,
                        abs(coconut_coupon_order_depth.buy_orders[best_bid]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.COCONUT_COUPON, best_bid, -quantity)], []
                    else:
                        return [Order(Product.COCONUT_COUPON, best_bid, -quantity)], [
                            Order(Product.COCONUT_COUPON, best_bid, -quote_quantity)
                        ]

        elif vol_z_score <= -self.params[Product.COCONUT_COUPON]["zscore_threshold"]:
            if coconut_coupon_position != self.LIMIT[Product.COCONUT_COUPON]:
                target_coconut_coupon_position = self.LIMIT[Product.COCONUT_COUPON]
                if len(coconut_coupon_order_depth.sell_orders) > 0:
                    best_ask = min(coconut_coupon_order_depth.sell_orders.keys())
                    target_quantity = abs(
                        target_coconut_coupon_position - coconut_coupon_position
                    )
                    quantity = min(
                        target_quantity,
                        abs(coconut_coupon_order_depth.sell_orders[best_ask]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.COCONUT_COUPON, best_ask, quantity)], []
                    else:
                        return [Order(Product.COCONUT_COUPON, best_ask, quantity)], [
                            Order(Product.COCONUT_COUPON, best_ask, quote_quantity)
                        ]

        return None, None

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        if Product.COCONUT_COUPON not in traderObject:
            traderObject[Product.COCONUT_COUPON] = {
                "prev_coupon_price": 0,
                "past_coupon_vol": [],
            }

        if (
            Product.COCONUT_COUPON in self.params
            and Product.COCONUT_COUPON in state.order_depths
        ):
            coconut_coupon_position = (
                state.position[Product.COCONUT_COUPON]
                if Product.COCONUT_COUPON in state.position
                else 0
            )

            coconut_position = (
                state.position[Product.COCONUT]
                if Product.COCONUT in state.position
                else 0
            )

            coconut_order_depth = state.order_depths[Product.COCONUT]
            coconut_coupon_order_depth = state.order_depths[Product.COCONUT_COUPON]
            coconut_mid_price = (
                min(coconut_order_depth.buy_orders.keys())
                + max(coconut_order_depth.sell_orders.keys())
            ) / 2
            coconut_coupon_mid_price = self.get_coconut_coupon_mid_price(
                coconut_coupon_order_depth, traderObject[Product.COCONUT_COUPON]
            )
            tte = (
                    self.params[Product.COCONUT_COUPON]["starting_time_to_expiry"]
                    - state.timestamp / 1000000 / 250
            )
            volatility = BlackScholes.implied_volatility(
                coconut_coupon_mid_price,
                coconut_mid_price,
                self.params[Product.COCONUT_COUPON]["strike"],
                tte,
            )
            delta = BlackScholes.delta(
                coconut_mid_price,
                self.params[Product.COCONUT_COUPON]["strike"],
                tte,
                volatility,
            )

            coconut_coupon_take_orders, coconut_coupon_make_orders = (
                self.coconut_coupon_orders(
                    state.order_depths[Product.COCONUT_COUPON],
                    coconut_coupon_position,
                    traderObject[Product.COCONUT_COUPON],
                    volatility,
                )
            )

            coconut_orders = self.coconut_hedge_orders(
                state.order_depths[Product.COCONUT],
                state.order_depths[Product.COCONUT_COUPON],
                coconut_coupon_take_orders,
                coconut_position,
                coconut_coupon_position,
                delta,
            )

            if coconut_coupon_take_orders is not None or coconut_coupon_make_orders is not None:
                result[Product.COCONUT_COUPON] = (
                    coconut_coupon_take_orders + coconut_coupon_make_orders
                )
                # print(f"COCONUT_COUPON: {result[Product.COCONUT_COUPON]}")

            if coconut_orders is not None:
                result[Product.COCONUT] = coconut_orders
                # print(f"COCONUT: {result[Product.COCONUT]}")

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
