# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
FMZ-style market-neutral strategy with fixed notional allocation.

Unlike LongShortTopKStrategy which allocates proportionally from cash,
this strategy assigns a fixed USDT notional value to each position,
allowing leveraged long-short portfolios.
"""

import copy
import numpy as np
import pandas as pd

from typing import List

from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.log import get_module_logger

logger = get_module_logger("FMZNeutralStrategy")


class FMZNeutralStrategy(BaseSignalStrategy):
    """
    FMZ-style market-neutral strategy — flip-only with fixed notional.

    Reproduces FMZ's Test() function exactly:
    - Only trades when a symbol's signal direction CONFLICTS with current position
    - buy_symbols (top scores) + currently short/flat → flip to long
    - sell_symbols (bottom scores) + currently long/flat → flip to short
    - Middle-ranked symbols → hold existing position (no action)
    - Positions accumulate beyond topk as symbols drift to the middle zone

    Parameters
    ----------
    topk_long : int
        Number of instruments to go long (e.g. 40).
    topk_short : int
        Number of instruments to go short (e.g. 40).
    value_per_position : float
        Fixed USDT notional per position (e.g. 300.0).
    """

    def __init__(
        self,
        *,
        topk_long: int = 40,
        topk_short: int = 40,
        value_per_position: float = 300.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.topk_long = topk_long
        self.topk_short = topk_short
        self.value_per_position = value_per_position

    def generate_trade_decision(self, execute_result=None):
        # --- 1. Get signal with shift=1 (previous bar's signal) ---
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)

        if pred_score is None:
            return TradeDecisionWO([], self)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]

        pred_score = pred_score.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        if len(pred_score) == 0:
            return TradeDecisionWO([], self)

        # --- 2. Select long/short sets ---
        # Highest scores → long (signal already negated in handler)
        # Lowest scores → short
        long_set = set(pred_score.index[: self.topk_long])
        short_set = set(pred_score.index[-self.topk_short :])
        # Remove overlap
        short_set -= long_set

        # --- 3. FMZ flip-only logic ---
        # Only trade when signal direction conflicts with current position:
        #   buy_symbols + (short or flat) → flip to +value/price
        #   sell_symbols + (long or flat) → flip to -value/price
        #   middle zone or same direction → hold, no action
        current_stock_list = set(self.trade_position.get_stock_list())
        valid_instruments = set(pred_score.index)

        sell_orders: List[Order] = []
        buy_orders: List[Order] = []

        # --- 3a. Close positions with missing factor values ---
        for code in current_stock_list:
            if code in valid_instruments:
                continue
            current_amt = self.trade_position.get_stock_amount(code)
            if current_amt == 0:
                continue
            direction = OrderDir.SELL if current_amt > 0 else OrderDir.BUY
            price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time,
                end_time=trade_end_time, direction=direction,
            )
            if price is None or not np.isfinite(price) or price <= 0:
                continue
            order = Order(
                stock_id=code, amount=abs(current_amt),
                start_time=trade_start_time, end_time=trade_end_time,
                direction=direction,
            )
            if direction == OrderDir.SELL:
                sell_orders.append(order)
            else:
                buy_orders.append(order)

        # --- 3b. Flip logic for instruments with valid factors ---
        for code in short_set:
            current_amt = (
                self.trade_position.get_stock_amount(code)
                if code in current_stock_list else 0.0
            )
            if current_amt < 0:
                # Already short → hold, no action (FMZ: amount < 0 skips sell)
                continue
            price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time,
                end_time=trade_end_time, direction=OrderDir.SELL,
            )
            if price is None or not np.isfinite(price) or price <= 0:
                continue
            # FMZ: Sell(price, value/price + amount)
            # target = -value/price, sell_amount = value/price + current_amt
            sell_amount = self.value_per_position / float(price) + current_amt
            sell_orders.append(Order(
                stock_id=code, amount=sell_amount,
                start_time=trade_start_time, end_time=trade_end_time,
                direction=OrderDir.SELL,
            ))

        for code in long_set:
            current_amt = (
                self.trade_position.get_stock_amount(code)
                if code in current_stock_list else 0.0
            )
            if current_amt > 0:
                # Already long → hold, no action (FMZ: amount > 0 skips buy)
                continue
            price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time,
                end_time=trade_end_time, direction=OrderDir.BUY,
            )
            if price is None or not np.isfinite(price) or price <= 0:
                continue
            # FMZ: Buy(price, value/price - amount)
            # target = +value/price, buy_amount = value/price - current_amt
            buy_amount = self.value_per_position / float(price) - current_amt
            buy_orders.append(Order(
                stock_id=code, amount=buy_amount,
                start_time=trade_start_time, end_time=trade_end_time,
                direction=OrderDir.BUY,
            ))

        # SELL first → generates cash for BUY orders (FMZ has no cash constraint)
        # Middle-zone symbols: NO ACTION — hold existing position
        return TradeDecisionWO(sell_orders + buy_orders, self)
