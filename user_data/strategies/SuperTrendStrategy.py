# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
"""
SuperTrend Strategy + 更优价格重入逻辑（优于止损位）
基于 TradingView Pine Script 的 SuperTrend 策略转换

策略逻辑:
1. 趋势反转时，重置所有标记，允许重新寻找入场机会
2. 记录第一次满足入场条件时的价格
3. 开仓规则（同一趋势内）：
   - direction < 0 (上涨趋势) 且 supertrend > supertrend[2] 且 MACD 多头
     * 第一次满足条件时可以开多
     * 或者：已平仓后，价格低于第一次入场的止损位（-5%）时，可以重新开多
   - direction > 0 (下跌趋势) 且 supertrend < supertrend[3] 且 MACD 空头
     * 第一次满足条件时可以开空
     * 或者：已平仓后，价格高于第一次入场的止损位（+5%）时，可以重新开空
4. 平仓条件：
   - 趋势反转平仓
   - RSI 超买超卖背离平仓
   - ROI 止盈（50%，10% 现货 × 5倍杠杆）
5. 风控：
   - 止损：-25%（5% 现货 × 5倍杠杆）
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional

from datetime import datetime
from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    Trade,
    stoploss_from_open,
)

import talib.abstract as ta
import pandas_ta as pta


class SuperTrendStrategy(IStrategy):
    """
    SuperTrend 趋势跟踪策略
    使用 SuperTrend 指标判断趋势方向并进行交易
    """

    INTERFACE_VERSION = 3

    # 允许做空
    can_short: bool = True
    max_profit_rate = DecimalParameter(0.1, 0.22, default=0.10, space="sell", optimize=True)
    max_stop_loss_rate = DecimalParameter(0.1, 0.15, default=0.05, space="sell", optimize=True)
    max_profit_tailing = DecimalParameter(0.02, 0.06, default=0.05, space="sell", optimize=True, load=True)

    # 杠杆倍数
    leverage_num = 5  # 5倍杠杆
    # 直接止盈 - 达到目标立即全部平仓
    # 格式: {"分钟数": 收益率}
    minimal_roi = {
        "0": 1*max_profit_rate.value * leverage_num,    # 达到 75% (15% × 5倍杠杆) 直接全部平仓
    }

    # 关闭内置移动止损，使用自定义分阶段止损
    trailing_stop = False

    # 初始止损（会被 custom_stoploss 覆盖）
    stoploss = -1*max_stop_loss_rate.value * leverage_num

    # 启用自定义止损
    use_custom_stoploss = False
    # 时间周期

    # 只在新K线时处理
    process_only_new_candles = True

    # 使用退出信号
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # 可优化参数 - TradingView 默认: factor=5, atrPeriod=10
    st_factor = DecimalParameter(1.0, 10.0, default=5.0, space="buy", optimize=True)
    st_period = IntParameter(5, 20, default=10, space="buy", optimize=True)
    # st_factor = 9
    # st_period = 17

    # 策略启动所需的K线数量
    startup_candle_count: int = 50

    # 订单类型
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,  # 关闭交易所止损，由 Bot 本地监控
    }

    plot_config = {
        "main_plot": {
            "supertrend": {"color": "blue"},
        },
        "subplots": {
            "Direction": {
                "direction": {"color": "green"},
            },
        },
    }

    count = 0

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # 为每个交易对存储状态（实盘兼容）
        self.pair_state = {}  # 存储每个交易对的状态

    def get_pair_state(self, pair: str):
        """获取交易对的状态，不存在则初始化"""
        if pair not in self.pair_state:
            self.pair_state[pair] = {
                'long_signal_triggered': False,
                'short_signal_triggered': False,
                'first_long_entry_price': None,
                'first_short_entry_price': None,
            }
        return self.pair_state[pair]

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool, **kwargs):
        """
        从当前价格的回撤止损：
        - 利润 < 50%（10% × 5倍杠杆）：从当前价格回撤 10% 止损
        - 利润 >= 50%：从当前价格回撤 5% 止损

        返回值：相对于当前价格的止损百分比（负数）
        """
        lev = self.leverage_num

        if current_profit >= (self.max_profit_rate.value - self.max_profit_tailing.value)* lev:  # 利润 >= 50%（10% × 5倍杠杆）
            # 从当前价格回撤 5% 止损
            return -1*self.max_profit_tailing.value*0.5*lev

        # else:
        #     # 利润 < 50%：从当前价格回撤 10% 止损
        #     return -1*self.max_stop_loss_rate.value*lev

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag,
                 side: str, **kwargs):
        """
        设置杠杆倍数
        """
        return self.leverage_num

    def check_position(self, pair: str, is_long: bool) -> bool:
        """检查是否有持仓（实盘安全）"""
        if hasattr(self, 'wallets'):  # 实盘模式
            # 从实际交易中检查持仓
            from freqtrade.persistence import Trade
            trades = Trade.get_trades_proxy(pair=pair, is_open=True)
            for trade in trades:
                if (is_long and not trade.is_short) or (not is_long and trade.is_short):
                    return True
        return False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict):
        """
        完全复制 Pine Script 逻辑：SuperTrend + MACD + 趋势反转计数
        """
        # ========== SuperTrend 指标 ==========
        st = pta.supertrend(
            dataframe['high'],
            dataframe['low'],
            dataframe['close'],
            length=int(self.st_period.value),
            multiplier=float(self.st_factor.value)
        )

        length = int(self.st_period.value)
        mult = float(self.st_factor.value)
        st_col = f'SUPERT_{length}_{mult}'
        std_col = f'SUPERTd_{length}_{mult}'

        dataframe['supertrend'] = st[st_col]
        dataframe['direction'] = -1 * st[std_col]

        # 前几根K线的 supertrend 值
        dataframe['supertrend_2'] = dataframe['supertrend'].shift(2)
        dataframe['supertrend_3'] = dataframe['supertrend'].shift(3)

        # 前一根K线的方向
        dataframe['prev_direction'] = dataframe['direction'].shift(1)
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe["rsi_prev"] = dataframe['rsi'].shift(1)

        # 前一根K线的 high 和 low（用于 RSI 背离检测）
        dataframe['high_prev'] = dataframe['high'].shift(1)
        dataframe['low_prev'] = dataframe['low'].shift(1)

        # ========== MACD 指标 ==========
        macd_line, signal_line, hist_line = ta.MACD(
            dataframe['close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        dataframe['macd'] = macd_line
        dataframe['macdsignal'] = signal_line
        dataframe['macdhist'] = hist_line

        # MACD 金叉死叉判断
        dataframe['macd_golden_cross'] = (
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1))
        )
        dataframe['macd_death_cross'] = (
            (dataframe['macd'] < dataframe['macdsignal']) &
            (dataframe['macd'].shift(1) >= dataframe['macdsignal'].shift(1))
        )

        # MACD 趋势判断
        dataframe['macd_bullish'] = dataframe['macd'] > dataframe['macdsignal']
        dataframe['macd_bearish'] = dataframe['macd'] < dataframe['macdsignal']

        # ========== SuperTrend 趋势反转检测 ==========
        dataframe['trend_changed_to_up'] = (
            (dataframe['direction'] < 0) & (dataframe['prev_direction'] > 0)
        )
        dataframe['trend_changed_to_down'] = (
            (dataframe['direction'] > 0) & (dataframe['prev_direction'] < 0)
        )

        # 初始化入场和出场信号列
        dataframe['enter_long'] = 0
        dataframe['exit_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['exit_short'] = 0
        dataframe['rsi_exit_long'] = 0
        dataframe['rsi_exit_short'] = 0

        # 获取当前交易对的状态（实盘兼容）
        pair = metadata.get('pair', 'unknown')
        state = self.get_pair_state(pair)

        # 使用持久化状态变量（从上次的状态继续）
        long_signal_triggered = state['long_signal_triggered']
        short_signal_triggered = state['short_signal_triggered']
        first_long_entry_price = state['first_long_entry_price']
        first_short_entry_price = state['first_short_entry_price']

        # 实盘安全：从实际交易查询持仓状态，不依赖本地变量
        has_long_position = self.check_position(pair, is_long=True)
        has_short_position = self.check_position(pair, is_long=False)

        for i in range(len(dataframe)):
            # 趋势反转为上涨 - 允许开多仓，重置记录价格
            if dataframe['trend_changed_to_up'].iloc[i]:
                long_signal_triggered = True  # 允许开多仓
                short_signal_triggered = False  # 禁止开空仓
                first_long_entry_price = None  # 重置第一次入场价格

            # 趋势反转为下跌 - 允许开空仓，重置记录价格
            if dataframe['trend_changed_to_down'].iloc[i]:
                short_signal_triggered = True  # 允许开空仓
                long_signal_triggered = False  # 禁止开多仓
                first_short_entry_price = None  # 重置第一次入场价格

            # 检测做多条件（第一次或已平仓后更优价格重入）
            if dataframe['direction'].iloc[i] < 0:
                if dataframe['supertrend'].iloc[i] > dataframe['supertrend_2'].iloc[i]:
                    # 记录第一次满足条件的价格
                    if first_long_entry_price is None and dataframe['macd_bullish'].iloc[i]:
                        first_long_entry_price = dataframe['close'].iloc[i]

                    # 判断是否可以开多（必须无持仓）
                    can_open_long = False
                    if dataframe['macd_bullish'].iloc[i] and not has_long_position:
                        if long_signal_triggered:
                            # 第一次入场
                            can_open_long = True
                        elif first_long_entry_price is not None and dataframe['close'].iloc[i] < first_long_entry_price * (1 - self.max_stop_loss_rate.value):
                            # 已平仓，价格低于止损位（-5%），可以重新开仓
                            can_open_long = True

                    if can_open_long:
                        dataframe.loc[dataframe.index[i], 'enter_long'] = 1
                        long_signal_triggered = False  # 标记已开仓
                else:
                    if has_short_position:
                        dataframe.loc[dataframe.index[i], 'exit_short'] = 1

            if dataframe['direction'].iloc[i] > 0:
                if dataframe['supertrend'].iloc[i] < dataframe['supertrend_3'].iloc[i]:
                    # 记录第一次满足条件的价格
                    if first_short_entry_price is None and dataframe['macd_bearish'].iloc[i]:
                        first_short_entry_price = dataframe['close'].iloc[i]

                    # 判断是否可以开空（必须无持仓）
                    can_open_short = False
                    if dataframe['macd_bearish'].iloc[i] and not has_short_position:
                        if short_signal_triggered:
                            # 第一次入场
                            can_open_short = True
                        elif first_short_entry_price is not None and dataframe['close'].iloc[i] > first_short_entry_price * (1 + self.max_stop_loss_rate.value):
                            # 已平仓，价格高于止损位（+5%），可以重新开仓
                            can_open_short = True

                    if can_open_short:
                        dataframe.loc[dataframe.index[i], 'enter_short'] = 1
                        short_signal_triggered = False  # 标记已开仓
                else:
                    if has_long_position:
                        dataframe.loc[dataframe.index[i], 'exit_long'] = 1

            # ========== RSI 超买超卖背离平仓 ==========
            # 平多仓：RSI > 70 连续两根 + 价格背离
            if i > self.startup_candle_count:  # 确保有前一根K线可以比较
                rsi = dataframe['rsi'].iloc[i]
                rsi_prev = dataframe['rsi_prev'].iloc[i]
                high = dataframe['high'].iloc[i]
                high_prev = dataframe['high_prev'].iloc[i]
                low = dataframe['low'].iloc[i]
                low_prev = dataframe['low_prev'].iloc[i]

                # 平多：RSI > 70 连续两根 + 背离
                if has_long_position and rsi > 70 and rsi_prev > 70:
                    # (RSI 下降但价格上升) 或 (RSI 上升但价格下降)
                    if (rsi < rsi_prev and high > high_prev) or (rsi > rsi_prev and high < high_prev):
                        dataframe.loc[dataframe.index[i], 'rsi_exit_long'] = 1

                # 平空：RSI < 30 连续两根 + 背离
                if has_short_position and rsi < 30 and rsi_prev < 30:
                    # (RSI 下降但价格上升) 或 (RSI 上升但价格下降)
                    if (rsi < rsi_prev and low > low_prev) or (rsi > rsi_prev and low < low_prev):
                        dataframe.loc[dataframe.index[i], 'rsi_exit_short'] = 1

        # 保存状态到实例变量（实盘兼容）
        state['long_signal_triggered'] = long_signal_triggered
        state['short_signal_triggered'] = short_signal_triggered
        state['first_long_entry_price'] = first_long_entry_price
        state['first_short_entry_price'] = first_short_entry_price
        # 注意：has_long/short_position 从实际交易查询，不需要保存

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict):
        """
        入场信号 - 第一次入场或已平仓后优于止损位重入

        开多：
        1. 无持仓 + supertrend > supertrend[2] + MACD 多头
        2. 第一次满足条件时开仓
        3. 或已平仓后，价格 < 第一次入场价格 × 0.95（止损位）时重新开仓

        开空：
        1. 无持仓 + supertrend < supertrend[3] + MACD 空头
        2. 第一次满足条件时开仓
        3. 或已平仓后，价格 > 第一次入场价格 × 1.05（止损位）时重新开仓

        注：止损率 5%（现货）对应 25%（5倍杠杆）
        """
        # 做多信号：首次满足 SuperTrend 条件 + MACD 多头
        dataframe.loc[
            (
                (dataframe['enter_long'] == 1)
            ),
            'enter_long'
        ] = 1

        # 做空信号：首次满足 SuperTrend 条件 + MACD 空头
        dataframe.loc[
            (
                (dataframe['enter_short'] == 1)
            ),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict):
        """
        出场信号 - 在 populate_indicators 的 for 循环中已经设置

        1. 趋势反转平仓
        2. RSI 超买超卖背离平仓
        """
        # 合并所有平多信号（趋势反转平仓 OR RSI 背离平仓）
        dataframe.loc[
            (
                (dataframe['exit_long'] == 1) |
                (dataframe['rsi_exit_long'] == 1)
            ),
            'exit_long'
        ] = 1

        # 合并所有平空信号（趋势反转平仓 OR RSI 背离平仓）
        dataframe.loc[
            (
                (dataframe['exit_short'] == 1) |
                (dataframe['rsi_exit_short'] == 1)
            ),
            'exit_short'
        ] = 1

        return dataframe