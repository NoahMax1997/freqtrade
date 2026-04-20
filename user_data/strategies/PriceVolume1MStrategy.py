# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class PriceVolume1MStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short: bool = True
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 30

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False
    }

    # Optional order time in force.
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }
    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            "main_plot": {
                "tema": {},
                "sar": {"color": "white"},
            },
            "subplots": {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    "macd": {"color": "blue"},
                    "macdsignal": {"color": "orange"},
                },
                "RSI": {
                    "rsi": {"color": "red"},
                }
            }
        }

    def is_order_active(self, pair: str):
        """
        检查是否有未平仓订单
        """
        # 可以通过自定义的代码来查询当前交易对是否有未平仓订单
        # 例如，通过 `self.dp` 获取当前持仓信息
        # dp 是 `freqtrade` 的数据提供者，包含当前的订单信息
        open_trades = self.dp.trades
        for trade in open_trades:
            if trade.pair == pair and trade.is_open:
                return True
        return False

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict):
        # 计算连续5根K线的涨跌幅
        dataframe['price_change'] = dataframe['close'].pct_change(periods=6)

        # 计算5分钟成交量
        dataframe["5m_volume"] = dataframe["volume"].rolling(window=5).sum()
        windows = 30
        mean_vol = dataframe['5m_volume'].rolling(window=windows).mean()
        std_vol = dataframe['5m_volume'].rolling(window=windows).std()
        dataframe["z_score"] = (dataframe['5m_volume'] - mean_vol) / std_vol

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict):
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe.loc[
            (
                (dataframe['price_change'] > 0.01) &  # 价格上涨
                (dataframe['z_score'] > 2)  # Z-score大于1
            ),
            'enter_long'] = 1
        dataframe.loc[
            (
                (dataframe['price_change'] < -0.01) &  # 价格下跌
                (dataframe['z_score'] > 2)  # Z-score小于-1
            ),
            'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict):
        return dataframe