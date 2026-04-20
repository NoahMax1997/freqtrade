# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame, Series
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

GREEN_COLOR = 1
RED_COLOR = -1
ELSE_COLOR = 0
UP_SMA = 1
DOWN_SMA = -1

UP_TREND = 1
DOWN_TREND = -1
ELSE_TREND = 0

class HeikenTrendTestStrategy(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    profit_rate = 2.5
    INTERFACE_VERSION = 3
    can_short: bool = True
    # Optimal timeframe for the strategy.
    # timeframe = "1m"
    # use_custom_stoploss = True
    use_custom_exit = True
    # 告诉 Freqtrade 我们还需要 1m 的数据
    # Can this strategy go short?
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {
    #     "0": 0.05
    # }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".

    # Trailing stoploss
    # trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Strategy parameters
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

    _full_dataframes = {}
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

    def kline_callback(self,dataframe:DataFrame):
        dataframe.loc[len(dataframe) - 1, 'trend_status'] = self.judge_trend_status(dataframe)

    def judge_trend_status(self,dataframe:DataFrame):
        try:
            """处理滚动窗口的单列数据"""
            if dataframe['up_ema'].iloc[-1] == UP_SMA \
                and dataframe['up_ema'].iloc[-2] == UP_SMA \
                and dataframe['up_ema'].iloc[-3] == UP_SMA \
                and dataframe['HA_Kline_Color'].iloc[-1] == GREEN_COLOR \
                and dataframe['HA_Kline_Color'].iloc[-2] == GREEN_COLOR \
                and dataframe['HA_Kline_Color'].iloc[-3] == GREEN_COLOR \
                and dataframe['HA_Kline_Color'].iloc[-4] == RED_COLOR \
                and dataframe['HA_Is_Bullish'].iloc[-1] == True \
                and dataframe['HA_Is_Bullish'].iloc[-2] == True \
                and dataframe['HA_Is_Bullish'].iloc[-3] == True:
                return UP_TREND
            elif dataframe['up_ema'].iloc[-1] == DOWN_SMA \
                and dataframe['up_ema'].iloc[-2] == DOWN_SMA \
                and dataframe['up_ema'].iloc[-3] == DOWN_SMA \
                and dataframe['HA_Kline_Color'].iloc[-1] == RED_COLOR \
                and dataframe['HA_Kline_Color'].iloc[-2] == RED_COLOR \
                and dataframe['HA_Kline_Color'].iloc[-3] == RED_COLOR \
                and dataframe['HA_Kline_Color'].iloc[-4] == GREEN_COLOR \
                and dataframe['HA_Is_Bullish'].iloc[-1] == False \
                and dataframe['HA_Is_Bullish'].iloc[-2] == False \
                and dataframe['HA_Is_Bullish'].iloc[-3] == False :
                return DOWN_TREND
            else:
                return ELSE_TREND
        except Exception as e:
            print(f"dataframe:{dataframe} columns:{dataframe.columns}")
    
    # def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
    #     """
    #     自定义止损逻辑: 返回相对于当前价格的止损百分比
    #     注意: 返回正值表示亏损百分比
    #     """
    #     try:
    #         dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #         last_five = dataframe.iloc[-5:].copy()
            
    #         # 检查交易方向
    #         is_long = trade.is_short == False
            
    #         if is_long:
    #             stop_loss_price = last_five['low'].min()  # 最近5根K线的最低点
    #             # 计算止损百分比 (正值)
    #             stoploss_pct = (current_rate - stop_loss_price) / current_rate
    #         else:  # 做空
    #             stop_loss_price = last_five['high'].max()  # 最近5根K线的最高点
    #             # 计算止损百分比 (正值)
    #             stoploss_pct = (stop_loss_price - current_rate) / current_rate

    #         # 存储在trade上，但不返回
    #         trade.stop_loss_pct = stoploss_pct
    #         trade.stop_loss = stop_loss_price
    #     except Exception as e:
    #         return 0.1

    def custom_exit_price(self, pair: str, trade: Trade, current_time: datetime, proposed_rate: float, current_profit: float, exit_tag: str, **kwargs):
           # 对于止损退出，使用精确的止损价格
        if "达到止盈目标" in exit_tag:
            ret = trade.get_custom_data("take_profit_price")
            return ret
        elif "达到止损目标" in exit_tag:
            ret = trade.get_custom_data("stop_loss_price")
            return ret
       
        return proposed_rate

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        """
        实现止盈是止损的两倍，使用开仓位置前5根K线（不包含开仓K线）
        """
        try:
            # 获取完整的数据帧
            dataframe = self._full_dataframes[pair]

            # 获取交易开仓时间
            open_date = trade.open_date
            # 找到开仓时间在数据帧中的位置
            candle_index = dataframe[dataframe['date'] <= open_date].index[-1]
            # 尝试使用索引获取当前K线（更可靠）
            try:
                # 首先尝试基于时间匹配找到当前K线
                matched_candles = dataframe[dataframe['date'] <= current_time]
                if not matched_candles.empty:
                    current_candle = matched_candles.iloc[-1]
                else:
                    # 如果找不到匹配的时间，使用最后一根K线
                    current_candle = dataframe.iloc[-1]
            except Exception as e:
                self.logger.error(f"获取当前K线时出错: {e}")
                return None
            
            # 获取开仓位置前5根K线（不包含开仓K线）
            start_index = max(0, candle_index - 5)  # 确保索引不为负
            end_index = max(0, candle_index - 1)    # 确保索引不为负，不包含开仓K线
            
            # 如果没有足够的历史K线数据，使用可用的最多K线数据
            if start_index <= end_index:
                entry_candles = dataframe.iloc[start_index:end_index + 1].copy()
            else:
                # 
                print(f"如果没有足够的历史数据，返回None（不触发止盈止损）")
                return None
            
            is_long = trade.is_short == False
            
            if is_long:
                # 如果有足够的历史K线
                if not entry_candles.empty:
                    # 计算止损价格
                    stop_loss_price = entry_candles['low'].min()

                    # 找到最低价格的索引位置
                    min_low_idx = entry_candles['low'].idxmin()
                    
                    # 获取该位置的时间戳
                    min_low_timestamp = entry_candles.loc[min_low_idx, 'date']
                
                    # 计算从入场价到止损价的距离
                    sl_distance = trade.open_rate - stop_loss_price
                    # 止盈价格 = 入场价格 + 2 * 止损距离
                    take_profit_price = trade.open_rate + self.profit_rate * sl_distance
                    # 如果价格达到或超过止盈目标
                    low_rate = current_candle['low']
                    high_rate = current_candle['high']
                    stoploss_pct = (trade.open_rate - stop_loss_price) / trade.open_rate
                    trade.set_custom_data("stop_loss_price",stop_loss_price)
                    trade.set_custom_data("stoploss_pct",stoploss_pct)
                    trade.set_custom_data("take_profit_price",take_profit_price)
                    trade.set_custom_data("takeprofit_pct",sl_distance/trade.open_rate)
                    if low_rate <= stop_loss_price and stop_loss_price <= high_rate :
                        return f"达到止损目标"
                        # return f"达到止损目标: min_low_timestamp:{min_low_timestamp} stop_loss_price:{stop_loss_price} stoploss_pct:{stoploss_pct} take_profit_price:{take_profit_price} current_time:{current_time}" 
                    elif low_rate <= take_profit_price and take_profit_price <= high_rate :
                        # return f"达到止盈目标: 止损的2倍收益 min_low_timestamp:{min_low_timestamp} stop_loss_price:{stop_loss_price} stoploss_pct:{stoploss_pct} take_profit_price:{take_profit_price} current_time:{current_time}"
                        return f"达到止盈目标"
                else:
                    print(f"如果有足够的历史K线 current_time:{current_time} trade.open_date:{trade.open_date} entry_candles:{entry_candles}")


            else:  # 做空
                # 如果有足够的历史K线
                if not entry_candles.empty:
                    # 计算止损价格
                    stop_loss_price = entry_candles['high'].max()
                    # 找到最高价格的索引位置
                    max_high_idx = entry_candles['high'].idxmax()
                    
                    # 获取该位置的时间戳
                    max_high_timestamp = entry_candles.loc[max_high_idx, 'date']
                
                    # 计算从入场价到止损价的距离
                    sl_distance = stop_loss_price - trade.open_rate
                    # 止盈价格 = 入场价格 - 2 * 止损距离
                    take_profit_price = trade.open_rate - self.profit_rate * sl_distance
                    # 如果价格达到或低于止盈目标
                    low_rate = current_candle['low']
                    high_rate = current_candle['high']
                    stoploss_pct = (stop_loss_price - trade.open_rate) / trade.open_rate
                    trade.set_custom_data("stop_loss_price",stop_loss_price)
                    trade.set_custom_data("stoploss_pct",stoploss_pct)
                    trade.set_custom_data("take_profit_price",take_profit_price)
                    trade.set_custom_data("takeprofit_pct",sl_distance/trade.open_rate)
                    if low_rate <= stop_loss_price and stop_loss_price <= high_rate :
                        return f"达到止损目标"
                        # return f"达到止损目标: max_high_timestamp:{max_high_timestamp} stop_loss_price:{stop_loss_price} stoploss_pct:{stoploss_pct} take_profit_price:{take_profit_price} current_time:{current_time}"
                    elif low_rate <= take_profit_price and take_profit_price <= high_rate :
                        return f"达到止盈目标"
                        # return f"达到止盈目标: 止损的2倍收益 max_high_timestamp:{max_high_timestamp} stop_loss_price:{stop_loss_price} stoploss_pct:{stoploss_pct} take_profit_price:{take_profit_price} current_time:{current_time}"
                    
                else:
                    date_start = dataframe["date"].head(1).values[0]  # 获取第一行的"date"值
                    date_end = dataframe["date"].tail(1).values[0] 
                    print(f"如果有足够的历史K线 current_time:{current_time} trade.open_date:{trade.open_date} len:{len(dataframe)} entry_candles:{entry_candles} date_start:{date_start} date_end:{date_end}")
        
        except Exception as e:
            print(f"trade.open_date:{trade.open_date} len:{len(dataframe)} error {str(e)}")
        return None  # 不退出交易

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
        """
            Adds several different TA indicators to the given DataFrame

            Performance Note: For the best performance be frugal on the number of indicators
            you are using. Let uncomment only the indicator you are using in your strategies
            or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
            :param dataframe: Dataframe with data from the exchange
            :param metadata: Additional information, like the currently traded pair
            :return: a Dataframe with all mandatory indicators for the strategies
        """
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe["ha_open"] = heikinashi["open"]
        dataframe["ha_close"] = heikinashi["close"]
        dataframe["ha_high"] = heikinashi["high"]
        dataframe["ha_low"] = heikinashi["low"]
        
        dataframe['sma7'] = ta.SMA(dataframe['close'], timeperiod=7)
        dataframe['sma14'] = ta.SMA(dataframe['close'], timeperiod=14)
        dataframe['ema7'] = ta.EMA(dataframe['close'], timeperiod=7)
        dataframe['ema14'] = ta.EMA(dataframe['close'], timeperiod=14)
        dataframe['HA_Is_Bullish'] = np.where(dataframe['ha_close'] > dataframe['ha_open'], True, False)     
        dataframe['HA_Kline_Color'] = np.where(dataframe['ha_close'] > dataframe['ha_open'].shift(1), GREEN_COLOR, RED_COLOR)
        dataframe['up_sma7'] = np.where(dataframe['close'] > dataframe['sma7'], UP_SMA, DOWN_SMA) 
        dataframe['up_ema7'] = np.where(dataframe['close'] > dataframe['ema7'], UP_SMA, DOWN_SMA)
        dataframe['up_sma14'] = np.where(dataframe['close'] > dataframe['sma14'], UP_SMA, DOWN_SMA)
        dataframe['up_ema14'] = np.where(dataframe['close'] > dataframe['ema14'], UP_SMA, DOWN_SMA)
        dataframe['up_ema'] = np.where(dataframe['close'] > dataframe['ema7'], UP_SMA, DOWN_SMA)
        # 获取当前活跃的交易（已开仓未平仓）
        dataframe['trend_status'] = ELSE_TREND
        for i in range(7,len(dataframe)):
            dataframe.loc[i,"judge"] =  self.judge_trend_status(dataframe.iloc[0:i+1])

        # dataframe.to_csv('dataframe.csv', index=False)
        self._full_dataframes[metadata['pair']] = dataframe
        return dataframe

    def judge_function(self,series:Series):
        if series.iloc[-5] == True and series.iloc[-4] == True and series.iloc[-3] == False and series.iloc[-2] == False and series.iloc[-1] == False:
            return -1
        elif series.iloc[-5] == False and series.iloc[-4] == False and series.iloc[-3] == True and series.iloc[-2] == True and series.iloc[-1] == True:
            return 1
        else:
            return 0

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict):
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        print(f"populate_entry_trend_len: {len(dataframe)} time:{time.time()}")
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        dataframe.loc[
            (
                (dataframe['judge'] == UP_TREND)
            ),
            'enter_long'] = 1
        dataframe.loc[
            (
                (dataframe['judge'] == DOWN_TREND)
            ),
            'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict):
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        # dataframe.loc[
        #     (
        #         (dataframe['judge'] == -1)
        #     ),
        #     'exit_long'] = 1
        # dataframe.loc[
        #     (
        #         (dataframe['judge'] == 1)
        #     ),
        #     'exit_short'] = 1
        return dataframe