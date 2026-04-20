# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import time
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame, Series
from typing import Dict, Optional, Union, Tuple
import logging

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

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志输出格式
    handlers=[logging.StreamHandler()]  # 输出到控制台和文件
)
# logging.FileHandler('user_data/logs/TradingViewStrategy.log')
logger = logging.getLogger(__name__)  # 获取日志记录器

class TradingViewStrategy(IStrategy):
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
    days = 5
    profit_rate = 3
    INTERFACE_VERSION = 3
    can_short: bool = True
    # Optimal timeframe for the strategy.
    timeframe = "1h"
    # use_custom_stoploss = True
    # use_custom_exit = True
    
    # 启用交易所上的止损单
    stoploss_on_exchange = True
    
    # 使用限价单类型作为止损
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": True
        # "stoploss_on_exchange_limit_ratio": 0.08  # 止损限价比例，仅对限价止损有效
    }
    # Optional order time in force.
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }
    
    # 增加配置项确保custom_exit_price被调用
    # use_exit_signal = True
    # ignore_roi_if_entry_signal = True

    _full_dataframes = {}
    def _calculate_tp_sl_prices(self, trade: Trade):
        """
        计算止盈止损价格
        tuple[float, float]
        """
        return None, None
        try:
            # 获取完整的数据帧
            pair = trade.pair
            dataframe = self._full_dataframes.get(pair)
            if dataframe is None:
                logger.error(f"未找到{pair}的数据帧")
                return None, None
            
            # 获取交易开仓时间
            open_date = trade.open_date
            if dataframe['date'].dt.tz is None:
                dataframe['date'] = dataframe['date'].dt.tz_localize('UTC') 
            # 找到开仓前一根k线在数据帧中的位置
            candles = dataframe[dataframe['date'] <= open_date][-4:-1]
            is_long = trade.is_short == False
            if is_long:
                # 计算做多的止损价格
                stop_loss_price = candles['low'].min()
                
                # 计算从入场价到止损价的距离
                sl_distance = trade.open_rate - stop_loss_price
                
                # 止盈价格 = 入场价格 + profit_rate * 止损距离
                take_profit_price = trade.open_rate + self.profit_rate * sl_distance
                
                # 计算止损百分比
                stoploss_pct = (trade.open_rate - stop_loss_price) / trade.open_rate
                
                # 计算止盈百分比
                takeprofit_pct = (take_profit_price - trade.open_rate) / trade.open_rate
                
                logger.info(f"设置做多止盈止损 - {pair}: 止损价格={stop_loss_price} 止损率:{stoploss_pct},  止盈价格={take_profit_price} 止盈率:{takeprofit_pct}")
                return take_profit_price, stop_loss_price
            else:
                # 计算做空的止损价格
                stop_loss_price = candles['high'].max()
                
                # 计算从入场价到止损价的距离
                sl_distance = stop_loss_price - trade.open_rate
                
                # 止盈价格 = 入场价格 - profit_rate * 止损距离
                take_profit_price = trade.open_rate - self.profit_rate * sl_distance
                
                # 计算止损百分比
                stoploss_pct = (stop_loss_price - trade.open_rate) / trade.open_rate

                # 计算止盈百分比
                takeprofit_pct = (trade.open_rate - take_profit_price) / trade.open_rate
                
                # 记录信息到交易对象
                logger.info(f"设置做空止盈止损 - {pair}: 止损价格={stop_loss_price} 止损率:{stoploss_pct},  止盈价格={take_profit_price} 止盈率:{takeprofit_pct}")
                return take_profit_price, stop_loss_price
        except Exception as e:
            error = traceback.format_exc()
            logger.error(f"计算止盈止损价格时出错: {e} {error}")
            return None, None
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, current_time: datetime, entry_tag: str | None, side: str, **kwargs):
        """
        确认交易入口
        """
        try:
            # 获取当前分钟数
            current_minute = current_time.minute
            
            # 检查是否在小时的前10分钟内 (0-9分钟)
            if current_minute <= 1:
                # logger.info(f"{pair}: 在小时开始后前1分钟内开仓 - 当前时间: {current_time}")
                return True
            else:
                # logger.info(f"{pair}: 不在开仓时间窗口内 - 当前时间: {current_time}, 分钟: {current_minute}")
                return False
        except Exception as e:
            logger.error(f"确认交易入口时出错: {e}")
            return False

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            "main_plot": {
                # 移除不需要的指标
            },
            "subplots": {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    "macd": {"color": "blue"},
                    "macdsignal": {"color": "orange"},
                    "macdhist": {"color": "gray", "type": "bar"}
                },
                "RSI": {
                    "rsi": {"color": "red"},
                    # 添加RSI的超买超卖线
                    "rsi_overbought": {"color": "red", "type": "line", "value": 70},
                    "rsi_oversold": {"color": "green", "type": "line", "value": 30}
                }
            }
        }

    def custom_exit_price(self, pair: str, trade: Trade, current_time: datetime, proposed_rate: float, current_profit: float, exit_tag: str, **kwargs):
        """
        自定义退出价格，从交易对象获取保存的价格信息
        """
        return proposed_rate
        logger.info(f"custom_exit_price被调用: {pair} - exit_tag:{exit_tag}, proposed_rate:{proposed_rate}, current_profit:{current_profit}")
        
        if "达到止盈目标" in exit_tag:
            ret = trade.get_custom_data("take_profit_price")
            logger.info(f"custom_exit_price: {trade.pair} - current_time:{current_time} proposed_rate:{proposed_rate} 止盈价格 {ret}")
            return ret
        elif "达到止损目标" in exit_tag:
            ret = trade.get_custom_data("stop_loss_price")
            logger.info(f"custom_exit_price: {trade.pair} - current_time:{current_time} proposed_rate:{proposed_rate} 止损价格 {ret}")
            return ret
       
        logger.info(f"custom_exit_price: exit_tag:{exit_tag} proposed_rate:{proposed_rate}")
        return proposed_rate

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        """
        利用原生Freqtrade的检查逻辑来触发止盈止损
        """
        return None
        take_profit_price = trade.get_custom_data("take_profit_price")
        stop_loss_price = trade.get_custom_data("stop_loss_price")
        if not take_profit_price or not stop_loss_price:    
            take_profit_price, stop_loss_price = self._calculate_tp_sl_prices(trade)
            if take_profit_price and stop_loss_price:
                trade.set_custom_data("take_profit_price", take_profit_price)
                trade.set_custom_data("stop_loss_price", stop_loss_price)
                msg = f"current_time:{current_time} pair:{pair} current_rate:{current_rate} current_profit:{current_profit} open_date:{trade.open_date} open_rate:{trade.open_rate} take_profit_price: {take_profit_price} stop_loss_price: {stop_loss_price}"
                logger.info(msg)
                self.dp.send_msg(f"止盈止损计算完毕:\n{msg}")
      # 如果没有设置止盈止损价格，则不触发退出
        if not take_profit_price or not stop_loss_price:
            return None
        
        is_long = not trade.is_short
        
        # 获取当前K线数据
        try:
            matched_candles = self._full_dataframes[pair][self._full_dataframes[pair]['date'] <= current_time]
            if not matched_candles.empty:
                current_candle = matched_candles.iloc[-1]
                # logger.info(f"找到当前K线: date={current_candle['date']}, open={current_candle['open']}, high={current_candle['high']}, low={current_candle['low']}, close={current_candle['close']}")
            else:
                # 如果找不到匹配的时间，使用最后一根K线
                current_candle = self._full_dataframes[pair].iloc[-1]
                # logger.info(f"使用最后一根K线: date={current_candle['date']}, open={current_candle['open']}, high={current_candle['high']}, low={current_candle['low']}, close={current_candle['close']}")
            
            current_low = current_candle['low']
            current_high = current_candle['high']
            
            # 记录价格比较
            # logger.info(f"{pair} 止盈价: {take_profit_price}, 止损价: {stop_loss_price}, 当前最高: {current_high}, 当前最低: {current_low}")
            
            # 根据交易方向检查止盈止损条件
            if is_long:
                if current_low <= stop_loss_price:
                    logger.info(f"触发做多止损: {trade.pair} - 当前最低价格 {current_low} <= 止损价格 {stop_loss_price}")
                    return "达到止损目标"
                # 做多: 当前价格 >= 止盈价格 或 当前价格 <= 止损价格
                if current_high >= take_profit_price:
                    logger.info(f"触发做多止盈: {trade.pair} - 当前最高价格 {current_high} >= 止盈价格 {take_profit_price}")
                    return "达到止盈目标"
            else:
                if current_high >= stop_loss_price:
                    logger.info(f"触发做空止损: {trade.pair} - 当前最高价格 {current_high} >= 止损价格 {stop_loss_price}")
                    return "达到止损目标"
                # 做空: 当前价格 <= 止盈价格 或 当前价格 >= 止损价格
                if current_low <= take_profit_price:
                    logger.info(f"触发做空止盈: {trade.pair} - 当前最低价格 {current_low} <= 止盈价格 {take_profit_price}")
                    return "达到止盈目标"
        except Exception as e:
            logger.error(f"检查止盈止损条件时出错: {e}, 详情: {traceback.format_exc()}")
            
        return None
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str, side: str,
                 **kwargs):
        """
        调整杠杆率
        """
        return 1
    
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
        dataframe["ema_20"] = ta.EMA(dataframe['close'], timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe['close'], timeperiod=50)
        dataframe["ema_20_up_cross_50"] = qtpylib.crossed_above(dataframe["ema_20"], dataframe["ema_50"])
        dataframe["ema_20_down_cross_50"] = qtpylib.crossed_below(dataframe["ema_20"], dataframe["ema_50"])
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict):
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        dataframe.loc[
            (
                (dataframe['ema_20_up_cross_50'] == True) 
            ),
            'enter_long'] = 1
        dataframe.loc[
            (
                (dataframe['ema_20_down_cross_50'] == True)
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
        # 多单平仓条件：MACD死叉
        dataframe.loc[
            (
                (dataframe['ema_20_down_cross_50'] == True)
            ),
            'exit_long'] = 1
            
        # 空单平仓条件：MACD金叉
        dataframe.loc[
            (
                (dataframe['ema_20_up_cross_50'] == True)
            ),
            'exit_short'] = 1
        return dataframe
        