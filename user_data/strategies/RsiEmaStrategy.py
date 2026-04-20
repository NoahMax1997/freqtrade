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
# logging.FileHandler('user_data/logs/RsiEmaStrategy.log')
logger = logging.getLogger(__name__)  # 获取日志记录器

class RsiEmaStrategy(IStrategy):
    """
    RSI DCA Strategy - 基于RSI信号和EMA交叉的交易策略
    
    开仓条件：
    - 做多：EMA7上穿EMA14 且 前7根K线中有RSI long信号（连续4根K线RSI<30且回穿到30以上）
    - 做空：EMA7下穿EMA14 且 前7根K线中有RSI short信号（连续4根K线RSI>70且回穿到70以下）
    
    退出条件：
    - RSI信号逆转退出
    
    时间框架：1小时
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    profit_rate = 4
    INTERFACE_VERSION = 3
    can_short: bool = True
    # Optimal timeframe for the strategy.
    timeframe = "1h"
    # use_custom_stoploss = True
    use_custom_exit = True
    
    # 启用交易所上的止损单
    stoploss_on_exchange = True
    
    # 使用限价单类型作为止损
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True
        # "stoploss_on_exchange_limit_ratio": 0.08  # 止损限价比例，仅对限价止损有效
    }
    # Optional order time in force.
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    _full_dataframes = {}
    def _calculate_tp_sl_prices(self, trade: Trade):
        """
        计算止盈止损价格
        tuple[float, float]
        """
        try:
            # 获取完整的数据帧
            pair = trade.pair
            dataframe = self._full_dataframes.get(pair)
            if dataframe is None:
                logger.error(f"未找到{pair}的数据帧")
                return None, None
            
            # 获取交易开仓时间
            open_date = trade.open_date
            # 找到开仓时间在数据帧中的位置
            candle_index = dataframe[dataframe['date'] <= open_date].index[-2]
            is_long = trade.is_short == False
            
            if is_long:
                # 计算做多的止损价格
                stop_loss_price = dataframe.iloc[candle_index]['low_price']
                
                # 计算从入场价到止损价的距离
                sl_distance = trade.open_rate - stop_loss_price
                
                # 止盈价格 = 入场价格 + profit_rate * 止损距离
                take_profit_price = trade.open_rate + self.profit_rate * sl_distance
                
                # 计算止损百分比
                stoploss_pct = (trade.open_rate - stop_loss_price) / trade.open_rate
                
                # 计算止盈百分比
                takeprofit_pct = (take_profit_price - trade.open_rate) / trade.open_rate
                
                # logger.info(f"设置做多止盈止损 - {pair}: 止损价格={stop_loss_price} 止损率:{stoploss_pct},  止盈价格={take_profit_price} 止盈率:{takeprofit_pct}")
                return take_profit_price, stop_loss_price
            else:
                # 计算做空的止损价格
                stop_loss_price = dataframe.iloc[candle_index]['high_price']
                
                # 计算从入场价到止损价的距离
                sl_distance = stop_loss_price - trade.open_rate
                
                # 止盈价格 = 入场价格 - profit_rate * 止损距离
                take_profit_price = trade.open_rate - self.profit_rate * sl_distance
                
                # 计算止损百分比
                stoploss_pct = (stop_loss_price - trade.open_rate) / trade.open_rate

                # 计算止盈百分比
                takeprofit_pct = (trade.open_rate - take_profit_price) / trade.open_rate
                
                # 记录信息到交易对象
                # logger.info(f"设置做空止盈止损 - {pair}: 止损价格={stop_loss_price} 止损率:{stoploss_pct},  止盈价格={take_profit_price} 止盈率:{takeprofit_pct}")
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
            if current_minute <= 10:
                # logger.info(f"{pair}: 在小时开始后前10分钟内开仓 - 当前时间: {current_time}")
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
            "subplots": {
                # RSI子图，包含RSI指标和关键水平线
                "RSI": {
                    "rsi": {"color": "blue"},
                    "rsi_sma_14": {"color": "orange"},
                    # 添加RSI的关键水平线
                    "rsi_overbought": {"color": "red", "type": "line"},
                    "rsi_oversold": {"color": "green", "type": "line"},
                }
            }
        }

    def custom_exit_price(self, pair: str, trade: Trade, current_time: datetime, proposed_rate: float, current_profit: float, exit_tag: str, **kwargs):
        """
        自定义退出价格，从交易对象获取保存的价格信息
        """
        if "达到止盈目标" in exit_tag:
            ret = trade.get_custom_data("take_profit_price")
            return ret
        elif "达到止损目标" in exit_tag:
            ret = trade.get_custom_data("stop_loss_price")
            return ret
       
        return proposed_rate

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        """
        利用原生Freqtrade的检查逻辑来触发止盈止损
        """
        take_profit_price = trade.get_custom_data("take_profit_price")
        stop_loss_price = trade.get_custom_data("stop_loss_price")
        if not take_profit_price or not stop_loss_price:    
            take_profit_price, stop_loss_price = self._calculate_tp_sl_prices(trade)
            if take_profit_price and stop_loss_price:
                profit_rate = abs((take_profit_price - trade.open_rate) / trade.open_rate)
                stop_loss_rate = abs((stop_loss_price - trade.open_rate) / trade.open_rate)
                trade.set_custom_data("take_profit_price", take_profit_price)
                trade.set_custom_data("stop_loss_price", stop_loss_price)
                msg = f"current_time:{current_time} pair:{pair} current_rate:{current_rate} current_profit:{current_profit} open_date:{trade.open_date} take_profit_price: {take_profit_price} stop_loss_price: {stop_loss_price} profit_rate:{profit_rate} stop_loss_rate:{stop_loss_rate}"
                logger.info(msg)
                # self.dp.send_msg(f"止盈止损计算完毕:\n{msg}")
      # 如果没有设置止盈止损价格，则不触发退出
        if not take_profit_price or not stop_loss_price:
            return None
        
        is_long = not trade.is_short
        dataframe = self._full_dataframes.get(pair)
        now_candle = dataframe[dataframe['date'] == current_time]
        low_price = now_candle['low'].iloc[0]
        high_price = now_candle['high'].iloc[0]
        # 根据交易方向检查止盈止损条件
        if is_long:
            # 做多: 当前价格 >= 止盈价格 或 当前价格 <= 止损价格
            if high_price >= take_profit_price:
                # logger.info(f"触发做多止盈: {trade.pair} - 当前价格 {current_rate} >= 止盈价格 {take_profit_price}")
                return "达到止盈目标"
            elif low_price <= stop_loss_price:
                # logger.info(f"触发做多止损: {trade.pair} - 当前价格 {current_rate} <= 止损价格 {stop_loss_price}")
                return "达到止损目标"
        else:
            # 做空: 当前价格 <= 止盈价格 或 当前价格 >= 止损价格
            if low_price <= take_profit_price:
                # logger.info(f"触发做空止盈: {trade.pair} - 当前价格 {current_rate} <= 止盈价格 {take_profit_price}")
                return "达到止盈目标"
            elif high_price >= stop_loss_price:
                # logger.info(f"触发做空止损: {trade.pair} - 当前价格 {current_rate} >= 止损价格 {stop_loss_price}")
                return "达到止损目标"
                
        return  None
    
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
        # 获取当前交易对名称
        pair_name = metadata['pair']
        logger.info(f"正在处理交易对: {pair_name}")
        # 计算RSI指标（14周期）
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe["rsi_sma_14"] = ta.SMA(dataframe['rsi'], timeperiod=14)
        dataframe['rsi_above_70'] = dataframe['rsi'] > 70
        dataframe['rsi_below_30'] = dataframe['rsi'] < 30

        shift_sum = 3
        shift_sum_bool = True
        for i in range(1, shift_sum + 1):
            shift_sum_bool = shift_sum_bool & dataframe['rsi_above_70'].shift(i)
        
        dataframe['rsi_short_signal'] = (
            shift_sum_bool &  # 前1根K线RSI>70
            (dataframe['rsi_above_70'] == False) &  # 当前RSI回穿到70以下
            (dataframe['rsi_above_70'].shift(1))  # 前一根RSI还在70以上，确保是回穿
        )
        
        shift_sum_bool = True
        for i in range(1, shift_sum + 1):
            shift_sum_bool = shift_sum_bool & dataframe['rsi_below_30'].shift(i)
        # 做多条件：连续4根K线RSI<30，且当前RSI回穿到30以上
        dataframe['rsi_long_signal'] = (
            shift_sum_bool &  # 前1根K线RSI<30
            (dataframe['rsi_below_30'] == False) &  # 当前RSI回穿到30以上
            (dataframe['rsi_below_30'].shift(1))  # 前一根RSI还在30以下，确保是回穿
        )

        dataframe["ema_7"] = ta.EMA(dataframe['close'], timeperiod=7)
        dataframe["ema_14"] = ta.EMA(dataframe['close'], timeperiod=14)
        dataframe["ema_7_up_cross_14"] = qtpylib.crossed_above(dataframe["ema_7"], dataframe["ema_14"])
        dataframe["ema_7_down_cross_14"] = qtpylib.crossed_below(dataframe["ema_7"], dataframe["ema_14"])
        
        # # 检测前7根K线中是否有RSI信号（使用rolling简化）
        dataframe['rsi_short_in_last_7'] = dataframe['rsi_short_signal'].rolling(window=7).sum().shift(1) > 0
        dataframe['rsi_long_in_last_7'] = dataframe['rsi_long_signal'].rolling(window=7).sum().shift(1) > 0

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percentage'] = (dataframe['atr'] / dataframe['close']) * 100  # ATR百分比
        dataframe["high_price"] = dataframe['high'].rolling(window=11).max() * (1 + dataframe['atr_percentage'] / 100)
        dataframe["low_price"] = dataframe['low'].rolling(window=11).min() * (1 - dataframe['atr_percentage'] / 100)

        dataframe['close_long_signal'] = (
            ((dataframe['rsi_above_70'].shift(1)) &
            (dataframe['rsi_above_70'].shift(2)) &
            (dataframe['rsi_above_70'] == False)) 
        )

        dataframe['close_short_signal'] = (
            ((dataframe['rsi_below_30'].shift(1)) &
            (dataframe['rsi_below_30'].shift(2)) &
            (dataframe['rsi_below_30'] == False)) 
        )

        self._full_dataframes[metadata['pair']] = dataframe
        # 保存包含关键价格信息的CSV文件
        # dataframe.to_csv(f"./user_data/data/{metadata['pair']}.csv")
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

        # 做多条件：RSI long信号
        dataframe.loc[
            (
                (dataframe['ema_7_up_cross_14']) &
                (dataframe['rsi_long_in_last_7'])
            ),
            'enter_long'] = 1
        
        # 做空条件：RSI short信号
        dataframe.loc[
            (
                (dataframe['ema_7_down_cross_14']) &
                (dataframe['rsi_short_in_last_7'])
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
        # 由于我们使用custom_exit方法来处理止盈止损，这里不设置任何退出信号
        # 所有的退出逻辑都在custom_exit方法中处理
         # 做多条件：连续4根K线RSI<30，且当前RSI回穿到30以上
        return dataframe
        dataframe.loc[
            (
                (dataframe['close_long_signal'])
            ),
            'exit_long'] = 1
        
        # 做空条件：连续4根K线RSI>70，且当前RSI回穿到70以下
        dataframe.loc[
            (
                (dataframe['close_short_signal'])
            ),
            'exit_short'] = 1
        
        return dataframe
        