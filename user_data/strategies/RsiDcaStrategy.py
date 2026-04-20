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

ORDER_COUNT = 2
ORDER_DOWN_RATE = 0.05
ORDER_AMOUNT_RATE = 2  # 加仓金额比例

PROFIT_RATE = 0.10
STOPLOSS_RATE = 0.05
# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志输出格式
    handlers=[logging.StreamHandler()]  # 输出到控制台和文件
)
# logging.FileHandler('user_data/logs/RsiDcaStrategy.log')
logger = logging.getLogger(__name__)  # 获取日志记录器

class RsiDcaStrategy(IStrategy):
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
    profit_rate = 3
    INTERFACE_VERSION = 3
    can_short: bool = True
    
    # 启用仓位调整功能
    position_adjustment_enable = True
    max_entry_position_adjustment = ORDER_COUNT - 1  # 最大加仓次数
    
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
                stop_loss_price = dataframe.iloc[candle_index]['rsi_long_ref_low']
                
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
                stop_loss_price = dataframe.iloc[candle_index]['rsi_short_ref_high']
                
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
            "main_plot": {
                "rsi_long_ref_low": {"color": "green", "type": "line"},
                "rsi_short_ref_high": {"color": "red", "type": "line"},
            },
            "subplots": {
                # RSI子图，包含RSI指标和关键水平线
                "RSI": {
                    "rsi": {"color": "blue"},
                    "rsi_sma_14": {"color": "orange"},
                    "rsi_1d": {"color": "purple"},  # 添加1天级别RSI
                    # 添加RSI的关键水平线
                    "rsi_overbought": {"color": "red", "type": "line"},
                    "rsi_oversold": {"color": "green", "type": "line"},
                },
                # 波动率子图
                "Volatility": {
                    "current_volatility": {"color": "blue"},
                    "volatility_1d": {"color": "red"},
                    "intraday_volatility_1d": {"color": "orange"},
                }
            }
        }

    def custom_exit_price(self, pair: str, trade: Trade, current_time: datetime, proposed_rate: float, current_profit: float, exit_tag: str, **kwargs):
        """
        自定义退出价格，从交易对象获取保存的价格信息
        """
        if "达到止盈目标" in exit_tag:
            ret = trade.get_custom_data("take_profit_price")
            # logger.info(f"custom_exit_price current_time:{current_time} trade:{trade} 止盈价格: {ret} order_count:{trade.nr_of_successful_entries}")
            return ret
        elif "达到止损目标" in exit_tag:
            ret = trade.get_custom_data("stop_loss_price")
            # logger.info(f"custom_exit_price current_time:{current_time} trade:{trade} 止损价格: {ret} order_count:{trade.nr_of_successful_entries}")
            return ret
       
        return proposed_rate

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        """
        利用原生Freqtrade的检查逻辑来触发止盈止损
        """
        is_long = not trade.is_short
        dataframe = self._full_dataframes.get(pair)
        now_candle = dataframe[dataframe['date'] == current_time]
        low_price = now_candle['low'].iloc[0]
        high_price = now_candle['high'].iloc[0]
        
        flag = 1 if is_long else -1 
        take_profit_price = trade.open_rate * (1 + flag * PROFIT_RATE/2**(trade.nr_of_successful_entries-1))
        if trade.nr_of_successful_entries == 1:
            stop_loss_price = trade.open_rate * (1 - flag * STOPLOSS_RATE)
        elif trade.nr_of_successful_entries > 1:
            stop_loss_price = trade.open_rate * (1 - flag * STOPLOSS_RATE)
        # # 根据交易方向检查止盈止损条件
        if is_long:
            if high_price >= take_profit_price:
                trade.set_custom_data("take_profit_price", take_profit_price)
                return "达到止盈目标"
            
            if trade.nr_of_successful_entries == ORDER_COUNT:
                if low_price <= stop_loss_price:
                    trade.set_custom_data("stop_loss_price", stop_loss_price)
                    return "达到止损目标"
        else:
            if low_price <= take_profit_price:
                trade.set_custom_data("take_profit_price", take_profit_price)
                return "达到止盈目标"
            if trade.nr_of_successful_entries == ORDER_COUNT:
                if high_price >= stop_loss_price:
                    trade.set_custom_data("stop_loss_price", stop_loss_price)
                    return "达到止损目标"
        return None
    
    def custom_entry_price(self, pair: str, trade: Trade | None, current_time: datetime, 
                       proposed_rate: float, entry_tag: str | None, side: str, **kwargs):
        """决定加仓时的价格"""
        if trade is None:
            return proposed_rate  # 首次开仓使用默认价格
        
        is_long = not trade.is_short
        
        price = trade.open_rate * (1 - ORDER_DOWN_RATE) if is_long else trade.open_rate * (1 + ORDER_DOWN_RATE)
        # logger.info(f"custom_entry_price current_time:{current_time} trade:{trade} 加仓价格: {price}")
        return price
        
    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: Optional[float],
                              max_stake: float, current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs):
        """
        自定义仓位调整逻辑 - 实现DCA加仓
        """
        pair = trade.pair
        
        # 检查是否还有加仓次数
        if trade.nr_of_successful_entries >= ORDER_COUNT:
            return None
            
        # 获取当前数据帧
        dataframe = self._full_dataframes.get(pair)
        if dataframe is None:
            return None
            
        # 找到当前时间对应的K线数据
        try:
            current_candle = dataframe[dataframe['date'] <= current_time].iloc[-1]
            low_price = current_candle['low']
            high_price = current_candle['high']
        except:
            return None
        
        is_long = not trade.is_short
        
        if is_long:
            if low_price <= trade.open_rate * (1 - ORDER_DOWN_RATE):
                # logger.info(f"adjust_trade_position current_time:{current_time} trade:{trade} 触发加仓 nr_of_successful_entries:{trade.nr_of_successful_entries}")
                return trade.amount * ORDER_AMOUNT_RATE * trade.open_rate * (1 - ORDER_DOWN_RATE)
        else:
            # 做空加仓条件：价格上涨超过设定比例  
            if high_price >= trade.open_rate * (1 + ORDER_DOWN_RATE):
                # logger.info(f"adjust_trade_position current_time:{current_time} trade:{trade} 触发加仓 nr_of_successful_entries:{trade.nr_of_successful_entries}")
                return trade.amount * ORDER_AMOUNT_RATE * trade.open_rate * (1 + ORDER_DOWN_RATE)
                
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
        # 添加1天级别的数据获取
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for pair in pairs:
            informative_pairs.append((pair, "1d"))
        return informative_pairs

    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 计算趋势方向
        dataframe["7_ema"] = ta.EMA(dataframe['close'], timeperiod=7)
        dataframe["14_ema"] = ta.EMA(dataframe['close'], timeperiod=14)
        dataframe["7_ema_down_cross_14"] = dataframe["7_ema"] < dataframe["14_ema"]
        dataframe["7_ema_up_cross_14"] = dataframe["14_ema"] > dataframe["7_ema"]
        logger.info("populate_indicators_1d")
        dataframe.to_csv(f"./user_data/data/1h.csv")
        return dataframe

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
        
        # 获取1天级别的informative数据
        informative_1d = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1d')
        
        # 合并1天级别数据到1小时级别
        dataframe = merge_informative_pair(dataframe, informative_1d, self.timeframe, '1d', ffill=True)
        
        # 计算1小时级别的当前K线波动率
        dataframe['current_volatility'] = (dataframe['high'] - dataframe['low']) / dataframe['open']
        
        # 计算RSI指标（14周期）
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe["rsi_sma_14"] = ta.SMA(dataframe['rsi'], timeperiod=14)
        dataframe['rsi_above_70'] = dataframe['rsi'] > 70
        dataframe['rsi_below_30'] = dataframe['rsi'] < 30
        dataframe["sma_7"] = ta.SMA(dataframe['close'], timeperiod=7)

        dataframe["close_down_sma_7"] = dataframe['close'] < dataframe['sma_7']
        dataframe["close_up_sma_7"] = dataframe['close'] > dataframe['sma_7']

        dataframe['rsi_short_signal'] = (
            dataframe['rsi_above_70'].shift(1) &  # 前1根K线RSI>70
            dataframe['rsi_above_70'].shift(2) &  # 前2根K线RSI>70
            # dataframe['rsi_above_70'].shift(3) &  # 前3根K线RSI>70
            # dataframe['rsi_above_70'].shift(4) &  # 前4根K线RSI>70
            (dataframe['rsi_above_70'] == False) &  # 当前RSI回穿到70以下
            (dataframe['rsi_above_70'].shift(1))   # 前一根RSI还在70以上，确保是回穿
            # (dataframe['close_down_sma_7'])  # 前一根K线收盘价在7SMA以下
        )
        
        # 做多条件：连续4根K线RSI<30，且当前RSI回穿到30以上
        dataframe['rsi_long_signal'] = (
            dataframe['rsi_below_30'].shift(1) &  # 前1根K线RSI<30
            dataframe['rsi_below_30'].shift(2) &  # 前2根K线RSI<30
            # dataframe['rsi_below_30'].shift(3) &  # 前3根K线RSI<30
            # dataframe['rsi_below_30'].shift(4) &  # 前4根K线RSI<30    
            (dataframe['rsi_below_30'] == False) &  # 当前RSI回穿到30以上
            (dataframe['rsi_below_30'].shift(1))  # 前一根RSI还在30以下，确保是回穿
            # (dataframe['close_up_sma_7'])  # 前一根K线收盘价在7SMA以下
        )

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
        
        # 记录开仓信号前RSI连续极值区间的关键价格点
        dataframe['rsi_long_ref_low'] = 0.0  # 开多信号前RSI连续<30期间的最低价
        dataframe['rsi_short_ref_high'] = 0.0  # 开空信号前RSI连续>70期间的最高价
        
        # 计算开多信号的参考最低价
        for i in range(len(dataframe)):
            if dataframe['rsi_long_signal'].iloc[i]:
                # 向前查找连续RSI<30的区间
                end_idx = i - 1  # 信号前一根K线
                start_idx = end_idx
                
                # 向前找到连续RSI<30区间的开始
                while start_idx >= 0 and dataframe['rsi_below_30'].iloc[start_idx]:
                    start_idx -= 1
                start_idx += 1  # 调整到第一个RSI<30的位置
                
                # 如果找到了连续RSI<30的区间，记录该区间的最低价
                if start_idx <= end_idx and start_idx >= 0:
                    low_price = dataframe['low'].iloc[start_idx:end_idx+1].min()
                    dataframe.iloc[i, dataframe.columns.get_loc('rsi_long_ref_low')] = low_price
                    
        # 计算开空信号的参考最高价        
        for i in range(len(dataframe)):
            if dataframe['rsi_short_signal'].iloc[i]:
                # 向前查找连续RSI>70的区间
                end_idx = i - 1  # 信号前一根K线
                start_idx = end_idx
                
                # 向前找到连续RSI>70区间的开始
                while start_idx >= 0 and dataframe['rsi_above_70'].iloc[start_idx]:
                    start_idx -= 1
                start_idx += 1  # 调整到第一个RSI>70的位置
                
                # 如果找到了连续RSI>70的区间，记录该区间的最高价
                if start_idx <= end_idx and start_idx >= 0:
                    high_price = dataframe['high'].iloc[start_idx:end_idx+1].max()
                    dataframe.iloc[i, dataframe.columns.get_loc('rsi_short_ref_high')] = high_price

        # dataframe['close_long_signal'] = (
        #     dataframe["ema_7_down_cross_14"]
        # )

        # dataframe['close_short_signal'] = (
        #     dataframe["ema_7_up_cross_14"]
        # )
        
        # 添加RSI水平线用于图表显示
        dataframe['rsi_overbought'] = 70  # 超买线
        dataframe['rsi_oversold'] = 30    # 超卖线
        
        self._full_dataframes[metadata['pair']] = dataframe
        
        # 保存包含关键价格信息的CSV文件
        # dataframe.to_csv(f"./user_data/data/{metadata['pair']}.csv")
        
        # 统计信号数量
        long_count = dataframe[dataframe['rsi_long_signal']].shape[0]
        short_count = dataframe[dataframe['rsi_short_signal']].shape[0]
        
        if long_count > 0:
            logger.info(f"{metadata['pair']} 总共发现 {long_count} 个做多信号")
        if short_count > 0:
            logger.info(f"{metadata['pair']} 总共发现 {short_count} 个做空信号")
            
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
                (dataframe['rsi_long_signal'])
            ),
            'enter_long'] = 1
        
        # 做空条件：RSI short信号
        dataframe.loc[
            (
                (dataframe['rsi_short_signal'])
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


# ======================================
# 加仓平均价格计算演示函数
# ======================================

def demonstrate_dca_pricing():
    """
    演示DCA加仓后的平均价格计算
    """
    print("\n" + "="*60)
    print("📊 DCA加仓平均价格计算演示")
    print("="*60)
    
    # 示例1: 做多BTC加仓
    print("\n🔵 做多BTC加仓示例:")
    print("-" * 40)
    
    # 初始开仓
    initial_price = 45000.0
    initial_amount = 0.002  # 0.002 BTC
    initial_value = initial_price * initial_amount
    
    print(f"初始开仓: {initial_price} USDT x {initial_amount} BTC = {initial_value} USDT")
    
    # 第一次加仓
    dca1_price = 42000.0  # 价格下跌到42000
    dca1_value = 90.0     # 加仓90 USDT
    dca1_amount = dca1_value / dca1_price
    
    # 计算新的平均价格
    total_value = initial_value + dca1_value
    total_amount = initial_amount + dca1_amount
    avg_price_1 = total_value / total_amount
    
    print(f"第1次加仓: {dca1_price} USDT, 投入 {dca1_value} USDT")
    print(f"加仓数量: {dca1_amount:.6f} BTC")
    print(f"总价值: {total_value} USDT")
    print(f"总数量: {total_amount:.6f} BTC")
    print(f"新平均价格: {avg_price_1:.2f} USDT")
    print(f"成本降低: {((initial_price - avg_price_1) / initial_price * 100):.2f}%")
    
    # 第二次加仓
    dca2_price = 40000.0  # 继续下跌到40000
    dca2_value = 99.0     # 再加仓99 USDT  
    dca2_amount = dca2_value / dca2_price
    
    # 计算最终平均价格
    final_value = total_value + dca2_value
    final_amount = total_amount + dca2_amount
    avg_price_2 = final_value / final_amount
    
    print(f"第2次加仓: {dca2_price} USDT, 投入 {dca2_value} USDT")
    print(f"加仓数量: {dca2_amount:.6f} BTC")
    print(f"最终价值: {final_value} USDT")
    print(f"最终数量: {final_amount:.6f} BTC") 
    print(f"最终平均价格: {avg_price_2:.2f} USDT")
    print(f"总成本降低: {((initial_price - avg_price_2) / initial_price * 100):.2f}%")
    
    print("\n" + "="*60)
    print("💡 关键要点:")
    print("- trade.open_rate 是加权平均价格，不是简单平均")  
    print("- 做多加仓：平均价格下降，降低成本")
    print("- 做空加仓：平均价格上升，提高成本")
    print("- Freqtrade会自动更新trade.open_rate和trade.amount")
    print("="*60)

# 如果直接运行此文件，执行演示
if __name__ == "__main__":
    demonstrate_dca_pricing()
        