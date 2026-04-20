import logging
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from functools import reduce
from datetime import datetime, timedelta

from freqtrade.strategy import IStrategy, IntParameter


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler('OptimizedRLStrategy.log'))

class OptimizedRLStrategy(IStrategy):
    """
    强化学习优化策略 - 使用RLOptimizedTrader模型
    该策略提供丰富的技术指标作为强化学习模型的输入特征
    用于通过历史k线数据训练模型，预测最佳入场点位和止盈止损设置
    
    该策略需要配合freqai和RLOptimizedTrader模型一起使用
    """

    # 定义基本策略参数
    minimal_roi = {
        "0": 0.1,   # 实际上，具体的ROI将由RL模型控制
    }
    
    stoploss = -0.05  # 这将作为一个安全网，实际止损由RL模型控制
    trailing_stop = False
    
    # 仅在新K线上处理数据
    process_only_new_candles = True
    
    # 使用退出信号
    use_exit_signal = True
    
    # 允许做空
    can_short = True
    
    # 需要预热的K线数量
    startup_candle_count: int = 200
    
    # 可以在freqai配置中自定义的优化参数
    # win_reward_factor = IntParameter(1, 5, default=2, space="buy", optimize=True)
    
    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict, **kwargs):
        """
        为每个周期扩展特征
        """
        # 添加各种周期的技术指标
        dataframe[f"%-rsi-period-{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-cci-period-{period}"] = ta.CCI(dataframe, timeperiod=period)
        dataframe[f"%-mfi-period-{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-adx-period-{period}"] = ta.ADX(dataframe, timeperiod=period)
        
        # 计算不同周期的布林带
        bollinger = ta.BBANDS(dataframe, timeperiod=period, nbdevup=2.0, nbdevdn=2.0)
        dataframe[f"%-bb_upperband-period-{period}"] = bollinger['upperband']
        dataframe[f"%-bb_middleband-period-{period}"] = bollinger['middleband']
        dataframe[f"%-bb_lowerband-period-{period}"] = bollinger['lowerband']
        dataframe[f"%-bb_width-period-{period}"] = (bollinger['upperband'] - bollinger['lowerband']) / bollinger['middleband']
        
        # 添加移动平均线
        dataframe[f"%-sma-period-{period}"] = ta.SMA(dataframe, timeperiod=period)
        dataframe[f"%-ema-period-{period}"] = ta.EMA(dataframe, timeperiod=period)
        
        # 计算MACD
        if period > 13:  # MACD通常需要更长的周期
            macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
            dataframe[f"%-macd-period-{period}"] = macd['macd']
            dataframe[f"%-macdsignal-period-{period}"] = macd['macdsignal']
            dataframe[f"%-macdhist-period-{period}"] = macd['macdhist']
        
        # ATR - 平均真实波动范围
        dataframe[f"%-atr-period-{period}"] = ta.ATR(dataframe, timeperiod=period)
        
        return dataframe
    
    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs):
        """
        添加基本特征
        """
        # 百分比变化
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-pct-change-2"] = dataframe["close"].pct_change(2)
        dataframe["%-pct-change-5"] = dataframe["close"].pct_change(5)
        
        # 交易量指标
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-volume-pct-change"] = dataframe["volume"].pct_change()
        dataframe["%-volume-mean-3"] = dataframe["volume"].rolling(3).mean()
        dataframe["%-volume-mean-10"] = dataframe["volume"].rolling(10).mean()
        
        # 价格-交易量关系
        dataframe["%-price-volume-ratio"] = dataframe["close"] * dataframe["volume"]
        
        # 波动率指标
        dataframe["%-volatility-3"] = dataframe["close"].rolling(3).std()
        dataframe["%-volatility-10"] = dataframe["close"].rolling(10).std()
        
        # 动量指标
        dataframe["%-mom-3"] = dataframe["close"] / dataframe["close"].shift(3)
        dataframe["%-mom-5"] = dataframe["close"] / dataframe["close"].shift(5)
        dataframe["%-mom-10"] = dataframe["close"] / dataframe["close"].shift(10)
        
        # 高低点信息
        dataframe["%-high-low-ratio"] = dataframe["high"] / dataframe["low"]
        dataframe["%-close-open-ratio"] = dataframe["close"] / dataframe["open"]
        
        # 添加各种K线形态特征
        try:
            # 锤子线和上吊线
            dataframe["%-hammer"] = ta.CDLHAMMER(dataframe)
            dataframe["%-hanging_man"] = ta.CDLHANGINGMAN(dataframe)
            
            # 吞没形态
            dataframe["%-engulfing"] = ta.CDLENGULFING(dataframe)
            
            # 十字星
            dataframe["%-doji"] = ta.CDLDOJI(dataframe)
            
            # 启明星和黄昏星
            dataframe["%-morning_star"] = ta.CDLMORNINGSTAR(dataframe)
            dataframe["%-evening_star"] = ta.CDLEVENINGSTAR(dataframe)
        except Exception as e:
            logger.warning(f"无法添加K线形态特征: {e}")
        
        return dataframe
    
    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs):
        """
        添加标准特征工程逻辑，包括时间特征和原始价格数据
        """
        # 添加时间特征
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        dataframe["%-month"] = dataframe["date"].dt.month
        
        # 添加原始价格数据 - 这是RL模型的环境所必需的
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]
        
        # 添加价格差异特征
        dataframe["%-close-open"] = dataframe["close"] - dataframe["open"]
        dataframe["%-high-low"] = dataframe["high"] - dataframe["low"]
        
        # 自定义指标
        dataframe["%-price_momentum"] = dataframe["close"] / dataframe["close"].shift(10)
        
        # 添加趋势判断特征
        dataframe["%-ema9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["%-ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["%-ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["%-ema200"] = ta.EMA(dataframe, timeperiod=200)
        
        # 趋势方向特征
        dataframe["%-trend_ema9_ema21"] = (dataframe["%-ema9"] > dataframe["%-ema21"]).astype(int)
        dataframe["%-trend_ema21_ema50"] = (dataframe["%-ema21"] > dataframe["%-ema50"]).astype(int)
        
        return dataframe
    
    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs):
        """
        为强化学习模型设置目标
        对于RL模型，我们不需要设置特定的目标标签，但需要初始化一个动作列
        """
        # 为RL，没有直接的目标要设置，这只是一个中性填充
        # 直到代理发送一个动作
        dataframe["&-action"] = 0
        
        return dataframe
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        调用FreqAI来处理数据并进行预测
        """
        # 确保在启动时可以访问freqai配置
        if not hasattr(self, "freqai") and "freqai" in self.config:
            self.freqai_info = self.config["freqai"]
        
        # 启动FreqAI进行预测
        if hasattr(self, "freqai"):
            dataframe = self.freqai.start(dataframe, metadata, self)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据RL模型的预测结果确定入场信号
        """
        enter_long_conditions = [
            dataframe["do_predict"] == 1,  # FreqAI认为我们应该进行预测
            dataframe["&-action"] == 1     # RL代理建议做多
        ]
        
        if enter_long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "rl_long")
        
        enter_short_conditions = [
            dataframe["do_predict"] == 1,  # FreqAI认为我们应该进行预测
            dataframe["&-action"] == 3     # RL代理建议做空
        ]
        
        if enter_short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "rl_short")
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据RL模型的预测结果确定出场信号
        """
        exit_long_conditions = [
            dataframe["do_predict"] == 1,  # FreqAI认为我们应该进行预测
            dataframe["&-action"] == 2     # RL代理建议平多
        ]
        
        if exit_long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, exit_long_conditions), ["exit_long", "exit_tag"]
            ] = (1, "rl_long_exit")
        
        exit_short_conditions = [
            dataframe["do_predict"] == 1,  # FreqAI认为我们应该进行预测
            dataframe["&-action"] == 4     # RL代理建议平空
        ]
        
        if exit_short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, exit_short_conditions), ["exit_short", "exit_tag"]
            ] = (1, "rl_short_exit")
        
        return dataframe 