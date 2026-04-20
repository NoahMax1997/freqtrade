freqtrade new-config --config config.json
freqtrade backtesting --config config.json --strategy AwesomeStrategy -i 1m --timerange 20230425-20250425
freqtrade new-strategy --strategy AwesomeStrategy
freqtrade download-data --config config.json --exchange binance --timeframes 1h --days 730 --dl-interval 2
source .venv/bin/activate
sh setup.sh -i
freqtrade trade -c ./config.json --strategy PriceVolume1MStrategy
freqtrade trade -c ./config_binance_all_notice.json --strategy PriceVolume1MNoticeStrategy


["BTC/USDT:USDT", "ETH/USDT:USDT", "TRUMP/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", "SUI/USDT:USDT", "DOGE/USDT:USDT", "ZEREBRO/USDT:USDT", "1000PEPE/USDT:USDT", "FARTCOIN/USDT:USDT", "TURBO/USDT:USDT", "ADA/USDT:USDT", "BNB/USDT:USDT","LINK/USDT:USDT", "ENA/USDT:USDT",  "AVAX/USDT:USDT", "ALPACA/USDT:USDT", "WIF/USDT:USDT", "NEIROETH/USDT:USDT", "JELLYJELLY/USDT:USDT", "TAO/USDT:USDT", "PNUT/USDT:USDT", "VOXEL/USDT:USDT", "AAVE/USDT:USDT", "CRV/USDT:USDT", "MAGIC/USDT:USDT", "LTC/USDT:USDT", "POPCAT/USDT:USDT", "IMX/USDT:USDT",  "ACT/USDT:USDT", "OM/USDT:USDT", "BABY/USDT:USDT",  "ONDO/USDT:USDT", "FIL/USDT:USDT", "LAYER/USDT:USDT", "MUBARAK/USDT:USDT", "MELANIA/USDT:USDT", "DOT/USDT:USDT", "1000SHIB/USDT:USDT", "AERGO/USDT:USDT",  "BCH/USDT:USDT", "HBAR/USDT:USDT",  "TIA/USDT:USDT", "VIRTUAL/USDT:USDT", "1000BONK/USDT:USDT", "SYN/USDT:USDT", "EOS/USDT:USDT", "MOVE/USDT:USDT", "NEAR/USDT:USDT", "PEOPLE/USDT:USDT", "ARC/USDT:USDT", "MYRO/USDT:USDT", "BIO/USDT:USDT", "PAXG/USDT:USDT", "GRIFFAIN/USDT:USDT", "FET/USDT:USDT"]
  

freqtrade hyperopt --config config_hyperopt.json --hyperopt-loss SharpeHyperOptLoss --strategy SampleStrategy --timerange 20210101-20211231 --spaces buy sell roi stoploss

freqtrade backtesting --config config_binance_average_backtest.json --strategy AverageKlineStrategy --enable-position-stacking  -i 15m --timerange 20250421-20250425

freqtrade backtesting --config config_binance_average_backtest.json --strategy AverageKlineStrategy  -i 15m --timeframe-detail 1m --timerange 20250421-20250425


["FARTCOIN/USDT:USDT","DOGE/USDT:USDT","ZEREBRO/USDT:USDT","TURBO/USDT:USDT","TAO/USDT:USDT","XRP/USDT:USDT","AERGO/USDT:USDT","BABY/USDT:USDT"]

freqtrade trade --config config_cci_strategy.json --dry-run -v