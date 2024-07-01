import numpy as np
import pandas as pd
import ccxt.pro
from tinydb import TinyDB

import asyncio
import logging

import config
from log_setup import setup_logger

class GoldenCrossLive:
    def __init__(self, exchange_id:str, symbols:list[str], black_list:list[str]=[], timeframe:str="1m",
                 testnet:bool=True, balance_reduction:float=0.05, 
                 fast_sma_window:int=50, slow_sma_window:int=200, logging_lvl:int=logging.DEBUG):
        self.exchange_id = exchange_id
        self.testnet = testnet
        self.symbols = symbols
        self.black_list = black_list
        self.timeframe = timeframe
        self.balance_reduction = balance_reduction

        self.fast_sma_window = fast_sma_window
        self.slow_sma_window = slow_sma_window

        self.logger = setup_logger(logging_lvl)

        self.strategy_id = "golden_cross"
        self.ohlcv_columns = ["timestamp", "open", "high", "low", "close", "volume"]

        self.exchange = getattr(ccxt.pro, self.exchange_id)({"apiKey": config.api_key, "secret": config.secret})

        self.db = TinyDB('closed_trades.json') # TODO: Replace with an actual online database -> neon.tech

        if self.testnet:
            self.logger.debug("testnet")
            self.exchange.set_sandbox_mode(True)
        else:
            self.logger.debug("production")

        self.ohlcvs = {}
        self.open_position = {}

    async def init(self):
        self.logger.info("Running %s strategy", self.strategy_id)
        self.markets = await self.exchange.load_markets()
        self.usdt_balance = await self.exchange.fetch_balance()["USDT"]["free"]

    async def watch_ohlcv(self, symbol:str, timeframe:str):
        self.ohlcvs[symbol] = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
        while True:
            try:
                trades = await self.exchange.watch_trades(symbol)
                # self.exchange.watch_ohlcv -> latency
                if len(trades) > 0:
                    ohlcvc = self.exchange.build_ohlcvc(trades, timeframe)
                    await self.handle_ohlcv(symbol, ohlcvc)

            except ccxt.NetworkError:
                asyncio.sleep(5)

    async def handle_ohlcv(self, symbol:str, ohlcvc:list):
        curr_ohlcv = self.ohlcvs[symbol] # dict -> 200 REST req
        latest_candle = ohlcvc[-1][:-1] # ws

        if latest_candle[0] > curr_ohlcv[-1][0]: # 
            self.ohlcvs[symbol].pop(0)
            self.ohlcvs[symbol].append(latest_candle)
        else:
            self.ohlcvs[symbol][-1] = latest_candle

        self.logger.debug("%s %s", symbol, latest_candle)

        await self.calc_features(symbol)

    async def calc_features(self, symbol:str):
        ohlcv = pd.DataFrame(self.ohlcvs[symbol], columns=self.ohlcv_columns)

        fast_sma = ohlcv["close"].rolling(window=self.fast_sma_window).mean()
        slow_sma = ohlcv["close"].rolling(window=self.slow_sma_window).mean()

        self.logger.debug("%s %s", fast_sma, slow_sma)

        if not self.open_position:
            long_signal = (fast_sma.iloc[-1] > slow_sma.iloc[-1]) and (fast_sma.iloc[-2] < slow_sma.iloc[-2])
            if long_signal:
                curr_price = ohlcv["close"].iloc[-1]
                await self.open_position(symbol, curr_price)
        else:
            sma_cross_down = (fast_sma.iloc[-1] < slow_sma.iloc[-1]) and (fast_sma.iloc[-2] > slow_sma.iloc[-2])
            if sma_cross_down:
                await self.close_position(symbol)

    async def open_position(self, symbol:str, curr_price:float):
        usdt_amount = self.usdt_balance * (1 - self.balance_reduction) / len(self.symbols)

        amount = usdt_amount / curr_price

        buy_market_order = await self.exchange.create_order(symbol, "market", "buy", amount)
        asyncio.sleep(0.1)
        entry_trade = await self.exchange.fetch_order(buy_market_order["id"], symbol)

        self.open_position["symbol"] = symbol
        self.open_position["entry_id"] = entry_trade["id"]
        self.open_position["entry_timestamp"] = entry_trade["timestamp"]
        self.open_position["entry_price"] = entry_trade["average"]
        self.open_position["amount"] = entry_trade["amount"]
        self.open_position["side"] = entry_trade["side"]

        base = symbol.split("/")[0] # -> check market structure ccxt
        self.logger.info("%s - %s trade opened with an amount of %s %s, at %s", entry_trade["timestamp"],
                         symbol, entry_trade["amount"], base, entry_trade["average"])

        self.balance = await self.exchange.fetch_balance()["USDT"]["free"]

    async def close_position(self, symbol:str):
        sell_market_order = await self.exchange.create_order(symbol, "market", "buy", self.open_position["amount"])
        asyncio.sleep(0.1)
        exit_trade = await self.exchange.fetch_order(sell_market_order["id"], symbol)

        self.open_position["exit_id"] = exit_trade["id"]
        self.open_position["exit_timestamp"] = exit_trade["timestamp"]
        self.open_position["exit_price"] = exit_trade["average"]
        self.open_position["pnl"] = (self.open_position["exit_price"] - self.open_position["entry_price"]) / self.open_position["entry_price"]

        self.logger.info("%s - %s trade closed at %s", exit_trade["timestamp"],
                         symbol, exit_trade["average"])

        self.db.insert(self.open_position) # TODO: replace by an actual DB
        self.open_position = {}

        self.balance = await self.exchange.fetch_balance()["USDT"]["free"]

    async def run(self):
        await self.init()
        await asyncio.gather(*[self.watch_ohlcv(symbol, self.timeframe) for symbol in self.symbols])

        await self.exchange.close()

if __name__ == "__main__":
    golden_cross = GoldenCrossLive(
        exchange_id="bybit",
        symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
        timeframe="1m",
        logging_lvl=logging.INFO
    )

    asyncio.run(golden_cross.run())
