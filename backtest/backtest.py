from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt

from backtest.utils import calculate_DD

class GoldenCrossBacktest:
    def __init__(self, exchange_id:str, symbols:list[str], fast_sma_window:int=50, slow_sma_window:int=200,
                 timeframe:str="1d", since:int="01/01/2017", export:bool=True, export_folder:str="",
                 plot_benchmark:bool=True) -> None:
        self.exchange_id = exchange_id
        self.symbols = symbols
        self.fast_sma_window = fast_sma_window
        self.slow_sma_window = slow_sma_window
        self.timeframe = timeframe
        self.since = since
        self.export = export
        self.export_folder = export_folder
        self.plot_benchmark = plot_benchmark

        self.since_millis = self.date_to_milliseconds(self.since)
        self.strategy_name = "golden_cross"
        self.fee = 0.0001
        self.default_figsize = (14, 7)
        self.ohlcv_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        
        
        self.exchange = getattr(ccxt, exchange_id)()

        self.fetch_raw_ohlcvs()

    def date_to_milliseconds(self, date_string:str):
        date_format = "%d/%m/%Y"
        date = datetime.strptime(date_string, date_format)
        milliseconds = int(date.timestamp() * 1000)
        return milliseconds
    
    def fetch_raw_ohlcvs(self):
        raw_ohlcvs = {}
        for symbol in self.symbols:
            price_date = self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, since=0, params={"paginate": True})
            
            ohlcv = pd.DataFrame(price_date, columns=self.ohlcv_columns)
            ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], unit="ms")
            ohlcv = ohlcv.set_index("timestamp")
            
            raw_ohlcvs[symbol] = ohlcv

        self.raw_ohlcvs = raw_ohlcvs
    
    def calc_features(self):
        ohlcvs = self.raw_ohlcvs.copy()
        for symbol, ohlcv in self.raw_ohlcvs.items():
            fast_sma = ohlcv["close"].rolling(window=self.fast_sma_window).mean()
            slow_sma = ohlcv["close"].rolling(window=self.slow_sma_window).mean()

            ohlcvs[symbol]["sma_cross_up"] = np.where((fast_sma > slow_sma) & (fast_sma.shift(1) < slow_sma.shift(1)), True, False)
            ohlcvs[symbol]["sma_cross_down"] = np.where((fast_sma < slow_sma) & (fast_sma.shift(1) > slow_sma.shift(1)), True, False)

            ohlcvs[symbol]["fast_sma"] = fast_sma
            ohlcvs[symbol]["slow_sma"] = slow_sma

            
            ohlcvs[symbol]["in_position"] = False # We add a in_position column to keep track wether we are in a position

        return ohlcvs
    
    def backtest(self):
        results = {}
        ohlcvs = self.calc_features()

        for symbol, ohlcv in ohlcvs.items():
            ohlcv = ohlcv.dropna()

            open_position = {}
            closed_trades = []

            for index, row in ohlcv.iterrows():
                if row["sma_cross_up"] and not open_position:
                    open_position = {
                        "entry_date": index,
                        "entry_price": row["close"],
                        "fee": self.fee
                    }

                    ohlcv.at[index, "in_position"] = True

                elif row["sma_cross_down"] and open_position:
                    open_position["exit_date"] = index
                    open_position["exit_price"] = row["close"]
                    open_position["fee"] += self.fee

                    open_position["pnl"] = open_position["exit_price"] / open_position["entry_price"] - 1 - open_position["fee"]

                    closed_trades.append(open_position)
                    open_position = {}

                    # Set "in_position" to True for the current row and False for the next row
                    ohlcv.at[index, "in_position"] = True
                    current_loc = ohlcv.index.get_loc(index)
                    if current_loc < len(ohlcv) - 1:
                        next_index = ohlcv.index[current_loc + 1]
                        ohlcv.at[next_index, "in_position"] = False

                else:
                    ohlcv.at[index, "in_position"] = bool(open_position)

            if self.export:
                exportable_symbol = symbol.replace("/", "").replace(":", "")
                closed_trades_df = pd.DataFrame(closed_trades)
                closed_trades_df.to_csv(f"{self.export_folder}closed_trades_{exportable_symbol}.csv")
            
            results[symbol] = (closed_trades, ohlcv["in_position"])

        return results

    def create_unrealised_returns_df(self, symbol, in_position_array):
        unrealised_returns = pd.concat([self.raw_ohlcvs[symbol]["close"], in_position_array], axis=1)
        unrealised_returns["pct_change"] = unrealised_returns["close"].pct_change()
        unrealised_returns = unrealised_returns.dropna()
        unrealised_returns["pnl"] = np.where(unrealised_returns["in_position"], unrealised_returns["pct_change"], 0)
        unrealised_returns["comp_pnl"] = (unrealised_returns["pnl"] + 1).cumprod() - 1

        return unrealised_returns

    def plot_and_stats(self, backtest_results):
        for symbol, backtest_result in backtest_results.items():
            in_position_array = backtest_result[1]
            unrealised_returns = self.create_unrealised_returns_df(symbol, in_position_array)

            self.print_stats(symbol, unrealised_returns["pnl"], unrealised_returns["comp_pnl"])
            self.plot_equity_curve(symbol, unrealised_returns["pnl"], unrealised_returns["comp_pnl"])

    def plot_equity_curve(self, symbol, returns, comp_returns):
        """
        Plots the equity curve, underwater drawdown, and benchmark (if applicable) for the given returns.
        """
        _, _, _, _, drawdown = calculate_DD(returns)

        start_date = comp_returns.index[0]
        benchmark_symbol = symbol

        if self.plot_benchmark:
            benchmark_pnl = self.raw_ohlcvs[benchmark_symbol]["close"].pct_change()
            benchmark_pnl = benchmark_pnl[start_date:]
            benchmark_comp_pnl = (benchmark_pnl + 1).cumprod() - 1
            benchmark_percentages = benchmark_comp_pnl * 100
        
        cum_returns_percentages = comp_returns * 100
        drawdown_percentages = drawdown * 100
        
        plt.figure(figsize=self.default_figsize)

        # Equity curve
        plt.subplot(2, 1, 1)
        plt.plot(cum_returns_percentages, label="Equity Curve", color="b")
        if self.plot_benchmark:
            symbol_base = benchmark_symbol.split("/")[0]
            plt.plot(benchmark_percentages, label=f"Benchmark ({symbol_base})", color="y")
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns (%)")
        plt.legend()
        plt.grid(True)

        # Drawdown
        plt.subplot(2, 1, 2)
        plt.fill_between(cum_returns_percentages.index, 0, drawdown_percentages, color="red", alpha=0.3)
        plt.title("Underwater Drawdown Plot")
        plt.xlabel("Date")
        plt.ylabel("Drawdown Percentage (%)")
        plt.grid(True)

        plt.suptitle(f"{self.strategy_name} Strategy - {symbol} {self.exchange_id}", fontsize=16)
        
        plt.tight_layout()
        
        if self.export:
            exportable_symbol = symbol.replace("/", "").replace(":", "")
            plt.savefig(f"{self.export_folder}{self.strategy_name}_{exportable_symbol}_{self.exchange_id}.png")
        
        plt.close()

    def print_stats(self, symbol, returns: pd.DataFrame, cum_returns: pd.DataFrame) -> dict:
        """
        Computes and displays various trading statistics, such as average trade returns, win rate, PnL, drawdowns,
        and other performance metrics, based on the provided returns.
        """
        maxDD, maxDDD, avgDD, maxDD_date, _ = calculate_DD(returns)

        stats = {
            "Avg Trade Return": returns.mean(),
            "Avg Positive Trade Return": returns[returns >= 0].mean(),
            "Avg Negative Trade Return": returns[returns < 0].mean(),
            "Win Rate": len(returns[returns >= 0]) / len(returns),
            "PnL (%)": cum_returns.iloc[-1] * 100,
            "Max Drawdown": maxDD,
            "Max Drawdown Duration": maxDDD,
            "Max Drawdown Date": maxDD_date,
            "Avg Drawdown": avgDD
        }

        print(symbol)
        pprint(stats)
        print("\n")
        
        return stats
 

backtest = GoldenCrossBacktest(
    exchange_id="binance",
    symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "AVAX/USDT", "DOGE/USDT", "ADA/USDT", "TRX/USDT", "SHIB/USDT"],
    fast_sma_window=50, slow_sma_window=200, timeframe="1d", since="01/01/2017", export_folder="results/", plot_benchmark=True
)


backtest_results = backtest.backtest()
backtest.plot_and_stats(backtest_results)