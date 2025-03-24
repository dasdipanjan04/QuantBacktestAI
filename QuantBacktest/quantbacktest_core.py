import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

class DataLoader:
    def __init__(self, filepath=None):
        self.filepath = filepath

    def load(self):
        if self.filepath:
            df = pd.read_csv(self.filepath, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
            return df
        else:
            raise ValueError("No file path provided")

    @staticmethod
    def fetch_yahoo(symbol: str, start: str, end: str, interval: str = "1d"):
        try:
            data = yf.download(symbol, start=start, end=end, interval=interval)
            data.reset_index(inplace=True)
            data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Date': 'timestamp'
            }, inplace=True)
            data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            data.set_index('timestamp', inplace=True)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None


class BaseStrategy:
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Must implement generate_signals method")


class MovingAverageCrossoverStrategy(BaseStrategy):
    def __init__(self, short_window=10, long_window=50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()
        signal = np.where(short_ma > long_ma, 1, 0)
        signal = pd.Series(signal, index=data.index)
        signal = signal.diff().fillna(0)  # +1 = Buy, -1 = Sell, 0 = Hold
        return signal


class Backtester:
    def __init__(self, data: pd.DataFrame, strategy: BaseStrategy, initial_capital: float = 10000.0):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.results = None

    def run(self):
        signals = self.strategy.generate_signals(self.data)
        positions = signals.replace(-1, 0).cumsum()
        portfolio = pd.DataFrame(index=self.data.index)
        portfolio['positions'] = positions
        portfolio['holdings'] = self.data['close'] * portfolio['positions']
        portfolio['cash'] = self.initial_capital - (signals * self.data['close']).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0)
        self.results = portfolio
        return portfolio

    def plot(self):
        if self.results is not None:
            self.results['total'].plot(title='Portfolio Value Over Time', figsize=(12, 6))
            plt.ylabel('Portfolio Value')
            plt.grid(True)
            plt.show()
        else:
            print("Run the backtest first.")


def compute_metrics(portfolio: pd.DataFrame):
    total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) - 1
    sharpe_ratio = (portfolio['returns'].mean() / portfolio['returns'].std()) * np.sqrt(252)
    drawdown = portfolio['total'] / portfolio['total'].cummax() - 1
    max_drawdown = drawdown.min()
    return {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }
