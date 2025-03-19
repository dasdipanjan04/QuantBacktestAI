import yfinance as yf

class DataLoader:
    @staticmethod
    def fetch_yahoo(symbol: str, start: str, end: str, interval: str = "1d"):
        """
        Fetch historical market data from Yahoo Finance.

        :param symbol: Ticker symbol (e.g., "AAPL" for Apple, "BTC-USD" for Bitcoin)
        :param start: Start date (YYYY-MM-DD)
        :param end: End date (YYYY-MM-DD)
        :param interval: Data interval ("1d", "1h", "5m")
        :return: Pandas DataFrame with historical OHLCV data
        """
        try:
            data = yf.download(symbol, start=start, end=end, interval=interval)
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

if __name__ == "__main__":
    df = DataLoader.fetch_yahoo(symbol="AAPL", start="2023-01-01", end="2023-12-31")
    print(df.head())
