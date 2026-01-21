"""
数据服务模块 - 股票数据获取与缓存
"""
import hashlib
import pickle
from pathlib import Path
from typing import Optional
import pandas as pd
import akshare as ak

from backend.core.settings import settings


class DataService:
    """数据服务类 - 负责股票数据获取和缓存"""

    def __init__(self):
        self.cache_dir = settings.AKSHARE_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, stock_code: str, start_date: str, end_date: str) -> Path:
        """生成缓存文件路径"""
        cache_key = f"{stock_code}_{start_date}_{end_date}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"{cache_hash}.pkl"

    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_to_cache(self, data: pd.DataFrame, cache_path: Path) -> None:
        """保存数据到缓存"""
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def get_stock_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        获取股票历史数据

        Args:
            stock_code: 股票代码，如 "000001" 或 "000001.SZ"
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"
            use_cache: 是否使用缓存

        Returns:
            包含OHLCV数据的DataFrame
        """
        # 标准化股票代码
        stock_code = self._normalize_stock_code(stock_code)

        # 检查缓存
        if use_cache and settings.AKSHARE_CACHE_ENABLED:
            cache_path = self._get_cache_path(stock_code, start_date, end_date)
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data

        # 从 akshare 获取数据
        try:
            if stock_code.endswith(".SH"):
                symbol = stock_code.replace(".SH", "")
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",  # 前复权
                )
            elif stock_code.endswith(".SZ"):
                symbol = stock_code.replace(".SZ", "")
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )
            else:
                # 尝试自动识别
                df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )

            # 标准化列名
            df = self._standardize_columns(df)

            # 保存到缓存
            if use_cache and settings.AKSHARE_CACHE_ENABLED:
                cache_path = self._get_cache_path(stock_code, start_date, end_date)
                self._save_to_cache(df, cache_path)

            return df

        except Exception as e:
            raise ValueError(f"获取股票 {stock_code} 数据失败: {e}")

    def _normalize_stock_code(self, code: str) -> str:
        """标准化股票代码格式"""
        code = code.strip().upper()
        if not code.endswith((".SH", ".SZ")):
            # 自动判断上海或深圳
            if code.startswith("6"):
                return f"{code}.SH"
            elif code.startswith(("0", "3")):
                return f"{code}.SZ"
        return code

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化DataFrame列名"""
        # akshare 返回的列名映射
        column_mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }

        df = df.rename(columns=column_mapping)

        # 确保日期列是datetime类型
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        # 确保数值列是正确的类型
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.sort_index()

    def get_multiple_stocks_data(
        self,
        stock_codes: list[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        获取多个股票的数据

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存

        Returns:
            字典，key为股票代码，value为对应的DataFrame
        """
        result = {}
        for code in stock_codes:
            try:
                df = self.get_stock_data(code, start_date, end_date, use_cache)
                result[code] = df
            except Exception as e:
                print(f"Warning: 获取股票 {code} 数据失败: {e}")
        return result


# 全局数据服务实例
data_service = DataService()
