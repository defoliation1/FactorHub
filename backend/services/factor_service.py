"""
因子服务模块 - 因子计算与管理
"""
import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml

from backend.core.database import get_db_session
from backend.core.settings import settings
from backend.models.factor import FactorModel
from backend.repositories.factor_repository import FactorRepository
from backend.services.data_service import data_service


class FactorCalculator:
    """因子计算器 - 执行因子计算逻辑"""

    def __init__(self):
        self.talib_funcs = {
            "SMA": talib.SMA,
            "EMA": talib.EMA,
            "RSI": talib.RSI,
            "MACD": talib.MACD,
            "ADX": talib.ADX,
            "CCI": talib.CCI,
            "ATR": talib.ATR,
            "BBANDS": talib.BBANDS,
            "OBV": talib.OBV,
        }

    def calculate(self, df: pd.DataFrame, factor_code: str) -> pd.Series:
        """
        计算单个因子

        Args:
            df: 包含OHLCV数据的DataFrame
            factor_code: 因子计算代码（支持表达式或函数形式）

        Returns:
            因子值的Series
        """
        import pandas as pd

        # 检测是否是函数形式
        is_function = factor_code.strip().startswith("def ")

        if is_function:
            # 函数形式：使用 exec 执行函数定义，然后调用函数
            # 提供完整的全局变量，包括 pandas, numpy 和常见函数
            global_vars = {
                "__builtins__": {
                    "__import__": __import__,
                    "abs": abs, "min": min, "max": max, "len": len,
                    "range": range, "float": float, "int": int, "bool": bool,
                    "list": list, "tuple": tuple, "dict": dict, "sum": sum,
                    "any": any, "all": all, "enumerate": enumerate, "zip": zip,
                    "round": round, "pow": pow, "divmod": divmod,
                },
                "pd": pd,
                "np": np,
                **self.talib_funcs,
            }

            local_vars = {}

            try:
                # 打印调试信息
                print(f"[DEBUG calculate] is_function: {is_function}")
                print(f"[DEBUG calculate] code length: {len(factor_code)}")
                print(f"[DEBUG calculate] first line: {repr(factor_code.split(chr(10))[0])}")

                # 执行函数定义
                exec(factor_code, global_vars, local_vars)

                # 调用函数（函数可能在 global_vars 或 local_vars 中）
                calc_func = local_vars.get("calculate_factor") or global_vars.get("calculate_factor")
                if calc_func is None:
                    raise ValueError("函数代码中未找到 'calculate_factor' 函数定义")

                result = calc_func(df)

                if isinstance(result, pd.DataFrame):
                    # 如果返回DataFrame，取第一列
                    result = result.iloc[:, 0]
                elif not isinstance(result, pd.Series):
                    raise ValueError(f"函数必须返回 pd.Series，实际返回了 {type(result)}")

                return result
            except Exception as e:
                import traceback
                print(f"[DEBUG calculate] Error in exec:")
                print(f"[DEBUG calculate] factor_code:\n{repr(factor_code)}")
                traceback.print_exc()
                raise ValueError(f"因子计算失败: {e}")
        else:
            # 表达式形式：使用 eval 执行（保持向后兼容）
            local_vars = {
                "df": df,
                "open": df["open"],
                "high": df["high"],
                "low": df["low"],
                "close": df["close"],
                "volume": df["volume"],
                "np": np,
                **self.talib_funcs,
            }

            try:
                result = eval(factor_code, {"__builtins__": {}}, local_vars)
                if isinstance(result, pd.DataFrame):
                    # 如果返回DataFrame，取第一列
                    result = result.iloc[:, 0]
                return result
            except Exception as e:
                raise ValueError(f"因子计算失败: {e}")

    def calculate_multiple(
        self, df: pd.DataFrame, factors: List[FactorModel]
    ) -> pd.DataFrame:
        """
        计算多个因子

        Args:
            df: 包含OHLCV数据的DataFrame
            factors: 因子模型列表

        Returns:
            包含所有因子值的DataFrame
        """
        result = pd.DataFrame(index=df.index)

        for factor in factors:
            try:
                factor_values = self.calculate(df, factor.code)
                result[factor.name] = factor_values
            except Exception as e:
                print(f"Warning: 计算因子 {factor.name} 失败: {e}")
                result[factor.name] = np.nan

        return result

    def rolling_standardize(self, df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        滚动窗口标准化

        Args:
            df: 因子数据DataFrame
            window: 滚动窗口大小

        Returns:
            标准化后的DataFrame
        """
        result = df.copy()
        for col in df.columns:
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window, min_periods=1).std()
            result[col] = (df[col] - rolling_mean) / rolling_std
        return result

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加时间特征

        Args:
            df: 包含日期索引的DataFrame

        Returns:
            添加时间特征后的DataFrame
        """
        result = df.copy()
        if isinstance(result.index, pd.DatetimeIndex):
            result["day_of_week"] = result.index.dayofweek
            result["month"] = result.index.month
            result["quarter"] = result.index.quarter
        return result


class FactorService:
    """因子服务类"""

    def __init__(self):
        self.calculator = FactorCalculator()

    def load_preset_factors(self) -> None:
        """从配置文件加载预置因子"""
        config_path = settings.CONFIG_DIR / "factors.yaml"

        if not config_path.exists():
            # 如果配置文件不存在，创建默认预置因子
            self._create_default_preset_factors()
            return

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 如果配置文件为空或加载失败，创建默认因子
        if config is None:
            self._create_default_preset_factors()
            return

        db = get_db_session()
        repo = FactorRepository(db)

        for category, factors in config.items():
            for factor_data in factors:
                # 检查因子是否已存在
                existing = repo.get_by_name(factor_data["name"])
                if existing:
                    continue

                factor = FactorModel(
                    name=factor_data["name"],
                    code=factor_data["code"],
                    description=factor_data.get("description", ""),
                    source="preset",
                    category=category,
                    is_active=1,
                )
                repo.create(factor)

        db.close()

    def _create_default_preset_factors(self) -> None:
        """创建默认预置因子"""
        preset_factors = self._get_default_factors()
        db = get_db_session()
        repo = FactorRepository(db)

        for category, factors in preset_factors.items():
            for factor_data in factors:
                existing = repo.get_by_name(factor_data["name"])
                if existing:
                    continue
                factor = FactorModel(
                    name=factor_data["name"],
                    code=factor_data["code"],
                    description=factor_data.get("description", ""),
                    source="preset",
                    category=category,
                    is_active=1,
                )
                repo.create(factor)

        db.close()

    def _get_default_factors(self) -> Dict[str, List[Dict]]:
        """获取默认预置因子定义"""
        return {
            "价格收益率": [
                {
                    "name": "log_return_1",
                    "code": "np.log(close / close.shift(1))",
                    "description": "日对数收益率",
                },
                {
                    "name": "log_return_5",
                    "code": "np.log(close / close.shift(5))",
                    "description": "5日累计收益",
                },
                {
                    "name": "price_vs_sma20",
                    "code": "close / SMA(close, timeperiod=20)",
                    "description": "相对20日均线位置",
                },
                {
                    "name": "price_vs_sma60",
                    "code": "close / SMA(close, timeperiod=60)",
                    "description": "相对60日均线位置",
                },
                {
                    "name": "sma20_vs_sma60",
                    "code": "SMA(close, timeperiod=20) / SMA(close, timeperiod=60)",
                    "description": "短期vs长期趋势方向",
                },
                {
                    "name": "high_low_ratio",
                    "code": "(high - low) / open",
                    "description": "日内波动幅度",
                },
                {
                    "name": "close_open_ratio",
                    "code": "close / open",
                    "description": "收盘相对开盘强度",
                },
            ],
            "动量趋势": [
                {
                    "name": "rsi_14",
                    "code": "RSI(close, timeperiod=14)",
                    "description": "RSI(14) 超买超卖指标",
                },
                {
                    "name": "macd_line",
                    "code": "MACD(close, fastperiod=12, slowperiod=26)[0]",
                    "description": "MACD差值线",
                },
                {
                    "name": "macd_signal",
                    "code": "MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[1]",
                    "description": "MACD信号线",
                },
                {
                    "name": "macd_hist",
                    "code": "MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[2]",
                    "description": "MACD柱状图",
                },
                {
                    "name": "adx_14",
                    "code": "ADX(high, low, close, timeperiod=14)",
                    "description": "趋势强度指标",
                },
                {
                    "name": "cci_20",
                    "code": "CCI(high, low, close, timeperiod=20)",
                    "description": "通道突破信号",
                },
                {
                    "name": "roc_10",
                    "code": "(close - close.shift(10)) / close.shift(10)",
                    "description": "10日变化率",
                },
            ],
            "波动率风险": [
                {
                    "name": "atr_14",
                    "code": "ATR(high, low, close, timeperiod=14)",
                    "description": "平均真实波幅",
                },
                {
                    "name": "atr_norm",
                    "code": "ATR(high, low, close, timeperiod=14) / close",
                    "description": "波动率相对价格水平",
                },
                {
                    "name": "volatility_10",
                    "code": "np.log(close / close.shift(1)).rolling(window=10).std()",
                    "description": "近10日收益率标准差",
                },
                {
                    "name": "bollinger_bandwidth",
                    "code": "(BBANDS(close, timeperiod=20)[0] - BBANDS(close, timeperiod=20)[2]) / BBANDS(close, timeperiod=20)[1]",
                    "description": "布林带宽度",
                },
                {
                    "name": "bollinger_position",
                    "code": "(close - BBANDS(close, timeperiod=20)[2]) / (BBANDS(close, timeperiod=20)[0] - BBANDS(close, timeperiod=20)[2])",
                    "description": "价格在布林带中的相对位置",
                },
            ],
            "成交量资金流": [
                {
                    "name": "volume_ma_ratio",
                    "code": "volume / SMA(volume, timeperiod=10)",
                    "description": "当日量能vs近期均量",
                },
                {
                    "name": "obv",
                    "code": "OBV(close, volume)",
                    "description": "累积量能趋势",
                },
                {
                    "name": "obv_slope",
                    "code": "OBV(close, volume) - OBV(close, volume).shift(5)",
                    "description": "OBV近5日斜率",
                },
            ],
            "结构模式": [
                {
                    "name": "is_bullish_candle",
                    "code": "(close > open).astype(int)",
                    "description": "是否阳线",
                },
                {
                    "name": "regime_volatility",
                    "code": "(np.log(close / close.shift(1)).rolling(window=10).std() > np.log(close / close.shift(1)).rolling(window=10).std().quantile(0.7)).astype(int)",
                    "description": "高波动regime标记",
                },
                {
                    "name": "regime_trend",
                    "code": "((ADX(high, low, close, timeperiod=14) > 25) & (SMA(close, timeperiod=20) > SMA(close, timeperiod=60))).astype(int)",
                    "description": "趋势市标记",
                },
            ],
        }

    def get_all_factors(self) -> List[Dict]:
        """获取所有因子"""
        db = get_db_session()
        repo = FactorRepository(db)
        factors = repo.get_all(active_only=True)
        db.close()
        return [f.to_dict() for f in factors]

    def get_factor_stats(self) -> Dict[str, int]:
        """获取因子统计信息"""
        db = get_db_session()
        repo = FactorRepository(db)
        stats = {
            "preset_count": repo.get_preset_count(),
            "user_count": repo.get_user_count(),
            "total_count": repo.get_preset_count() + repo.get_user_count(),
        }
        db.close()
        return stats

    def create_factor(
        self, name: str, code: str, description: str = ""
    ) -> Dict:
        """创建用户自定义因子"""
        db = get_db_session()
        repo = FactorRepository(db)

        # 检查名称是否已存在
        if repo.get_by_name(name):
            db.close()
            raise ValueError(f"因子名称 '{name}' 已存在")

        factor = FactorModel(
            name=name,
            code=code,
            description=description,
            source="user",
            category="自定义",
            is_active=1,
        )
        result = repo.create(factor)
        db.close()
        return result.to_dict()

    def update_factor(
        self, factor_id: int, name: str = None, code: str = None, description: str = None
    ) -> Dict:
        """更新因子"""
        db = get_db_session()
        repo = FactorRepository(db)
        factor = repo.get_by_id(factor_id)

        if not factor:
            db.close()
            raise ValueError(f"因子ID {factor_id} 不存在")

        if factor.source == "preset" and (name or code):
            db.close()
            raise ValueError("预置因子的名称和代码不能修改")

        if name:
            factor.name = name
        if code:
            factor.code = code
        if description is not None:
            factor.description = description

        result = repo.update(factor)
        db.close()
        return result.to_dict()

    def delete_factor(self, factor_id: int) -> bool:
        """删除因子"""
        db = get_db_session()
        repo = FactorRepository(db)
        try:
            result = repo.delete(factor_id)
            db.close()
            return result
        except ValueError as e:
            db.close()
            raise e

    def validate_factor_code(self, code: str) -> tuple[bool, str]:
        """验证因子代码"""
        # 打印调试信息
        print(f"[DEBUG] Validating factor code...")
        print(f"[DEBUG] Code length: {len(code)}")
        print(f"[DEBUG] Code starts with 'def': {code.strip().startswith('def ')}")
        print(f"[DEBUG] First 200 chars: {repr(code[:200])}")

        # 创建测试数据
        test_df = pd.DataFrame({
            "open": [10.0] * 100,
            "high": [11.0] * 100,
            "low": [9.0] * 100,
            "close": [10.5] * 100,
            "volume": [1000000] * 100,
        })

        try:
            calculator = FactorCalculator()
            result = calculator.calculate(test_df, code)
            if result is None or len(result) == 0:
                return False, "代码未返回任何结果"
            return True, "验证通过"
        except Exception as e:
            import traceback
            print(f"[DEBUG] Error traceback:")
            traceback.print_exc()
            return False, f"验证失败: {str(e)}"

    def calculate_factors_for_stock(
        self,
        stock_code: str,
        factor_names: List[str],
        start_date: str,
        end_date: str,
        rolling_window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        为单个股票计算因子

        Args:
            stock_code: 股票代码
            factor_names: 因子名称列表
            start_date: 开始日期
            end_date: 结束日期
            rolling_window: 滚动标准化窗口大小

        Returns:
            包含因子值的DataFrame
        """
        # 获取股票数据
        df = data_service.get_stock_data(stock_code, start_date, end_date)

        # 获取因子定义
        db = get_db_session()
        repo = FactorRepository(db)
        factors = []
        for name in factor_names:
            factor = repo.get_by_name(name)
            if factor:
                factors.append(factor)
        db.close()

        if not factors:
            raise ValueError("未找到有效的因子")

        # 计算因子
        factor_df = self.calculator.calculate_multiple(df, factors)

        # 滚动标准化
        if rolling_window:
            factor_df = self.calculator.rolling_standardize(factor_df, rolling_window)

        # 添加时间特征
        factor_df = self.calculator.add_time_features(factor_df)

        # 合并原始数据
        result = pd.concat([df, factor_df], axis=1)

        return result

    def calculate_factors_for_stocks(
        self,
        stock_codes: List[str],
        factor_names: List[str],
        start_date: str,
        end_date: str,
        rolling_window: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """为多个股票计算因子"""
        results = {}
        for code in stock_codes:
            try:
                result = self.calculate_factors_for_stock(
                    code, factor_names, start_date, end_date, rolling_window
                )
                results[code] = result
            except Exception as e:
                print(f"Warning: 为股票 {code} 计算因子失败: {e}")
        return results


# 全局因子服务实例
factor_service = FactorService()
