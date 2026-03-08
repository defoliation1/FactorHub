"""
策略回测API路由
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.services.vectorbt_backtest_service import VectorBTBacktestService, check_vectorbt_available
from backend.repositories.backtest_repository import BacktestRepository
from backend.repositories.factor_repository import FactorRepository
from backend.core.database import get_db_session

router = APIRouter()


# ========== 数据模型 ==========

class SingleBacktestRequest(BaseModel):
    """单策略回测请求"""
    data_mode: str = "single"  # single 或 pool
    stock_codes: List[str]
    factor_name: str
    strategy_type: str = "single_factor"  # single_factor 或 multi_factor
    start_date: str
    end_date: str
    initial_capital: float = 1000000
    commission_rate: float = 0.0003
    slippage: float = 0.0
    percentile: int = 50
    direction: str = "long"
    n_quantiles: int = 5
    weight_method: str = "equal_weight"  # 多因子时的权重方法


class ComparisonRequest(BaseModel):
    """策略对比请求"""
    data_mode: str = "single"
    stock_codes: List[str]
    strategies: List[Dict]  # 策略配置列表
    start_date: str
    end_date: str
    initial_capital: float = 1000000
    commission_rate: float = 0.0003
    rebalance_freq: str = "monthly"


# ========== API端点 ==========

@router.post("/single")
async def run_single_backtest(request: SingleBacktestRequest):
    """运行单策略回测"""
    try:
        if not check_vectorbt_available():
            raise HTTPException(
                status_code=503,
                detail="VectorBT未安装，请先安装: pip install vectorbt"
            )

        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db_session
        import pandas as pd

        # 从数据库获取因子定义
        db = get_db_session()
        repo = FactorRepository(db)
        factor_def = repo.get_by_name(request.factor_name)
        db.close()

        if not factor_def:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        # 获取数据
        all_factor_data = {}
        for stock_code in request.stock_codes:
            stock_data = data_service.get_stock_data(
                stock_code,
                request.start_date,
                request.end_date
            )

            if stock_data is not None and len(stock_data) > 0:
                # 计算因子（使用因子的代码）
                factor_calculator = factor_service.calculator
                factor_values = factor_calculator.calculate(
                    stock_data, factor_def.code  # 使用 factor_def.code 而不是 request.factor_name
                )
                stock_data[request.factor_name] = factor_values
                all_factor_data[stock_code] = stock_data

        if not all_factor_data:
            raise HTTPException(status_code=404, detail="未获取到有效数据")

        # 创建回测服务
        backtest_service = VectorBTBacktestService(
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate,
            slippage=request.slippage,
        )

        is_single_stock = len(all_factor_data) == 1

        # 执行回测
        if request.strategy_type == "single_factor":
            if is_single_stock:
                df = list(all_factor_data.values())[0].copy()
                result = backtest_service.single_factor_backtest(
                    df=df,
                    factor_name=request.factor_name,
                    percentile=request.percentile,
                    direction=request.direction,
                    n_quantiles=request.n_quantiles,
                )
            else:
                # 横截面回测
                merged_list = []
                for code, data in all_factor_data.items():
                    data_copy = data.copy()
                    data_copy["stock_code"] = code
                    if "date" not in data_copy.columns:
                        data_copy = data_copy.reset_index()
                    merged_list.append(data_copy)

                df = pd.concat(merged_list, ignore_index=True)
                top_percentile = (100 - request.percentile) / 100.0
                result = backtest_service.cross_sectional_backtest(
                    df=df,
                    factor_name=request.factor_name,
                    top_percentile=top_percentile,
                    direction=request.direction,
                )
        else:
            # 多因子回测
            df = list(all_factor_data.values())[0].copy()
            result = backtest_service.multi_factor_backtest(
                df=df,
                factor_names=[request.factor_name],
                method=request.weight_method,
                percentile=request.percentile,
                direction=request.direction,
            )

        # 提取指标
        metrics = {k: v for k, v in result.items() if k in [
            "total_return", "annual_return", "volatility", "sharpe_ratio",
            "max_drawdown", "calmar_ratio", "win_rate", "sortino_ratio",
            "var_95", "cvar_95"
        ]}

        # 转换 pandas Series 为列表，以便 JSON 序列化
        result_serializable = {}
        for k, v in result.items():
            if hasattr(v, 'tolist'):  # pandas Series or numpy array
                result_serializable[k] = v.tolist()
            elif k == "trades" and v is not None:
                # 转换 DataFrame 为字典列表
                result_serializable[k] = v.to_dict('records')
            else:
                result_serializable[k] = v

        return {
            "success": True,
            "data": {
                "metrics": metrics,
                "result": result_serializable
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/comparison")
async def run_strategy_comparison(request: ComparisonRequest):
    """运行策略对比"""
    try:
        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        import pandas as pd
        import numpy as np

        # 获取数据
        all_data = {}
        for stock_code in request.stock_codes:
            data = data_service.get_stock_data(
                stock_code,
                request.start_date,
                request.end_date
            )
            if not data.empty:
                all_data[stock_code] = data

        if not all_data:
            raise HTTPException(status_code=404, detail="未获取到任何数据")

        # 合并数据
        data_frames = []
        for stock_code, data in all_data.items():
            df_copy = data.copy()
            df_copy['stock_code'] = stock_code
            df_copy = df_copy.reset_index()
            data_frames.append(df_copy)

        merged_data = pd.concat(data_frames, ignore_index=True)

        # 确保有return列
        if 'return' not in merged_data.columns and 'close' in merged_data.columns:
            merged_data = merged_data.sort_values(['stock_code', 'date'])
            merged_data['return'] = merged_data.groupby('stock_code')['close'].pct_change().shift(-1)

        merged_data = merged_data.sort_values(["date", "stock_code"])

        # 计算每个策略的收益
        results = {}
        for config in request.strategies:
            strategy_name = config.get("name", "未命名策略")

            # 从数据库获取因子定义
            db = get_db_session()
            repo = FactorRepository(db)
            factor_def = repo.get_by_name(config["factor"])
            db.close()

            if not factor_def:
                continue

            # 计算因子值（使用因子代码）
            factor_values = factor_service.calculator.calculate(
                merged_data, factor_def.code
            )

            if factor_values is not None:
                # 创建因子DataFrame
                factor_df = merged_data.copy()
                factor_df["factor_value"] = factor_values

                # 按日期分组，选择top N股票
                selected_returns = []
                for date, group in factor_df.groupby("date"):
                    # 计算选择的股票数量 - 确保至少选择1只
                    top_pct = config.get("top_pct", 20) / 100
                    n_select = max(1, int(len(group) * top_pct))

                    if config.get("direction", "long") == "long":
                        top_stocks = group.nlargest(n_select, "factor_value")
                    else:
                        top_stocks = group.nsmallest(n_select, "factor_value")

                    # 计算下期收益
                    if "stock_code" in merged_data.columns:
                        next_returns = merged_data[
                            (merged_data["date"] > date) &
                            (merged_data["stock_code"].isin(top_stocks["stock_code"]))
                        ].groupby("stock_code")["return"].first()

                        if len(next_returns) > 0:
                            avg_return = next_returns.mean()
                            selected_returns.append(avg_return)

                # 构建策略收益序列
                if selected_returns:
                    returns_series = pd.Series(selected_returns)
                    total_return = float((1 + returns_series).prod() - 1)
                    n_days = len(returns_series)

                    # 使用复利计算年化收益率，与单策略回测保持一致
                    if n_days > 0:
                        annual_return = float((1 + total_return) ** (252 / n_days) - 1)
                    else:
                        annual_return = 0.0

                    results[strategy_name] = {
                        "returns": returns_series.tolist(),
                        "total_return": total_return,
                        "annual_return": annual_return,
                        "volatility": float(returns_series.std() * np.sqrt(252)),
                    }

                    # 计算夏普比率
                    if results[strategy_name]["volatility"] > 0:
                        results[strategy_name]["sharpe_ratio"] = (
                            results[strategy_name]["annual_return"] /
                            results[strategy_name]["volatility"]
                        )
                    else:
                        results[strategy_name]["sharpe_ratio"] = 0.0

        # 清理结果中的NaN和inf值，确保JSON可序列化
        def clean_value(v):
            if isinstance(v, float):
                if np.isnan(v) or np.isinf(v):
                    return 0.0
            return v

        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_value(v) for v in d]
            else:
                return clean_value(d)

        results_clean = clean_dict(results)

        return {
            "success": True,
            "data": {
                "results": results_clean
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_backtest_history(limit: int = 10):
    """获取回测历史"""
    try:
        repo = BacktestRepository()
        history = repo.get_history(limit=limit)

        # 转换为字典列表
        history_list = []
        for record in history:
            history_list.append({
                "id": record.id,
                "strategy_name": record.strategy_name,
                "factor_combination": record.factor_combination,
                "start_date": record.start_date,
                "end_date": record.end_date,
                "total_return": record.total_return,
                "sharpe_ratio": record.sharpe_ratio,
                "max_drawdown": record.max_drawdown,
                "created_at": record.created_at.isoformat()
            })

        return {
            "success": True,
            "data": history_list,
            "total": len(history_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{record_id}")
async def delete_backtest_history(record_id: int):
    """删除回测历史"""
    try:
        repo = BacktestRepository()
        success = repo.delete_by_id(record_id)

        if not success:
            raise HTTPException(status_code=404, detail="记录不存在或删除失败")

        return {
            "success": True,
            "message": "删除成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
