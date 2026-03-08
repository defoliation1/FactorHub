"""
因子分析API路由
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.services.analysis_service import analysis_service
from backend.services.factor_stability_service import factor_stability_service
from backend.services.enhanced_analysis_service import enhanced_analysis_service

router = APIRouter()


# ========== 数据模型 ==========

class CalculateRequest(BaseModel):
    """计算因子值请求"""
    factor_name: str
    stock_codes: List[str]
    start_date: str
    end_date: str


class ICAnalysisRequest(BaseModel):
    """IC分析请求"""
    factor_name: str
    stock_codes: List[str]
    start_date: str
    end_date: str


class StabilityRequest(BaseModel):
    """稳定性检验请求"""
    factor_name: str
    stock_codes: List[str]
    start_date: str
    end_date: str


class MultiPeriodRequest(BaseModel):
    """多周期分析请求"""
    factor_name: str
    stock_codes: List[str]
    start_date: str
    end_date: str


# ========== API端点 ==========

@router.post("/calculate")
async def calculate_factor(request: CalculateRequest):
    """计算因子值"""
    try:
        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db_session

        # 获取因子定义
        db = get_db_session()
        repo = FactorRepository(db)
        factor = repo.get_by_name(request.factor_name)
        db.close()

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        # 获取数据并计算因子
        result_data = {}
        for stock_code in request.stock_codes:
            data = data_service.get_stock_data(
                stock_code,
                request.start_date,
                request.end_date
            )
            if data is not None and len(data) > 0:
                # 使用 calculator 计算因子
                factor_series = factor_service.calculator.calculate(data, factor.code)
                if factor_series is not None:
                    # 将因子值添加到数据中
                    data[request.factor_name] = factor_series
                    # 转换为字典格式返回
                    result_data[stock_code] = {
                        "dates": data.index.strftime('%Y-%m-%d').tolist(),
                        "factor_values": factor_series.dropna().tolist(),
                        "statistics": {
                            "mean": float(factor_series.mean()),
                            "std": float(factor_series.std()),
                            "min": float(factor_series.min()),
                            "max": float(factor_series.max()),
                            "count": int(factor_series.count())
                        }
                    }

        if not result_data:
            raise HTTPException(status_code=500, detail="因子计算失败或无有效数据")

        return {
            "success": True,
            "data": result_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ic")
async def calculate_ic(request: ICAnalysisRequest):
    """计算IC/IR"""
    try:
        # 调用分析服务的 analyze 方法
        result = analysis_service.analyze(
            stock_codes=request.stock_codes,
            factor_names=[request.factor_name],
            start_date=request.start_date,
            end_date=request.end_date,
            use_cache=True,
            rolling_window=252
        )

        # 提取 IC/IR 相关数据并简化返回格式
        simplified_result = {
            "metadata": result.get("metadata", {}),
            "ic_stats": result.get("ic_ir", {}).get("ic_stats", {}),
        }

        return {
            "success": True,
            "data": simplified_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stability")
async def stability_test(request: StabilityRequest):
    """稳定性检验"""
    try:
        # 调用稳定性服务
        result = factor_stability_service.comprehensive_stability_test(
            factor_name=request.factor_name,
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date
        )

        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-period")
async def multi_period_analysis(request: MultiPeriodRequest):
    """多周期分析"""
    try:
        # 调用增强分析服务
        result = enhanced_analysis_service.analyze_multi_period_ic(
            factor_name=request.factor_name,
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date
        )

        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decay")
async def decay_analysis(request: ICAnalysisRequest):
    """因子衰减分析"""
    try:
        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db_session
        import pandas as pd
        import numpy as np

        # 获取因子定义
        db = get_db_session()
        repo = FactorRepository(db)
        factor = repo.get_by_name(request.factor_name)
        db.close()

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        # 计算因子在不同周期的IC（衰减分析）
        decay_periods = [1, 3, 5, 10, 20]  # 1日, 3日, 5日, 10日, 20日
        decay_results = []

        for period in decay_periods:
            all_ics = []
            for stock_code in request.stock_codes:
                data = data_service.get_stock_data(
                    stock_code,
                    request.start_date,
                    request.end_date
                )
                if data is not None and len(data) > 0:
                    # 计算因子
                    factor_series = factor_service.calculator.calculate(data, factor.code)
                    if factor_series is not None:
                        # 计算未来收益率
                        future_returns = data["close"].pct_change(period).shift(-period)
                        # 计算IC
                        ic = factor_series.rolling(20).corr(future_returns)
                        if not ic.empty and ic.dropna().count() > 0:
                            all_ics.append(ic.dropna().mean())

            if all_ics:
                mean_ic = np.mean(all_ics)
                decay_results.append({
                    "period": f"{period}日",
                    "ic_mean": float(mean_ic),
                    "period_days": period
                })

        result = {
            "factor_name": request.factor_name,
            "decay_analysis": decay_results
        }

        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
