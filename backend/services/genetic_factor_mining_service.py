"""
遗传算法因子挖掘服务 - 使用遗传算法自动发现最优因子
"""
import logging
from typing import List, Dict, Callable, Optional
import pandas as pd
import numpy as np
import random

# 配置日志
logger = logging.getLogger(__name__)

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    logger.warning("DEAP库未安装，遗传算法功能将不可用")

from backend.services.factor_generator_service import factor_generator_service
from backend.services.factor_validation_service import factor_validation_service


class GeneticFactorMiningService:
    """遗传算法因子挖掘服务"""

    def __init__(
        self,
        base_factors: List[str],
        data: pd.DataFrame,
        return_column: str = "return",
        population_size: int = 50,
        n_generations: int = 20,
        cx_prob: float = 0.7,
        mut_prob: float = 0.3,
    ):
        """
        初始化遗传算法挖掘服务

        Args:
            base_factors: 基础因子列表
            data: 数据DataFrame
            return_column: 收益率列名
            population_size: 种群大小
            n_generations: 迭代代数
            cx_prob: 交叉概率
            mut_prob: 变异概率
        """
        if not DEAP_AVAILABLE:
            raise ImportError("DEAP库未安装，请运行: pip install DEAP")

        self.base_factors = base_factors
        self.data = data
        self.return_column = return_column
        self.population_size = population_size
        self.n_generations = n_generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob

        # 准备收益率数据
        self.return_values = data[return_column] if return_column in data.columns else None

        # 初始化遗传算法
        self._setup_genetic_algorithm()

    def _setup_genetic_algorithm(self):
        """设置遗传算法"""
        # 定义适应度函数（最大化IC绝对值和IR）
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # 定义个体（因子表达式）
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # 创建工具箱
        self.toolbox = base.Toolbox()

        # 注册个体生成函数
        self.toolbox.register(
            "individual",
            self._generate_random_individual,
        )

        # 注册种群生成函数
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,
        )

        # 注册遗传操作
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # 注册评估函数
        self.toolbox.register("evaluate", self._evaluate_factor)

        # 统计信息
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def _generate_random_individual(self):
        """生成随机个体（因子表达式）"""
        # 使用因子生成器生成混合因子
        factors = factor_generator_service.generate_hybrid_factors(
            self.base_factors,
            n_factors=1
        )

        if factors:
            expr = factors[0]["expression"]
            individual = creator.Individual()
            individual.extend([expr])
            return individual
        else:
            # 如果生成失败，返回简单的二元运算
            if len(self.base_factors) >= 2:
                factor1, factor2 = random.sample(self.base_factors, 2)
                op = random.choice(["+", "-", "*", "/"])
                individual = creator.Individual()
                individual.extend([f"({factor1} {op} {factor2})"])
                return individual
            else:
                individual = creator.Individual()
                individual.extend([self.base_factors[0]])
                return individual

    def _evaluate_factor(self, individual: list) -> tuple:
        """
        评估因子适应度

        Args:
            individual: 个体（因子表达式列表）

        Returns:
            适应度值元组
        """
        expr = individual[0]

        # 尝试计算因子值
        try:
            factor_values = self._compute_factor_expression(expr)

            if factor_values is None or len(factor_values.dropna()) < 10:
                return (0.0,)

            # 验证因子
            if self.return_values is not None:
                validation = factor_validation_service.validate_factor(
                    factor_values=factor_values,
                    return_values=self.return_values,
                    existing_factors=None,
                )

                # 适应度 = 综合得分
                fitness = validation["score"] / 100.0
            else:
                # 如果没有收益率数据，使用因子的标准差作为适应度
                fitness = factor_values.std() / (factor_values.mean() + 1e-8)

            return (fitness,)

        except Exception as e:
            return (0.0,)

    def _compute_factor_expression(self, expr: str) -> Optional[pd.Series]:
        """
        计算因子表达式的值

        Args:
            expr: 因子表达式

        Returns:
            因子值序列
        """
        try:
            # 构建安全的执行环境
            safe_dict = {}

            # 添加基础因子数据到环境
            for factor_name in self.base_factors:
                if factor_name in self.data.columns:
                    safe_dict[factor_name] = self.data[factor_name]

            # 如果没有数据，返回None
            if not safe_dict:
                return None

            # 使用pandas.eval进行安全计算（比eval更安全）
            try:
                result = pd.eval(expr, local_dict=safe_dict)
                # 确保返回的是Series
                if isinstance(result, pd.Series):
                    return result
                elif isinstance(result, (int, float)):
                    # 如果是标量值，返回与数据长度相同的Series
                    return pd.Series([result] * len(self.data), index=self.data.index)
                else:
                    return None
            except Exception:
                # 如果pandas.eval失败，尝试简单的二元运算
                return self._compute_binary_operation(expr)

        except Exception as e:
            logger.warning(f"计算因子表达式失败 {expr}: {e}")
            return None

    def _compute_binary_operation(self, expr: str) -> Optional[pd.Series]:
        """
        计算简单的二元运算表达式（备用方法）

        Args:
            expr: 因子表达式

        Returns:
            因子值序列
        """
        try:
            # 去除外层括号
            expr = expr.strip()
            if expr.startswith("(") and expr.endswith(")"):
                expr = expr[1:-1].strip()

            # 尝试匹配二元运算模式: factor1 op factor2
            import re
            pattern = r'^(\w+)\s*([+\-*/])\s*(\w+)$'
            match = re.match(pattern, expr)

            if match:
                left_factor = match.group(1)
                operator = match.group(2)
                right_factor = match.group(3)

                left = self._get_factor_value(left_factor)
                right = self._get_factor_value(right_factor)

                if left is not None and right is not None:
                    # 对齐索引
                    aligned_data = pd.DataFrame({
                        'left': left,
                        'right': right
                    }).dropna()

                    if len(aligned_data) == 0:
                        return None

                    if operator == '+':
                        result = aligned_data['left'] + aligned_data['right']
                    elif operator == '-':
                        result = aligned_data['left'] - aligned_data['right']
                    elif operator == '*':
                        result = aligned_data['left'] * aligned_data['right']
                    elif operator == '/':
                        result = aligned_data['left'] / (aligned_data['right'] + 1e-8)
                    else:
                        return None

                    return result

            # 如果无法解析为二元运算，尝试直接获取因子值
            return self._get_factor_value(expr)

        except Exception as e:
            logger.warning(f"计算二元运算失败 {expr}: {e}")
            return None

    def _get_factor_value(self, factor_name: str) -> Optional[pd.Series]:
        """获取因子值"""
        # 去除空格
        factor_name = factor_name.strip()

        # 检查是否是基础因子
        if factor_name in self.data.columns:
            return self.data[factor_name]

        # 检查是否是技术指标
        if "SMA" in factor_name or "EMA" in factor_name or "RSI" in factor_name:
            # 简化处理：返回close列
            if "close" in self.data.columns:
                return self.data["close"]

        # 如果都不匹配，返回None
        return None

    def _extract_inner_expression(self, expr: str) -> str:
        """提取最内层的括号表达式"""
        # 找到第一个完整的括号对
        start = expr.find("(")
        if start == -1:
            return expr

        count = 1
        end = start + 1
        while end < len(expr) and count > 0:
            if expr[end] == "(":
                count += 1
            elif expr[end] == ")":
                count -= 1
            end += 1

        return expr[start + 1:end - 1]

    def _split_binary_operation(self, expr: str) -> List[str]:
        """分割二元运算表达式"""
        operators = ["+", "-", "*", "/"]

        for op in operators:
            if op in expr:
                # 简单分割（实际需要更复杂的解析）
                parts = expr.split(op)
                if len(parts) == 2:
                    return [p.strip() for p in parts]

        return []

    def _crossover(self, ind1, ind2):
        """交叉操作"""
        # 由于每个Individual只有一个表达式元素，我们简单地交换表达式中的因子
        expr1 = ind1[0]
        expr2 = ind2[0]

        # 提取因子并交换
        factors1 = [f for f in self.base_factors if f in expr1]
        factors2 = [f for f in self.base_factors if f in expr2]

        if factors1 and factors2 and random.random() < 0.7:
            # 70%概率交换因子
            factor1 = random.choice(factors1)
            factor2 = random.choice(factors2)

            new_expr1 = expr1.replace(factor1, factor2)
            new_expr2 = expr2.replace(factor2, factor1)

            # 创建新的Individual对象
            child1 = creator.Individual()
            child1.extend([new_expr1])
            child2 = creator.Individual()
            child2.extend([new_expr2])
            return (child1, child2)
        else:
            # 30%概率直接交换整个表达式
            return (ind2, ind1)

    def _mutate(self, individual, indpb: float) -> tuple:
        """变异操作"""
        expr = individual[0]

        # 可能的变异操作：
        # 1. 更换运算符
        # 2. 更换因子
        # 3. 添加统计函数

        if random.random() < 0.3:
            # 更换运算符
            operators = ["+", "-", "*", "/"]
            for op in operators:
                if op in expr and random.random() < indpb:
                    new_op = random.choice([o for o in operators if o != op])
                    expr = expr.replace(op, new_op, 1)

        if random.random() < 0.3:
            # 更换因子
            for factor in self.base_factors:
                if factor in expr and random.random() < indpb:
                    new_factor = random.choice(
                        [f for f in self.base_factors if f != factor]
                    )
                    expr = expr.replace(factor, new_factor, 1)

        if random.random() < 0.2:
            # 添加统计函数
            stats = ["rank", "zscore", "mean", "std"]
            stat = random.choice(stats)
            # 简化版：不实际添加，只在记录中标记
            pass

        # 创建新的Individual对象并返回
        mutated = creator.Individual()
        mutated.extend([expr])
        return (mutated,)

    def mine_factors(self) -> Dict:
        """
        执行因子挖掘

        Returns:
            挖掘结果
        """
        if not DEAP_AVAILABLE:
            return {
                "success": False,
                "message": "DEAP库未安装",
                "best_factors": [],
            }

        logger.info(f"开始遗传算法因子挖掘...")
        logger.info(f"种群大小: {self.population_size}")
        logger.info(f"迭代代数: {self.n_generations}")

        # 初始化种群
        population = self.toolbox.population(n=self.population_size)

        # 创建Hall of Fame保存最优个体
        halloffame = tools.HallOfFame(10)

        # 运行遗传算法
        population, logbook = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=self.cx_prob,
            mutpb=self.mut_prob,
            ngen=self.n_generations,
            stats=self.stats,
            halloffame=halloffame,
            verbose=True,
        )

        # 提取最优因子
        best_factors = []
        for i, individual in enumerate(halloffame):
            factor_info = {
                "rank": i + 1,
                "expression": individual[0],
                "fitness": float(individual.fitness.values[0]),
            }

            # 重新计算详细指标
            try:
                factor_values = self._compute_factor_expression(individual[0])
                if factor_values is not None and self.return_values is not None:
                    validation = factor_validation_service.validate_factor(
                        factor_values=factor_values,
                        return_values=self.return_values,
                    )
                    factor_info["validation"] = validation
            except Exception as e:
                # 记录个体评估失败的异常，但继续处理其他个体
                import logging
                logging.getLogger(__name__).warning(f"因子个体评估失败: {e}")

            best_factors.append(factor_info)

        return {
            "success": True,
            "best_factors": best_factors,
            "logbook": logbook,
            "final_population": population,
        }

    def evolve_factor(
        self,
        initial_expression: str,
        n_generations: int = 10,
    ) -> Dict:
        """
        基于初始表达式进化优化

        Args:
            initial_expression: 初始因子表达式
            n_generations: 进化代数

        Returns:
            进化结果
        """
        if not DEAP_AVAILABLE:
            return {
                "success": False,
                "message": "DEAP库未安装",
            }

        # 创建初始种群
        population = [self._generate_random_individual() for _ in range(self.population_size - 1)]
        # 将初始表达式转换为Individual对象
        initial_individual = creator.Individual()
        initial_individual.extend([initial_expression])
        population.insert(0, initial_individual)

        # 创建Hall of Fame
        halloffame = tools.HallOfFame(5)

        # 运行进化
        population, logbook = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=self.cx_prob,
            mutpb=self.mut_prob,
            ngen=n_generations,
            stats=self.stats,
            halloffame=halloffame,
            verbose=False,
        )

        # 返回最优个体
        best = halloffame[0]

        return {
            "success": True,
            "original_expression": initial_expression,
            "evolved_expression": best[0],
            "original_fitness": self._evaluate_factor([initial_expression])[0],
            "evolved_fitness": float(best.fitness.values[0]),
            "improvement": float(best.fitness.values[0]) - self._evaluate_factor([initial_expression])[0],
        }


# 全局遗传算法挖掘服务实例（需要初始化参数）
def create_genetic_mining_service(
    base_factors: List[str],
    data: pd.DataFrame,
    **kwargs
) -> GeneticFactorMiningService:
    """
    创建遗传算法挖掘服务

    Args:
        base_factors: 基础因子列表
        data: 数据
        **kwargs: 其他参数

    Returns:
        遗传算法挖掘服务实例
    """
    return GeneticFactorMiningService(
        base_factors=base_factors,
        data=data,
        **kwargs
    )
