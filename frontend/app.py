"""
FactorFlow - 因子分析系统
基于 Streamlit 的股票因子分析平台
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.settings import settings
from backend.core.database import init_db
from backend.services.factor_service import factor_service
from backend.services.analysis_service import analysis_service
from backend.services.vectorbt_backtest_service import VectorBTBacktestService, check_vectorbt_available
from backend.repositories.backtest_repository import BacktestRepository
from backend.services.data_service import data_service
from backend.services.factor_import_service import factor_import_service
from backend.services.formula_compiler_service import formula_compiler_service
from backend.services.genetic_factor_mining_service import create_genetic_mining_service


# ============ 初始化 ============
st.set_page_config(
    page_title="FactorFlow - 因子分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 初始化数据库
@st.cache_resource
def init_database():
    """初始化数据库和预置因子"""
    init_db()
    factor_service.load_preset_factors()


init_database()


# ============ 辅助函数 ============
def normalize_price_data(df_dict: dict, stock_codes: list) -> pd.DataFrame:
    """归一化价格数据（初始值为100）"""
    normalized = {}
    for code in stock_codes:
        if code in df_dict:
            df = df_dict[code]
            if "close" in df.columns and len(df) > 0:
                normalized[code] = (df["close"] / df["close"].iloc[0]) * 100
    return pd.DataFrame(normalized)


def plot_price_chart(df_dict: dict, stock_codes: list, selected_factors: list = None):
    """绘制价格走势图"""
    normalized_df = normalize_price_data(df_dict, stock_codes)

    fig = go.Figure()
    for code in stock_codes:
        if code in normalized_df.columns:
            fig.add_trace(go.Scatter(
                x=normalized_df.index,
                y=normalized_df[code],
                mode="lines",
                name=f"{code} 价格",
                line=dict(width=2),
            ))

    fig.update_layout(
        title="股票价格走势（归一化，初始100）",
        xaxis_title="日期",
        yaxis_title="相对价格",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_factor_with_price(df: pd.DataFrame, factor_name: str, stock_code: str):
    """绘制因子与价格叠加图"""
    fig = go.Figure()

    # 添加价格（归一化）
    if "close" in df.columns:
        normalized_price = (df["close"] / df["close"].iloc[0]) * 100
        fig.add_trace(go.Scatter(
            x=df.index,
            y=normalized_price,
            mode="lines",
            name="价格 (归一化)",
            line=dict(color="blue", width=2),
            yaxis="y",
        ))

    # 添加因子
    if factor_name in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[factor_name],
            mode="lines",
            name=factor_name,
            line=dict(color="orange", width=1.5),
            yaxis="y2",
        ))

    fig.update_layout(
        title=f"{stock_code} - {factor_name} 与价格走势",
        xaxis_title="日期",
        yaxis=dict(title="相对价格", side="left"),
        yaxis2=dict(title=factor_name, side="right", overlaying="y"),
        hovermode="x unified",
        height=400,
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_ic_time_series(ic_data: dict):
    """绘制IC时间序列图"""
    if not ic_data:
        return

    fig = go.Figure()
    for factor_name, ic_series in ic_data.items():
        if isinstance(ic_series, dict):
            dates = list(ic_series.keys())
            values = list(ic_series.values())
        elif hasattr(ic_series, 'index'):
            dates = ic_series.index.astype(str).tolist()
            values = ic_series.values.tolist()
        else:
            continue

        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            name=factor_name,
        ))

    fig.update_layout(
        title="IC时间序列",
        xaxis_title="日期",
        yaxis_title="IC值",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_monthly_ic_heatmap(monthly_ic_df: pd.DataFrame, factor_name: str):
    """绘制月度IC热力图"""
    if monthly_ic_df.empty:
        st.warning(f"因子 {factor_name} 没有月度IC数据")
        return

    fig = go.Figure(data=go.Heatmap(
        z=monthly_ic_df.values,
        x=["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"],
        y=monthly_ic_df.index.astype(str),
        colorscale="RdBu",
        zmid=0,
        text=monthly_ic_df.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 10},
    ))

    fig.update_layout(
        title=f"{factor_name} - 月度IC热力图",
        xaxis_title="月份",
        yaxis_title="年份",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_rolling_ir(rolling_ir: dict):
    """绘制滚动窗口IR图"""
    if not rolling_ir:
        return

    fig = go.Figure()
    for factor_name, ir_series in rolling_ir.items():
        if isinstance(ir_series, dict):
            dates = list(ir_series.keys())
            values = list(ir_series.values())
        elif hasattr(ir_series, 'index'):
            dates = ir_series.index.astype(str).tolist()
            values = ir_series.values.tolist()
        else:
            continue

        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            name=factor_name,
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="滚动窗口IR（60日）",
        xaxis_title="日期",
        yaxis_title="IR值",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_shap_importance(feature_importance: list):
    """绘制SHAP特征重要性柱状图"""
    df = pd.DataFrame(feature_importance)
    df = df.sort_values("importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
    ))

    fig.update_layout(
        title="SHAP特征重要性",
        xaxis_title="重要性（mean(|SHAP|)）",
        yaxis_title="特征名称",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_matrix(factor_data: dict, factor_names: list, stock_code: str = None):
    """绘制因子相关性矩阵热力图"""
    # 合并所有股票的因子数据
    all_dfs = []
    for code, df in factor_data.items():
        df_copy = df.copy()
        df_copy["stock_code"] = code
        all_dfs.append(df_copy)

    merged_df = pd.concat(all_dfs, ignore_index=False)

    # 选择因子列
    factor_cols = [col for col in factor_names if col in merged_df.columns]

    # 添加收益率列（如果存在）
    if "future_return_1" in merged_df.columns:
        merged_df = merged_df.copy()
        merged_df["未来收益率"] = merged_df["future_return_1"]
        factor_cols.append("未来收益率")
    elif "close" in merged_df.columns:
        # 使用价格收益率作为替代
        merged_df = merged_df.copy()
        merged_df["未来收益率"] = merged_df["close"].pct_change(1)
        factor_cols.append("未来收益率")

    if not factor_cols:
        st.warning("没有可用的因子数据")
        return

    # 计算相关性矩阵
    corr_df = merged_df[factor_cols].corr()

    # 绘制热力图
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr_df.values, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="相关系数"),
    ))

    title = f"因子相关性矩阵 ({stock_code})" if stock_code else "因子相关性矩阵 (多股票)"
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        height=500,
        width=700,
    )

    st.plotly_chart(fig, use_container_width=True)

    # 显示相关性统计说明
    st.caption("""
    💡 **说明**：相关系数范围为[-1, 1]，绝对值越接近1表示相关性越强。
    - 正值（红色）：正相关，因子值上升时另一个也上升
    - 负值（蓝色）：负相关，因子值上升时另一个下降
    - 接近0（白色）：无明显相关性
    """)


# ============ 回测可视化函数 ============

def plot_equity_curve(equity_series, benchmark_series: pd.Series = None, title: str = "净值曲线"):
    """绘制净值曲线图"""
    fig = go.Figure()

    # 策略净值
    if hasattr(equity_series, 'index'):
        x_values = equity_series.index.astype(str) if hasattr(equity_series.index, 'astype') else equity_series.index
        y_values = equity_series.values

        # 如果是2D数组（多资产情况），求和得到组合净值
        if len(y_values.shape) > 1 and y_values.shape[1] > 1:
            y_values = y_values.sum(axis=1)
    else:
        x_values = list(equity_series.keys())
        y_values = list(equity_series.values())

    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines',
        name='策略净值',
        line=dict(color='blue', width=2),
    ))

    # 基准净值（如果有）
    if benchmark_series is not None and len(benchmark_series) > 0:
        if hasattr(benchmark_series, 'index'):
            x_bench = benchmark_series.index.astype(str) if hasattr(benchmark_series.index, 'astype') else benchmark_series.index
            y_bench = benchmark_series.values
        else:
            x_bench = list(benchmark_series.keys())
            y_bench = list(benchmark_series.values())

        fig.add_trace(go.Scatter(
            x=x_bench,
            y=y_bench,
            mode='lines',
            name='基准净值',
            line=dict(color='gray', width=1, dash='dash'),
        ))

    fig.update_layout(
        title=title,
        xaxis_title='日期',
        yaxis_title='净值',
        hovermode='x unified',
        height=400,
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_drawdown_curve(equity_series, title: str = "回撤曲线"):
    """绘制回撤曲线"""
    # 计算回撤
    equity_array = equity_series.values if hasattr(equity_series, 'values') else list(equity_series.values())

    # 如果是2D数组（多资产情况），求和得到组合净值
    if len(equity_array.shape) > 1 and equity_array.shape[1] > 1:
        equity_array = equity_array.sum(axis=1)

    equity_cummax = pd.Series(equity_array).cummax()
    drawdown = (equity_cummax - pd.Series(equity_array)) / equity_cummax

    x_values = equity_series.index.astype(str) if hasattr(equity_series, 'index') and hasattr(equity_series.index, 'astype') else list(range(len(drawdown)))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_values,
        y=drawdown.values * 100,  # 转换为百分比
        mode='lines',
        name='回撤',
        fill='tozeroy',
        line=dict(color='red', width=1.5),
    ))

    fig.update_layout(
        title=title,
        xaxis_title='日期',
        yaxis_title='回撤 (%)',
        hovermode='x unified',
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_monthly_returns_heatmap(returns, title: str = "月度收益热力图"):
    """绘制月度收益热力图"""
    if len(returns) == 0:
        st.warning("没有收益率数据")
        return

    # 确保索引是datetime
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)

    # 如果是DataFrame（多资产），计算组合收益
    if isinstance(returns, pd.DataFrame):
        # 对多资产求和或平均
        returns = returns.sum(axis=1) if len(returns.columns) > 1 else returns.iloc[:, 0]

    # 计算月度收益
    monthly_returns = (1 + returns).resample('M').prod() - 1

    # 创建透视表
    if isinstance(monthly_returns, pd.Series):
        monthly_df = monthly_returns.to_frame(name='return')
    else:
        # 如果monthly_returns还是DataFrame，取第一列
        monthly_df = monthly_returns.iloc[:, [0]].copy()
        monthly_df.columns = ['return']

    monthly_df['year'] = monthly_df.index.year
    monthly_df['month'] = monthly_df.index.month

    pivot_table = monthly_df.pivot(index='year', columns='month', values='return') * 100

    # 月份名称
    month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                   '7月', '8月', '9月', '10月', '11月', '12月']

    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=month_names,
        y=pivot_table.index.astype(str),
        colorscale='RdBu',
        zmid=0,
        text=np.round(pivot_table.values, 2),
        texttemplate='%{text}%',
        textfont={'size': 10},
    ))

    fig.update_layout(
        title=title,
        xaxis_title='月份',
        yaxis_title='年份',
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_quantile_returns_comparison(quantile_returns: dict, title: str = "各分层收益对比"):
    """绘制各分层收益对比图"""
    fig = go.Figure()

    for quantile_name, returns in quantile_returns.items():
        if len(returns) == 0:
            continue

        # 计算累计收益
        cumulative_returns = (1 + returns.fillna(0)).cumprod()

        x_values = cumulative_returns.index.astype(str) if hasattr(cumulative_returns.index, 'astype') else cumulative_returns.index

        fig.add_trace(go.Scatter(
            x=x_values,
            y=cumulative_returns.values,
            mode='lines',
            name=quantile_name,
        ))

    fig.update_layout(
        title=title,
        xaxis_title='日期',
        yaxis_title='累计收益',
        hovermode='x unified',
        height=400,
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)


def display_backtest_metrics(metrics: dict):
    """显示回测指标"""
    # 创建指标字典（包含说明）
    metric_info = {
        'total_return': {
            'label': '累计收益率',
            'format': '{:.2%}',
            'help': '整个回测期间的总收益率，使用复利计算'
        },
        'annual_return': {
            'label': '年化收益率',
            'format': '{:.2%}',
            'help': '将累计收益率按时间折算为年化收益率'
        },
        'volatility': {
            'label': '年化波动率',
            'format': '{:.2%}',
            'help': '收益率序列的年化标准差，衡量策略风险水平'
        },
        'sharpe_ratio': {
            'label': '夏普比率',
            'format': '{:.4f}',
            'help': '>1为良好，>2为优秀，衡量单位风险的超额收益'
        },
        'max_drawdown': {
            'label': '最大回撤',
            'format': '{:.2%}',
            'help': '从历史最高点到最低点的最大跌幅'
        },
        'calmar_ratio': {
            'label': '卡玛比率',
            'format': '{:.4f}',
            'help': '年化收益与最大回撤的比值，>1为优秀'
        },
        'win_rate': {
            'label': '胜率',
            'format': '{:.2%}',
            'help': '盈利交易日占总交易日的比例'
        },
        'sortino_ratio': {
            'label': '索提诺比率',
            'format': '{:.4f}',
            'help': '类似夏普比率，但只考虑下行风险（负收益的波动）'
        },
        'var_95': {
            'label': 'VaR (95%)',
            'format': '{:.2%}',
            'help': '在95%置信水平下，单日最大可能损失'
        },
        'cvar_95': {
            'label': 'CVaR (95%)',
            'format': '{:.2%}',
            'help': '超过VaR时的平均损失（条件VaR或期望损失）'
        },
    }

    # 按两列显示指标
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]

    for idx, (key, info) in enumerate(metric_info.items()):
        if key in metrics and metrics[key] is not None:
            with cols[idx % 4]:
                st.metric(info['label'], info['format'].format(metrics[key]), help=info['help'])


def plot_factor_decay_curve(decay_data: dict, title: str = "因子衰减曲线"):
    """绘制因子衰减曲线"""
    periods = []
    ic_values = []

    for key, value in decay_data.items():
        if key.startswith('period_'):
            period = int(key.split('_')[1])
            if not np.isnan(value):
                periods.append(period)
                ic_values.append(value)

    if not periods:
        st.warning("没有衰减数据")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=periods,
        y=ic_values,
        mode='lines+markers',
        name='IC值',
        line=dict(color='blue', width=2),
        marker=dict(size=8),
    ))

    fig.add_hline(y=0, line_dash='dash', line_color='gray')

    fig.update_layout(
        title=title,
        xaxis_title='向前周期',
        yaxis_title='IC值',
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ============ 遗传算法因子挖掘页面 ============
def genetic_mining_main():
    """遗传算法因子挖掘主页面"""
    st.title("🧬 因子挖掘")
    st.markdown("---")

    # 检查DEAP是否可用
    try:
        import deap
        deap_available = True
    except ImportError:
        deap_available = False
        st.error("❌ DEAP库未安装，请运行以下命令安装：")
        st.code("pip install deap")
        st.info("💡 提示：DEAP是遗传算法库，用于自动发现最优因子表达式")
        st.stop()

    # 侧边栏配置
    with st.sidebar:
        st.subheader("⚙️ 挖掘配置")

        # 数据源配置
        st.markdown("#### 📊 数据源")
        data_mode = st.radio(
            "数据模式",
            ["单股票", "股票池"],
            horizontal=True,
        )

        if data_mode == "单股票":
            stock_code = st.text_input(
                "股票代码",
                value="000001",
                help="支持格式：000001 或 000001.SZ"
            )
        else:
            stock_pool_input = st.text_area(
                "股票池（一行一个）",
                value="000001\n000002\n600000",
                help="每行一个股票代码"
            )
            stock_codes = [code.strip() for code in stock_pool_input.split("\n") if code.strip()]

        # 时间范围
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=365),
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=datetime.now(),
            )

        st.markdown("---")

        # 因子配置
        st.markdown("#### 🔬 基础因子")
        all_factors = factor_service.get_all_factors()

        # 按类别分组显示因子
        factor_categories = {}
        for factor in all_factors:
            category = factor.get("category", "未分类")
            if category not in factor_categories:
                factor_categories[category] = []
            factor_categories[category].append(factor["name"])

        # 默认选择一些常用因子
        default_factors = ["close", "sma_20", "rsi_14", "macd_line", "atr_14"]
        available_factor_names = [f["name"] for f in all_factors]

        # 过滤存在的默认因子
        selected_defaults = [f for f in default_factors if f in available_factor_names]

        selected_factors = st.multiselect(
            "选择基础因子",
            available_factor_names,
            default=selected_defaults if selected_defaults else None,
            help="选择用于组合生成新因子的基础因子"
        )

        st.markdown("---")

        # 遗传算法参数
        st.markdown("#### 🧬 遗传算法参数")

        population_size = st.slider(
            "种群大小",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="每代的因子数量，越大搜索越全面但耗时越长"
        )

        n_generations = st.slider(
            "迭代代数",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="进化的代数，更多代数可能找到更优解"
        )

        col1, col2 = st.columns(2)
        with col1:
            cx_prob = st.slider(
                "交叉概率",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="因子交叉组合的概率"
            )
        with col2:
            mut_prob = st.slider(
                "变异概率",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="因子变异的概率"
            )

        st.markdown("---")

        # 开始挖掘按钮
        start_mining = st.button(
            "🚀 开始挖掘",
            type="primary",
            use_container_width=True,
        )

        # 如果没有选择基础因子，显示警告
        if not selected_factors:
            st.warning("⚠️ 请至少选择一个基础因子")

    # 主内容区
    if not start_mining:
        st.info("👈 请在左侧配置参数后点击「开始挖掘」按钮")

        # 显示说明
        with st.expander("💡 使用说明", expanded=True):
            st.markdown("""
            ### 遗传算法因子挖掘

            **原理**：
            - 模拟生物进化过程，通过选择、交叉、变异操作自动发现优质因子
            - 每个个体代表一个因子表达式
            - 适应度函数评估因子质量（基于IC、IR等指标）

            **步骤**：
            1. 📊 配置数据源和基础因子
            2. ⚙️ 设置遗传算法参数
            3. 🚀 点击「开始挖掘」
            4. 📈 查看挖掘结果和最优因子

            **参数说明**：
            - **种群大小**：每代因子数量（10-200），越大搜索越全面
            - **迭代代数**：进化轮数（5-100），更多代数可能找到更优解
            - **交叉概率**：因子组合概率（0.0-1.0）
            - **变异概率**：因子变异概率（0.0-1.0）

            **输出**：
            - Top 10 最优因子表达式
            - 适应度进化曲线
            - 因子性能指标
            """)
        return

    # 验证配置
    if not selected_factors:
        st.error("❌ 请至少选择一个基础因子")
        return

    if len(selected_factors) < 2:
        st.warning("⚠️ 建议至少选择2个基础因子以获得更好的挖掘效果")

    # 显示挖掘进度
    progress_container = st.container()

    with progress_container:
        st.subheader("⏳ 挖掘进度")

        # 创建进度条和状态显示
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

        with metrics_col1:
            gen_metric = st.metric("当前代数", "0 / 0")
        with metrics_col2:
            best_fitness = st.metric("最优适应度", "-")
        with metrics_col3:
            avg_fitness = st.metric("平均适应度", "-")

    # 获取数据
    try:
        with st.spinner("正在获取数据..."):
            if data_mode == "单股票":
                data = data_service.get_stock_data(
                    stock_code,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )
                # 计算收益率
                if "close" in data.columns:
                    data["return"] = data["close"].pct_change()
            else:
                # 股票池模式：获取第一只股票的数据（简化版）
                stock_code = stock_codes[0]
                data = data_service.get_stock_data(
                    stock_code,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )
                if "close" in data.columns:
                    data["return"] = data["close"].pct_change()

        st.success(f"✅ 数据获取成功：{len(data)} 条记录")

    except Exception as e:
        st.error(f"❌ 数据获取失败：{str(e)}")
        return

    # 准备基础因子数据
    try:
        with st.spinner("正在计算基础因子..."):
            # 确保数据包含必要的列
            if "close" not in data.columns:
                st.error("❌ 数据缺少close列")
                return

            # 为每个基础因子计算因子值
            for factor_name in selected_factors:
                if factor_name in data.columns:
                    continue

                # 简化处理：使用close列作为因子
                if factor_name == "close":
                    continue
                elif factor_name.startswith("sma_"):
                    period = int(factor_name.split("_")[1])
                    data[factor_name] = data["close"].rolling(window=period).mean()
                elif factor_name == "rsi_14":
                    delta = data["close"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / (loss + 1e-8)
                    data[factor_name] = 100 - (100 / (1 + rs))
                elif factor_name.startswith("macd"):
                    ema12 = data["close"].ewm(span=12, adjust=False).mean()
                    ema26 = data["close"].ewm(span=26, adjust=False).mean()
                    data[factor_name] = ema12 - ema26
                elif factor_name.startswith("atr_"):
                    high_low = data["close"] - data["close"].shift(1)
                    high_low = high_low.abs()
                    data[factor_name] = high_low.rolling(window=14).mean()
                else:
                    # 其他因子使用close列的变换
                    data[factor_name] = data["close"]

            # 填充缺失值
            data.fillna(method="ffill", inplace=True)
            data.fillna(method="bfill", inplace=True)

        st.success(f"✅ 基础因子计算完成")

    except Exception as e:
        st.error(f"❌ 基础因子计算失败：{str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return

    # 执行遗传算法挖掘
    try:
        with st.spinner("正在执行遗传算法挖掘..."):
            # 创建挖掘服务
            service = create_genetic_mining_service(
                base_factors=selected_factors,
                data=data,
                population_size=population_size,
                n_generations=n_generations,
                cx_prob=cx_prob,
                mut_prob=mut_prob,
            )

            # 执行挖掘
            result = service.mine_factors()

        st.success("✅ 挖掘完成！")

    except Exception as e:
        st.error(f"❌ 挖掘失败：{str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return

    # 显示挖掘结果
    st.markdown("---")
    st.subheader("🏆 挖掘结果")

    if not result.get("success"):
        st.error(f"❌ 挖掘失败：{result.get('message', '未知错误')}")
        return

    best_factors = result.get("best_factors", [])

    if not best_factors:
        st.warning("⚠️ 未找到有效的因子")
        return

    # 结果统计
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("发现因子数", len(best_factors))
    with col2:
        st.metric("最优适应度", f"{best_factors[0]['fitness']:.4f}")
    with col3:
        avg_fitness_value = sum(f['fitness'] for f in best_factors) / len(best_factors)
        st.metric("平均适应度", f"{avg_fitness_value:.4f}")

    st.markdown("---")

    # Top 10 因子展示
    st.subheader("📋 最优因子 Top 10")

    for i, factor_info in enumerate(best_factors[:10], 1):
        # 标题行：显示因子表达式和保存按钮
        col_title, col_save = st.columns([4, 1])
        with col_title:
            st.markdown(f"**#{i}** `{factor_info['expression']}`  (适应度: {factor_info['fitness']:.4f})")

        with col_save:
            # 保存按钮（放在外部，更容易访问）
            if st.button("💾 保存", key=f"save_{i}", use_container_width=True):
                try:
                    # 创建新因子
                    factor_name = f"genetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{i}"
                    factor_service.create_factor(
                        name=factor_name,
                        code=factor_info['expression'],
                        description=f"遗传算法挖掘的第{i}名因子，适应度: {factor_info['fitness']:.4f}"
                    )
                    st.success(f"✅ 因子 `{factor_name}` 已保存到因子库")
                    # 使用session_state记录保存成功
                    st.session_state[f"saved_{i}"] = True
                except Exception as e:
                    st.error(f"❌ 保存失败：{str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

        # 显示详细信息的折叠面板
        with st.expander(f"查看详情 #{i}", expanded=(i == 1)):
            col1 = st.columns(1)[0]

            with col1:
                st.markdown("**因子表达式**：")
                st.code(
                    factor_info['expression'],
                    language="python"
                )

                # 显示验证信息（如果有）
                if "validation" in factor_info:
                    validation = factor_info["validation"]
                    st.markdown("**验证指标**：")

                    val_col1, val_col2, val_col3 = st.columns(3)

                    with val_col1:
                        # IC从ic_validation中获取
                        ic_val = validation.get('ic_validation', {}).get('ic', 0)
                        st.metric("IC", f"{ic_val:.4f}")

                        # IR从ir_validation中获取
                        ir_val = validation.get('ir_validation', {}).get('ir', 0)
                        st.metric("IR", f"{ir_val:.4f}")

                    with val_col2:
                        # 换手率从turnover_validation中获取
                        turnover_val = validation.get('turnover_validation', {}).get('turnover', 0)
                        st.metric("换手率", f"{turnover_val:.2%}")

                        # IC胜率计算：IC>0的比例（使用ic_mean计算）
                        ic_mean_val = validation.get('ir_validation', {}).get('ic_mean', 0)
                        ic_std_val = validation.get('ir_validation', {}).get('ic_std', 0.001)
                        win_rate_val = 0.5 + 0.5 * (ic_mean_val / (ic_std_val + 1e-8)) if ic_std_val > 0 else 0.5
                        win_rate_val = max(0, min(1, win_rate_val))  # 限制在0-1之间
                        st.metric("IC胜率", f"{win_rate_val:.2%}")

                    with val_col3:
                        # 稳定性从stability_validation中获取
                        stability_val = validation.get('stability_validation', {}).get('stability_score', 0)
                        st.metric("稳定性", f"{stability_val:.4f}")

                        # 综合得分
                        score_val = validation.get('score', 0)
                        st.metric("综合得分", f"{score_val:.2f}")

    st.markdown("---")

    # 进化曲线
    if "logbook" in result and result["logbook"] is not None:
        st.subheader("📈 进化曲线")

        logbook = result["logbook"]

        if len(logbook) > 0:
            # 提取数据
            try:
                generations = list(logbook.select("gen"))

                # 检查是否有fitness章节
                if "fitness" in logbook.chapters:
                    avg_fitness = list(logbook.chapters["fitness"].select("avg"))
                    max_fitness = list(logbook.chapters["fitness"].select("max"))
                    min_fitness = list(logbook.chapters["fitness"].select("min"))
                else:
                    # 备用方法：直接从logbook记录中提取
                    avg_fitness = []
                    max_fitness = []
                    min_fitness = []

                    for record in logbook:
                        if hasattr(record, '__getitem__'):
                            record_dict = dict(record) if hasattr(record, 'keys') else record
                            avg_fitness.append(float(record_dict.get('avg', 0)))
                            max_fitness.append(float(record_dict.get('max', 0)))
                            min_fitness.append(float(record_dict.get('min', 0)))

                # 显示数据表格
                with st.expander("📋 详细数据", expanded=False):
                    import pandas as pd
                    min_len = min(len(generations), len(min_fitness), len(avg_fitness), len(max_fitness))

                    if min_len > 0:
                        df_stats = pd.DataFrame({
                            "代数": generations[:min_len],
                            "最小适应度": [round(float(f), 4) for f in min_fitness[:min_len]],
                            "平均适应度": [round(float(f), 4) for f in avg_fitness[:min_len]],
                            "最大适应度": [round(float(f), 4) for f in max_fitness[:min_len]],
                        })
                        st.dataframe(df_stats, use_container_width=True)

                # 绘制图表
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=generations,
                    y=max_fitness,
                    mode="lines+markers",
                    name="最大适应度",
                    line=dict(color="green", width=2),
                ))

                fig.add_trace(go.Scatter(
                    x=generations,
                    y=avg_fitness,
                    mode="lines+markers",
                    name="平均适应度",
                    line=dict(color="blue", width=2),
                ))

                fig.add_trace(go.Scatter(
                    x=generations,
                    y=min_fitness,
                    mode="lines+markers",
                    name="最小适应度",
                    line=dict(color="red", width=2, dash="dash"),
                ))

                fig.update_layout(
                    title=f"适应度进化曲线 (代数: {generations[-1] + 1 if generations else 0})",
                    xaxis_title="代数",
                    yaxis_title="适应度",
                    hovermode="x unified",
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                )

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ 绘制进化曲线失败: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Logbook 为空，无法绘制进化曲线")
            st.info("""
            **可能的原因**：
            1. 种群太小或迭代代数太少
            2. 所有因子评估都失败（fitness都是0）
            3. DEAP统计配置问题

            **建议**：
            - 增加种群大小（population_size）
            - 增加迭代代数（n_generations）
            - 检查数据和因子配置
            """)
    else:
        st.info("ℹ️ 未找到进化曲线数据")

    st.markdown("---")

    # 批量操作
    st.subheader("🛠️ 批量操作")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 批量保存Top 10", use_container_width=True):
            saved_count = 0
            for i, factor_info in enumerate(best_factors[:10], 1):
                try:
                    factor_service.create_factor(
                        name=f"genetic_factor_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                        code=factor_info['expression'],
                        description=f"遗传算法挖掘因子 #{i}，适应度: {factor_info['fitness']:.4f}"
                    )
                    saved_count += 1
                except:
                    pass
            if saved_count > 0:
                st.success(f"✅ 成功保存 {saved_count} 个因子到因子库")
            else:
                st.error("❌ 保存失败")

    with col2:
        # 导出功能
        if st.button("📥 导出为CSV", use_container_width=True):
            # 准备导出数据
            export_data = []
            for i, factor_info in enumerate(best_factors, 1):
                export_data.append({
                    "排名": i,
                    "表达式": factor_info['expression'],
                    "适应度": factor_info['fitness'],
                })

            df_export = pd.DataFrame(export_data)

            # 提供下载
            csv = df_export.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="⬇️ 下载CSV文件",
                data=csv,
                file_name=f"genetic_factors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


# ============ 策略回测页面 ============
def backtest_main():
    """策略回测主页面"""
    st.title("策略回测")

    # 添加标签页选择
    tab1, tab2 = st.tabs(["单策略回测", "多策略对比"])

    with tab1:
        backtest_single()

    with tab2:
        backtest_comparison()


def backtest_single():
    """单策略回测"""

    # 检查vectorbt是否可用
    if not check_vectorbt_available():
        st.error("❌ VectorBT未安装，请运行以下命令安装：")
        st.code("pip install vectorbt")
        st.stop()

    # 初始化服务
    if "backtest_repo" not in st.session_state:
        st.session_state.backtest_repo = BacktestRepository()

    # 侧边栏配置
    with st.sidebar:
        st.subheader("回测配置")

        # 数据模式
        data_mode = st.radio("数据模式", ["单股票", "股票池"], key="single_data_mode")

        # 股票代码输入
        if data_mode == "单股票":
            stock_codes_input = st.text_input(
                "股票代码",
                value="000001",
                help="支持格式：000001 或 000001.SZ",
                key="single_stock_input"
            )
            stock_codes = [stock_codes_input.strip()]
        else:
            stock_codes_input = st.text_area(
                "股票代码（每行一个）",
                value="000001\n600000",
                height=100,
                key="single_stock_area"
            )
            stock_codes = [code.strip() for code in stock_codes_input.strip().split("\n") if code.strip()]

        # 选择因子
        all_factors = factor_service.get_all_factors()
        factor_options = {f["name"]: f["description"] for f in all_factors}
        selected_factors = st.multiselect(
            "选择因子",
            options=list(factor_options.keys()),
            format_func=lambda x: f"{x} - {factor_options.get(x, '')}",
            default=[],
            key="single_factors"
        )

        # 回测参数
        st.markdown("### 回测参数")
        initial_capital = st.number_input("初始资金", value=1000000, min_value=10000, step=10000, key="single_initial")
        commission_rate = st.number_input("费率(%)", value=0.03, min_value=0.0, max_value=1.0, step=0.01, key="single_comm") / 100
        slippage = st.number_input("滑点(%)", value=0.00, min_value=0.0, max_value=1.0, step=0.01, help="买卖时的价格滑点百分比，例如0.01表示万分之一", key="single_slippage") / 100

        # 策略类型
        strategy_type = st.selectbox(
            "策略类型",
            options=["单因子分层", "多因子组合"],
            index=0,
            key="single_strategy_type"
        )

        if strategy_type == "单因子分层":
            percentile = st.slider("分层阈值（百分位）", 0, 100, 50, key="single_percentile")
            direction = st.selectbox("交易方向", ["long", "short"], key="single_direction")
            n_quantiles = st.slider("分层数量", 3, 10, 5, key="single_quantiles")
        else:
            weight_method = st.selectbox(
                "权重分配",
                ["等权重", "风险平价"],
                index=0,
                key="single_weight_method"
            )
            percentile = st.slider("组合分层阈值（百分位）", 0, 100, 50, key="single_portfolio_percentile")
            direction = st.selectbox("交易方向", ["long", "short"], key="single_portfolio_direction")

        # 时间范围
        default_end = datetime.now()
        default_start = default_end - timedelta(days=365*3)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("开始日期", value=default_start, key="single_start_date")
        with col2:
            end_date = st.date_input("结束日期", value=default_end, key="single_end_date")

        # 运行回测按钮
        run_backtest = st.button("运行回测", type="primary", use_container_width=True, key="run_single_backtest")

        # 历史回测记录
        st.divider()
        if st.button("查看历史记录", key="view_history"):
            st.session_state.show_backtest_history = True

    # 显示历史记录
    if st.session_state.get("show_backtest_history", False):
        st.subheader("历史回测记录")

        history = st.session_state.backtest_repo.get_history(limit=10)

        if not history:
            st.info("暂无历史记录")
        else:
            history_data = []
            for record in history:
                history_data.append({
                    "策略名称": record.strategy_name,
                    "因子组合": record.factor_combination[:50] + "..." if len(record.factor_combination) > 50 else record.factor_combination,
                    "时间范围": f"{record.start_date} ~ {record.end_date}",
                    "累计收益": f"{record.total_return:.2%}" if record.total_return else "N/A",
                    "夏普比率": f"{record.sharpe_ratio:.4f}" if record.sharpe_ratio else "N/A",
                    "最大回撤": f"{record.max_drawdown:.2%}" if record.max_drawdown else "N/A",
                    "创建时间": record.created_at.strftime("%Y-%m-%d %H:%M"),
                })

            st.dataframe(
                pd.DataFrame(history_data),
                use_container_width=True,
                hide_index=True,
            )

            # 删除记录
            with st.expander("删除历史记录"):
                record_to_delete = st.selectbox(
                    "选择要删除的记录",
                    options=range(len(history)),
                    format_func=lambda i: f"{history[i].strategy_name} - {history[i].created_at.strftime('%Y-%m-%d')}",
                    key="select_history_record"
                )
                if st.button("确认删除选中记录", type="secondary", key="delete_history_record"):
                    if st.session_state.backtest_repo.delete_by_id(history[record_to_delete].id):
                        st.success("删除成功")
                        st.rerun()
                    else:
                        st.error("删除失败")

        if st.button("返回回测配置", key="return_from_history"):
            st.session_state.show_backtest_history = False
            st.rerun()

        return

    # 执行回测
    if run_backtest:
        if not stock_codes or not stock_codes[0]:
            st.error("请输入股票代码")
        elif not selected_factors:
            st.error("请至少选择一个因子")
        elif strategy_type == "单因子分层" and len(selected_factors) > 1:
            st.warning("单因子分层策略只选择一个因子，将使用第一个因子")
            selected_factors = [selected_factors[0]]
        else:
            with st.spinner("正在运行回测，请稍候..."):
                try:
                    # 获取数据
                    from backend.services.data_service import data_service

                    all_factor_data = {}
                    all_price_data = {}

                    for stock_code in stock_codes:
                        # 获取股票数据
                        stock_data = data_service.get_stock_data(
                            stock_code,
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d"),
                        )

                        if stock_data is not None and len(stock_data) > 0:
                            # 计算因子
                            factor_calculator = factor_service.calculator
                            for factor_name in selected_factors:
                                factor_code = None
                                for f in all_factors:
                                    if f["name"] == factor_name:
                                        factor_code = f["code"]
                                        break

                                if factor_code:
                                    factor_values = factor_calculator.calculate(
                                        stock_data, factor_code
                                    )
                                    stock_data[factor_name] = factor_values

                            all_factor_data[stock_code] = stock_data

                    if not all_factor_data:
                        st.error("未获取到有效数据，请检查股票代码和时间范围")
                        return

                    # 使用VectorBT回测引擎执行回测
                    backtest_service = VectorBTBacktestService(
                        initial_capital=initial_capital,
                        commission_rate=commission_rate,
                        slippage=slippage,
                    )

                    # 判断是单股票还是股票池
                    is_single_stock = len(all_factor_data) == 1

                    if strategy_type == "单因子分层":
                        factor_name = selected_factors[0]

                        if is_single_stock:
                            # 单股票：使用时序回测
                            df = list(all_factor_data.values())[0].copy()
                            result = backtest_service.single_factor_backtest(
                                df=df,
                                factor_name=factor_name,
                                percentile=percentile,
                                direction=direction,
                                n_quantiles=n_quantiles,
                            )
                            # 单因子时序回测的指标已在回测方法中计算
                            metrics = {k: v for k, v in result.items() if k in [
                                "total_return", "annual_return", "volatility", "sharpe_ratio",
                                "max_drawdown", "calmar_ratio", "win_rate", "sortino_ratio",
                                "var_95", "cvar_95"
                            ]}
                        else:
                            # 股票池：使用横截面回测
                            # 合并数据
                            merged_list = []
                            for code, data in all_factor_data.items():
                                data_copy = data.copy()
                                data_copy["stock_code"] = code
                                # 确保有日期列
                                if "date" not in data_copy.columns:
                                    data_copy = data_copy.reset_index()
                                merged_list.append(data_copy)

                            df = pd.concat(merged_list, ignore_index=True)

                            # 使用横截面回测
                            top_percentile = (100 - percentile) / 100.0  # 转换为选择比例
                            result = backtest_service.cross_sectional_backtest(
                                df=df,
                                factor_name=factor_name,
                                top_percentile=top_percentile,
                                direction=direction,
                            )
                            # 横截面回测的指标已在回测方法中计算
                            metrics = {k: v for k, v in result.items() if k in [
                                "total_return", "annual_return", "volatility", "sharpe_ratio",
                                "max_drawdown", "calmar_ratio", "win_rate", "sortino_ratio",
                                "var_95", "cvar_95"
                            ]}
                    else:
                        # 多因子组合回测（目前仅支持单股票时序回测）
                        if not is_single_stock:
                            st.warning("多因子组合策略目前仅支持单股票模式，将使用第一只股票")
                            df = list(all_factor_data.values())[0].copy()
                        else:
                            df = list(all_factor_data.values())[0].copy()

                        weight_map = {"等权重": "equal_weight", "风险平价": "risk_parity"}
                        result = backtest_service.multi_factor_backtest(
                            df=df,
                            factor_names=selected_factors,
                            method=weight_map[weight_method],
                            percentile=percentile,
                            direction=direction,
                        )
                        # 多因子回测的指标已在回测方法中计算
                        metrics = {k: v for k, v in result.items() if k in [
                            "total_return", "annual_return", "volatility", "sharpe_ratio",
                            "max_drawdown", "calmar_ratio", "win_rate", "sortino_ratio",
                            "var_95", "cvar_95"
                        ]}

                    # 保存到session state
                    st.session_state.backtest_result = {
                        "result": result,
                        "metrics": metrics,
                        "config": {
                            "strategy_type": strategy_type,
                            "factors": selected_factors,
                            "stock_codes": stock_codes,
                            "start_date": start_date.strftime("%Y-%m-%d"),
                            "end_date": end_date.strftime("%Y-%m-%d"),
                            "initial_capital": initial_capital,
                        },
                        "raw_data": all_factor_data  # 保存原始的价格和因子数据
                    }

                    st.success("回测完成！")

                except Exception as e:
                    st.error(f"回测失败: {e}")
                    import traceback
                    st.error(traceback.format_exc())

    # 显示回测结果
    if "backtest_result" in st.session_state:
        result = st.session_state.backtest_result["result"]
        metrics = st.session_state.backtest_result["metrics"]
        config = st.session_state.backtest_result["config"]

        st.subheader("回测结果")

        # Tab1: 概览, Tab2: 净值分析, Tab3: 分层分析, Tab4: 交易日志
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["概览", "净值分析", "分层分析", "交易日志", "持仓分析"])

        with tab1:
            # 策略说明
            with st.expander("📖 策略说明与指标解释", expanded=False):
                st.markdown("""
                ### 策略概述（基于VectorBT回测引擎）

                **回测引擎说明**：
                本系统使用 VectorBT 专业回测引擎，提供高效、准确的向量化回测计算。
                VectorBT 基于 pandas 和 NumPy，采用事件驱动的回测框架，能够精确模拟真实交易环境。

                ---

                ### 📊 单因子分层策略交易逻辑

                **适用场景**：单股票回测、单一因子的择时策略

                **核心步骤**：

                1️⃣ **因子排名计算**
                - 使用 **252天滚动窗口** 计算因子值的百分位排名
                - 排名范围 0-1 之间（0% 最低，100% 最高）
                - 公式：`factor_rank = factor.rolling(252).rank(pct=True)`

                2️⃣ **交易信号生成**（以做多为例）
                | 条件 | 信号 | 持仓状态 |
                |------|------|----------|
                | 因子排名 ≥ 阈值（默认50%） | 买入（Entry） | 满仓（100%） |
                | 因子排名 < 阈值 | 卖出（Exit） | 空仓（0%） |

                3️⃣ **交易执行**
                - 二元交易策略：要么满仓，要么空仓
                - 当信号触发时，VectorBT 自动执行交易
                - 自动扣除交易费用和滑点

                **示例**：
                ```
                日期       因子值    因子排名   阈值50%   信号   持仓
                2024-01-01   0.8      0.85      ✓       买入   100%
                2024-01-02   0.7      0.75      ✓       持有   100%
                2024-01-03   0.3      0.45      ✗       卖出   0%
                ```

                ---

                ### 🎯 多因子组合策略交易逻辑

                **适用场景**：多个因子共同决策、提升策略稳定性

                **核心步骤**：

                1️⃣ **因子标准化（Z-score Normalization）**
                - 消除不同因子间的量纲差异
                - 公式：`normalized_factor = (factor - mean) / std`
                - 所有因子转换为均值为0、标准差为1的分布

                2️⃣ **综合得分计算**
                | 权重方法 | 计算逻辑 | 适用场景 |
                |----------|----------|----------|
                | 等权重 | 所有标准化因子简单平均 | 因子重要性相近 |
                | 风险平价 | 波动率小的因子权重更大 | 追求风险均衡 |
                | IC加权 | 与收益率相关性高的因子权重更大 | 基于历史效果 |

                3️⃣ **交易执行**
                - 将综合得分作为"超级因子"
                - 后续逻辑与单因子策略完全相同
                - 按综合得分的百分位排名生成买卖信号

                **示例**：
                ```
                日期       PE因子   ROE因子   MOM因子   综合得分   排名   信号
                2024-01-01  -0.5     1.2      0.8       0.5       0.70  买入
                2024-01-02  -0.3     1.1      0.9       0.57      0.75  持有
                2024-01-03   0.2     0.3      0.1       0.2       0.45  卖出
                ```

                ---

                ### 🏢 股票池横截面回测逻辑

                **适用场景**：多股票选股策略、因子有效性验证

                **核心步骤**：
                1. 每个交易日计算所有股票的因子值排名
                2. 选择排名前 N% 的股票（做多）或后 N% 的股票（做空）
                3. 对选中的股票进行等权重配置，每日调仓
                4. 使用 VectorBT 的多资产回测功能，精确计算每只股票的收益贡献

                ---

                ### 📈 关键指标说明（VectorBT标准计算）

                - **累计收益率**：投资组合在整个回测期间的总收益率，使用复利计算
                - **年化收益率**：将累计收益率按时间折算为年化收益率
                - **年化波动率**：收益率序列的年化标准差，衡量策略风险水平
                - **夏普比率**：(年化收益 - 无风险利率) / 年化波动率
                  - 衡量单位风险的超额收益，> 1 为良好，> 2 为优秀
                - **最大回撤**：从历史最高净值到最低点的最大跌幅
                - **卡玛比率**：年化收益率 / 最大回撤，衡量单位回撤风险获得的收益
                - **胜率**：盈利交易日占总交易日的比例
                - **索提诺比率**：类似夏普比率，但只考虑下行风险（负收益的波动）
                - **VaR (95%)**：在95%置信水平下，单日最大可能损失
                - **CVaR (95%)**：超过VaR时的平均损失（条件VaR或期望损失）
                """)

            st.markdown("### 性能指标")
            display_backtest_metrics(metrics)

            st.markdown("### 回测配置")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**策略类型**:", config["strategy_type"])
                st.write("**因子**:", ", ".join(config["factors"]))
            with col2:
                st.write("**股票代码**:", ", ".join(config["stock_codes"]))
                st.write("**时间范围**:", f"{config['start_date']} ~ {config['end_date']}")

            # 保存回测结果
            if st.button("保存回测结果", type="primary"):
                try:
                    save_data = {
                        "strategy_name": f"{config['strategy_type']}_{','.join(config['factors'])}",
                        "factor_combination": str(config["factors"]),
                        "start_date": config["start_date"],
                        "end_date": config["end_date"],
                        "initial_capital": config["initial_capital"],
                        "final_capital": result["equity_curve"].iloc[-1] if hasattr(result["equity_curve"], "iloc") else list(result["equity_curve"].values())[-1],
                        **metrics,
                        "trades_count": result.get("trades_count", 0),
                    }

                    saved_record = st.session_state.backtest_repo.save_result(save_data)
                    st.success(f"回测结果已保存 (ID: {saved_record.id})")
                except Exception as e:
                    st.error(f"保存失败: {e}")

        with tab2:
            # 净值分析说明
            with st.expander("📖 净值分析说明（VectorBT计算）", expanded=False):
                st.markdown("""
                ### 净值曲线

                **计算逻辑（VectorBT）**：
                - 使用 VectorBT 的 `Portfolio.value()` 方法计算
                - 净值 = 初始资金 + 所有交易收益的累计和
                - 精确模拟每日持仓变化、交易成本、资金流动
                - 可用于观察策略的累积表现和资金使用效率

                ### 回撤曲线

                **计算逻辑（VectorBT）**：
                - 回撤 = (历史最高净值 - 当前净值) / 历史最高净值
                - 使用 VectorBT 的 `drawdown()` 方法计算
                - 衡量从历史最高点到当前点的跌幅
                - 最大回撤是评估策略风险的重要指标

                ### 月度收益热力图

                **计算逻辑**：
                - 将 VectorBT 计算的每日收益率按月汇总
                - 使用月度复利计算：(1 + 日收益率) 累乘 - 1
                - 使用颜色深浅表示收益正负（红色为负，蓝色为正）
                - 可用于识别策略的季节性规律和周期性特征

                **使用建议**：
                - 净值曲线应持续向上，避免大幅波动
                - 回撤应快速恢复，避免长期处于低位
                - 月度收益应保持相对稳定，避免大起大落
                """)

            st.markdown("### 净值曲线")
            plot_equity_curve(result["equity_curve"], title="策略净值曲线")

            st.markdown("### 回撤曲线")
            plot_drawdown_curve(result["equity_curve"], title="策略回撤曲线")

            if "portfolio_returns" in result:
                st.markdown("### 月度收益热力图")
                plot_monthly_returns_heatmap(result["portfolio_returns"], title="月度收益率")

        with tab3:
            # 分层分析说明
            with st.expander("📖 分层分析说明（VectorBT计算）", expanded=False):
                st.markdown("""
                ### 各分层收益对比

                **分层逻辑（VectorBT实现）**：
                - 根据因子值将数据分为N层（通常为5层）
                - 使用滚动窗口（252天）计算因子值的动态分位数排名
                - Q1表示因子值最低的组，Q5表示因子值最高的组
                - 使用 `pd.qcut()` 确保等频分层，每组数据量相同

                **计算逻辑**：
                - 每层分别计算收益，基于 VectorBT 的信号回测结果
                - 使用 `rolling(252).rank(pct=True)` 计算动态排名
                - 每日重新分层，动态调整持仓
                - 用于检验因子的单调性和预测能力

                **解读方法**：
                - **有效因子**：Q1 < Q2 < Q3 < Q4 < Q5（单调递增）
                - **无效因子**：各层收益无明显差异
                - **反向因子**：Q1 > Q2 > Q3 > Q4 > Q5（单调递减）

                ### 各分层统计

                **指标说明**：
                - **日均收益**：该层所有交易日的平均日收益率（VectorBT精确计算）
                - **年化收益**：日均收益 × 252（年化交易日数）
                - **夏普比率**：衡量该层风险调整后的收益表现
                - **胜率**：该层盈利交易日占比

                **使用建议**：
                - 优先选择各层收益差异明显的因子
                - Q5与Q1的收益差越大，因子区分度越好
                - 夏普比率在各层都应该相对稳定
                - 有效因子的分层收益应呈现明显的单调性
                """)

            if "quantile_returns" in result and result["quantile_returns"]:
                st.markdown("### 各分层收益对比")
                plot_quantile_returns_comparison(result["quantile_returns"])

                # 各分层统计
                st.markdown("### 各分层统计")
                from backend.services.statistics_service import StatisticsService
                stats_service = StatisticsService()

                quantile_stats = stats_service.analyze_quantile_returns(result["quantile_returns"])

                stats_data = []
                for quantile_name, stats in quantile_stats.items():
                    stats_data.append({
                        "分层": quantile_name,
                        "日均收益": stats['mean'],
                        "年化收益": stats['annual_return'],
                        "夏普比率": stats['sharpe'],
                        "胜率": stats['win_rate'],
                    })

                st.dataframe(
                    pd.DataFrame(stats_data),
                    column_config={
                        "分层": st.column_config.TextColumn("分层", width="small"),
                        "日均收益": st.column_config.NumberColumn(
                            "日均收益",
                            format="%.4f",
                            help="📖 **日均收益**：该层所有交易日的平均日收益率\n\n- 正值表示平均盈利，负值表示平均亏损\n- 绝对值越大，该层表现越极端\n- 用于比较不同层的平均表现"
                        ),
                        "年化收益": st.column_config.NumberColumn(
                            "年化收益",
                            format="%.2%",
                            help="📖 **年化收益**：将日均收益按年化交易日数推算\n\n- 公式：日均收益 × 252\n- 便于与年度基准比较\n- 年化收益越高，该层长期表现越好"
                        ),
                        "夏普比率": st.column_config.NumberColumn(
                            "夏普比率",
                            format="%.4f",
                            help="📖 **夏普比率**：风险调整后的收益指标\n\n- >1 为良好，>2 为优秀\n- 衡量单位风险的超额收益\n- 公式：(年化收益 - 无风险利率) / 年化波动率"
                        ),
                        "胜率": st.column_config.NumberColumn(
                            "胜率",
                            format="%.2%",
                            help="📖 **胜率**：盈利交易日占比\n\n- 盈利天数 / 总交易天数\n- >50% 表示盈利天数多于亏损天数\n- 衡量该层的胜率稳定性"
                        ),
                    },
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("当前回测结果没有分层数据（股票池横截面回测不显示分层）")

        with tab4:
            st.subheader("交易日志")

            # 交易日志说明
            with st.expander("📖 交易日志说明", expanded=False):
                st.markdown("""
                ### 交易记录详情

                **数据来源**：VectorBT 回测引擎自动记录每笔交易

                **字段说明**：
                - **日期**：交易发生的日期
                - **方向**：买入(Long)或卖出(Short)
                - **价格**：成交价格
                - **数量**：交易数量（股数或单位）
                - **价值**：交易价值（价格×数量）
                - **收益**：该笔交易的盈亏
                - **收益率**：该笔交易的收益率

                **使用建议**：
                - 查看每笔交易的详细记录
                - 分析交易的盈利和亏损模式
                - 识别交易频繁的时段
                """)

            # 绘制行情和指标图表
            if "raw_data" in st.session_state.backtest_result and st.session_state.backtest_result["raw_data"]:
                raw_data = st.session_state.backtest_result["raw_data"]
                factors = config.get("factors", [])

                if len(raw_data) > 0:
                    st.markdown("---")
                    st.markdown("### 📈 行情与指标分析")

                    # 如果有多只股票，为每只股票创建一个tab
                    stock_codes_list = list(raw_data.keys())

                    if len(stock_codes_list) > 1:
                        stock_tabs = st.tabs([f"{code}" for code in stock_codes_list])
                    else:
                        stock_tabs = [st.container()]

                    for idx, stock_code in enumerate(stock_codes_list):
                        stock_data = raw_data[stock_code]

                        with stock_tabs[idx]:
                            # 获取价格数据
                            if "close" in stock_data.columns:
                                prices = stock_data["close"]
                                dates = stock_data.index

                                # 准备交易信号标记
                                buy_signals = []
                                sell_signals = []

                                # 从trades中提取交易信号
                                if "trades" in result and result["trades"] is not None:
                                    trades_df = result["trades"]

                                    # 过滤当前股票的交易记录
                                    # 对于单股票回测，VectorBT的Column字段可能是"close"或索引，不是股票代码
                                    if "股票代码" in trades_df.columns:
                                        # 尝试按股票代码过滤
                                        stock_trades = trades_df[trades_df["股票代码"] == stock_code]
                                        # 如果过滤后没有数据，且只有一只股票，则使用所有交易
                                        if len(stock_trades) == 0 and len(stock_codes_list) == 1:
                                            # 可能是单股票回测，Column字段值为0或"close"
                                            stock_trades = trades_df
                                    else:
                                        stock_trades = trades_df

                                    # 使用iterrows()安全地遍历交易记录
                                    for trade_idx, trade in stock_trades.iterrows():
                                        # trade_idx是入场时间（索引）
                                        if "方向" in trade.index and "入场价格" in trade.index:
                                            entry_price = trade["入场价格"]
                                            exit_price = trade.get("出场价格")
                                            exit_timestamp = trade.get("出场时间")
                                            direction = trade["方向"]

                                            # 跳过NaN值
                                            if pd.isna(entry_price) or pd.isna(direction):
                                                continue

                                            # 添加入场信号
                                            try:
                                                entry_price = float(entry_price)
                                                if direction == "做多":
                                                    buy_signals.append({
                                                        "date": trade_idx,
                                                        "price": entry_price
                                                    })
                                                elif direction == "做空":
                                                    sell_signals.append({
                                                        "date": trade_idx,
                                                        "price": entry_price
                                                    })
                                            except (ValueError, TypeError):
                                                pass

                                            # 添加出场信号（如果存在）
                                            if not pd.isna(exit_timestamp) and not pd.isna(exit_price):
                                                try:
                                                    exit_price = float(exit_price)
                                                    # 确保出场时间是datetime类型
                                                    if not isinstance(exit_timestamp, pd.Timestamp):
                                                        exit_timestamp = pd.to_datetime(exit_timestamp)

                                                    # 做多平仓是卖出，做空平仓是买入
                                                    if direction == "做多":
                                                        sell_signals.append({
                                                            "date": exit_timestamp,
                                                            "price": exit_price
                                                        })
                                                    elif direction == "做空":
                                                        buy_signals.append({
                                                            "date": exit_timestamp,
                                                            "price": exit_price
                                                        })
                                                except (ValueError, TypeError):
                                                    pass

                                # 创建子图，共用x轴
                                from plotly.subplots import make_subplots
                                fig = make_subplots(
                                    rows=2, cols=1,
                                    shared_xaxes=True,
                                    vertical_spacing=0.08,
                                    row_heights=[0.6, 0.4],
                                    subplot_titles=("行情走势与交易信号", "")
                                )

                                # 添加价格线
                                fig.add_trace(
                                    go.Scatter(
                                        x=dates,
                                        y=prices,
                                        mode='lines',
                                        name='价格',
                                        line=dict(color='#1f77b4', width=1.5)
                                    ),
                                    row=1, col=1
                                )

                                # 添加买入信号
                                if buy_signals:
                                    buy_dates = [s["date"] for s in buy_signals]
                                    buy_prices = [s["price"] for s in buy_signals]
                                    fig.add_trace(
                                        go.Scatter(
                                            x=buy_dates,
                                            y=buy_prices,
                                            mode='markers',
                                            name='买入',
                                            marker=dict(
                                                symbol='triangle-up',
                                                size=14,
                                                color='red',
                                                line=dict(width=2, color='darkred')
                                            ),
                                            hovertemplate='买入<br>日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
                                        ),
                                        row=1, col=1
                                    )

                                # 添加卖出信号
                                if sell_signals:
                                    sell_dates = [s["date"] for s in sell_signals]
                                    sell_prices = [s["price"] for s in sell_signals]
                                    fig.add_trace(
                                        go.Scatter(
                                            x=sell_dates,
                                            y=sell_prices,
                                            mode='markers',
                                            name='卖出',
                                            marker=dict(
                                                symbol='triangle-down',
                                                size=14,
                                                color='green',
                                                line=dict(width=2, color='darkgreen')
                                            ),
                                            hovertemplate='卖出<br>日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
                                        ),
                                        row=1, col=1
                                    )

                                # 添加因子线（归一化处理）
                                colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

                                for factor_idx, factor_name in enumerate(factors):
                                    if factor_name in stock_data.columns:
                                        factor_values = stock_data[factor_name]

                                        # 归一化处理到 [0, 1] 区间
                                        scaler = MinMaxScaler()
                                        factor_values_normalized = scaler.fit_transform(factor_values.values.reshape(-1, 1)).flatten()

                                        # 使用第二个y轴（对应row=2）
                                        fig.add_trace(
                                            go.Scatter(
                                                x=dates,
                                                y=factor_values_normalized,
                                                mode='lines',
                                                name=f"{factor_name} (归一化)",
                                                line=dict(color=colors[factor_idx % len(colors)], width=1.5)
                                            ),
                                            row=2, col=1
                                        )

                                # 更新布局
                                fig.update_layout(
                                    height=700,
                                    hovermode='x unified',
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    ),
                                    # 价格曲线不显示x轴标题
                                    xaxis=dict(showticklabels=True),
                                    # 因子曲线不显示标题
                                    xaxis2=dict(showticklabels=True),
                                    yaxis=dict(title='价格', side='left'),
                                    yaxis2=dict(
                                        title='因子值 (归一化)',
                                        side='left',
                                        showgrid=True
                                    ),
                                    margin=dict(l=60, r=60, t=40, b=60)
                                )

                                st.plotly_chart(fig, use_container_width=True)

            # 检查是否有交易记录
            if "trades" in result and result["trades"] is not None:
                trades_df = result["trades"]

                if len(trades_df) > 0:
                    st.markdown("---")
                    # 显示统计信息
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("总交易次数", len(trades_df))
                    with col2:
                        profit_count = len(trades_df[trades_df["收益"] > 0]) if "收益" in trades_df.columns else 0
                        st.metric("盈利交易", profit_count)
                    with col3:
                        loss_count = len(trades_df[trades_df["收益"] < 0]) if "收益" in trades_df.columns else 0
                        st.metric("亏损交易", loss_count)
                    with col4:
                        win_rate = profit_count / len(trades_df) if len(trades_df) > 0 else 0
                        st.metric("胜率", f"{win_rate:.2%}")

                    st.markdown("---")

                    # 交易记录表格
                    st.markdown("### 交易明细")

                    # 隐藏的列
                    hidden_columns = ['方向', '状态', 'Exit Trade Id', 'Position Id']

                    # 常见字段配置
                    field_configs = {
                        "交易ID": st.column_config.NumberColumn("交易ID", format="%.0f", width="small"),
                        "股票代码": st.column_config.TextColumn("股票代码", width="small"),
                        "数量": st.column_config.NumberColumn("数量", format="%.2f", width="small"),
                        "入场价格": st.column_config.NumberColumn("入场价格", format="%.2f", width="medium"),
                        "入场手续费": st.column_config.NumberColumn("入场手续费", format="%.2f", width="small"),
                        "出场时间": st.column_config.TextColumn("出场时间", width="medium"),
                        "出场价格": st.column_config.NumberColumn("出场价格", format="%.2f", width="medium"),
                        "出场手续费": st.column_config.NumberColumn("出场手续费", format="%.2f", width="small"),
                        "收益": st.column_config.NumberColumn("收益", format="%.2f", width="medium"),
                        "收益率": st.column_config.NumberColumn("收益率", format="%.4f", width="small"),
                        "父ID": st.column_config.NumberColumn("父ID", format="%.0f", width="small"),
                        "价值": st.column_config.NumberColumn("价值", format="%.2f", width="medium"),
                    }

                    # 过滤掉隐藏的列
                    display_columns = [col for col in trades_df.columns if col not in hidden_columns]

                    # 为显示的列创建配置
                    column_config = {}
                    for col in display_columns:
                        column_config[col] = field_configs.get(col, st.column_config.TextColumn(col, width="medium"))

                    st.dataframe(
                        trades_df[display_columns],
                        column_config=column_config if column_config else None,
                        use_container_width=True,
                        height=400,
                    )

                    # 提供下载选项
                    csv = trades_df[display_columns].to_csv()
                    st.download_button(
                        label="📥 下载交易日志 (CSV)",
                        data=csv,
                        file_name=f"trades_{config['start_date']}_to_{config['end_date']}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("回测期间没有产生交易记录")
            else:
                st.info("""
                **交易日志功能说明**：

                当前回测结果未包含详细的交易记录。这可能是因为：
                1. 使用的是 VectorBT 的 stats() 方法，该方法主要返回性能指标
                2. 需要在回测时显式提取交易记录

                **建议**：如需查看详细交易记录，可以：
                - 查看"净值分析"标签了解整体表现
                - 查看"风险分析"标签了解风险指标
                - 联系开发者添加交易记录提取功能
                """)

        with tab5:
            st.markdown("### 持仓分析")

            # 检查是否有回测结果
            if "backtest_result" in st.session_state and st.session_state.backtest_result:
                result = st.session_state.backtest_result

                # 检查是否有持仓数据
                if "positions" in result and result["positions"] is not None:
                    from backend.services.position_analysis_service import position_analysis_service

                    positions = result["positions"]
                    initial_capital = result.get("initial_capital", 1000000)

                    # 执行持仓分析
                    with st.spinner("正在分析持仓数据..."):
                        analysis = position_analysis_service.analyze_positions(
                            positions=positions,
                            initial_capital=initial_capital
                        )

                        concentration = position_analysis_service.calculate_position_concentration(
                            positions=positions
                        )

                        position_history = position_analysis_service.analyze_position_history(
                            positions=positions,
                            window=20
                        )

                    # 1. 基础统计指标
                    st.markdown("#### 📊 基础持仓统计")

                    basic_stats = analysis["basic_stats"]
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "平均持仓",
                            f"{basic_stats['avg_position']:.2%}",
                            help="平均持仓比例"
                        )
                    with col2:
                        st.metric(
                            "最大持仓",
                            f"{basic_stats['max_position']:.2%}",
                            help="最大持仓比例"
                        )
                    with col3:
                        st.metric(
                            "空仓天数占比",
                            f"{basic_stats['position_zero_ratio']:.2%}",
                            help="空仓天数占总天数的比例"
                        )
                    with col4:
                        st.metric(
                            "满仓天数占比",
                            f"{basic_stats['position_full_ratio']:.2%}",
                            help="满仓（>=90%）天数占比"
                        )

                    st.markdown("---")

                    # 2. 持仓时长统计
                    st.markdown("#### ⏱️ 持仓时长统计")

                    holding_stats = analysis["holding_stats"]
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "持仓时段数",
                            holding_stats["invested_periods"],
                            help="独立的持仓时段数量"
                        )
                    with col2:
                        st.metric(
                            "总持仓天数",
                            holding_stats["total_invested_days"],
                            help="实际持仓的天数总和"
                        )
                    with col3:
                        st.metric(
                            "平均持仓周期",
                            f"{holding_stats['avg_holding_period']:.1f} 天",
                            help="每个持仓时段平均持续天数"
                        )

                    st.markdown("---")

                    # 3. 持仓价值分析
                    st.markdown("#### 💰 持仓价值分析")

                    position_values = analysis["position_values"]
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "平均持仓价值",
                            f"¥{position_values['avg_position_value']:,.0f}",
                            help="平均持仓金额"
                        )
                    with col2:
                        st.metric(
                            "最大持仓价值",
                            f"¥{position_values['max_position_value']:,.0f}",
                            help="最大持仓金额"
                        )

                    st.markdown("---")

                    # 4. 持仓集中度分析
                    st.markdown("#### 🎯 持仓集中度分析")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "集中度比率",
                            f"{concentration['concentration_ratio']:.2%}",
                            help="前N大持仓占比（单股票为100%）"
                        )
                    with col2:
                        st.metric(
                            "HHI指数",
                            f"{concentration['herfindahl_index']:.4f}",
                            help="赫芬达尔-赫希曼指数（1表示完全集中）"
                        )
                    with col3:
                        st.metric(
                            "基尼系数",
                            f"{concentration['gini_coefficient']:.4f}",
                            help="0表示完全平均，1表示完全不平均"
                        )

                    # HHI走势图
                    if len(position_history) > 0:
                        st.markdown("##### HHI 指数走势")

                        # 计算滚动HHI
                        hhi_series = pd.Series(index=positions.index, dtype=float)
                        for i in range(len(positions)):
                            if i >= 20:
                                window_positions = positions.iloc[i-20:i+1]
                                # 单股票情况，HHI始终为1
                                hki_val = (window_positions.abs().values ** 2).sum() / (
                                    window_positions.abs().sum() ** 2 if window_positions.abs().sum() > 0 else 1
                                )
                                hhi_series.iloc[i] = hki_val

                        fig_hhi = go.Figure()
                        fig_hhi.add_trace(go.Scatter(
                            x=hhi_series.index,
                            y=hhi_series.values,
                            mode='lines',
                            name='HHI指数',
                            line=dict(color='#FF6B6B', width=2)
                        ))

                        fig_hhi.update_layout(
                            title="持仓集中度（HHI）走势",
                            xaxis_title="日期",
                            yaxis_title="HHI指数",
                            height=400,
                            hovermode='x unified',
                            margin=dict(l=60, r=60, t=40, b=60)
                        )

                        st.plotly_chart(fig_hhi, use_container_width=True)

                    st.markdown("---")

                    # 5. 换手率分析
                    st.markdown("#### 🔄 换手率分析")

                    turnover_stats = analysis["position_changes"]
                    total_turnover = analysis["turnover"]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "总换手率",
                            f"{total_turnover:.2%}",
                            help="累计换手率"
                        )
                    with col2:
                        st.metric(
                            "平均持仓变化",
                            f"{turnover_stats['avg_position_change']:.2%}",
                            help="每日平均持仓变化幅度"
                        )
                    with col3:
                        st.metric(
                            "最大持仓变化",
                            f"{turnover_stats['max_position_change']:.2%}",
                            help="单日最大持仓变化幅度"
                        )

                    # 换手率走势图
                    if len(position_history) > 0:
                        st.markdown("##### 换手率走势")

                        fig_turnover = go.Figure()

                        fig_turnover.add_trace(go.Scatter(
                            x=position_history.index,
                            y=position_history["position_change"],
                            mode='lines',
                            name='持仓变化',
                            line=dict(color='#4ECDC4', width=1.5),
                            fill='tozeroy',
                            fillcolor='rgba(78, 205, 196, 0.2)'
                        ))

                        # 添加滚动平均
                        rolling_avg = position_history["position_change"].rolling(window=20).mean()
                        fig_turnover.add_trace(go.Scatter(
                            x=position_history.index,
                            y=rolling_avg,
                            mode='lines',
                            name='20日平均',
                            line=dict(color='#FF6B6B', width=2, dash='dash')
                        ))

                        fig_turnover.update_layout(
                            title="持仓变化（换手率）走势",
                            xaxis_title="日期",
                            yaxis_title="持仓变化比例",
                            height=400,
                            hovermode='x unified',
                            margin=dict(l=60, r=60, t=40, b=60),
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )

                        st.plotly_chart(fig_turnover, use_container_width=True)

                    # 换手率分布直方图
                    st.markdown("##### 换手率分布")

                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=position_history["position_change"].dropna(),
                        nbinsx=50,
                        name='换手率分布',
                        marker_color='#4ECDC4',
                        opacity=0.75
                    ))

                    fig_hist.update_layout(
                        title="持仓变化分布直方图",
                        xaxis_title="持仓变化比例",
                        yaxis_title="频数",
                        height=350,
                        margin=dict(l=60, r=60, t=40, b=60)
                    )

                    st.plotly_chart(fig_hist, use_container_width=True)

                    st.markdown("---")

                    # 换手率成本分析
                    st.markdown("#### 💰 换手率成本分析")
                    st.info("💡 估算交易成本对收益的影响")

                    # 获取回测配置中的费率
                    config = st.session_state.backtest_result.get("config", {})
                    commission_rate = config.get("commission_rate", 0.0003)  # 默认万三
                    slippage = config.get("slippage", 0.0)  # 滑点

                    # 计算换手率成本
                    position_changes = position_history["position_change"].dropna()
                    total_turnover = position_changes.sum()

                    # 成本估算
                    # 1. 手续费成本
                    commission_cost = total_turnover * commission_rate * initial_capital

                    # 2. 滑点成本（简化估算）
                    if slippage > 0:
                        # 假设每次换手有一半买入一半卖出
                        slippage_cost = total_turnover * slippage * initial_capital * 0.5
                    else:
                        slippage_cost = 0

                    # 3. 总成本
                    total_cost = commission_cost + slippage_cost

                    # 4. 年化成本（按交易日252天）
                    n_days = len(position_changes)
                    if n_days > 0:
                        annualized_cost = total_cost * (252 / n_days)
                    else:
                        annualized_cost = 0

                    # 5. 成本占初始资金比例
                    cost_ratio = total_cost / initial_capital if initial_capital > 0 else 0

                    # 显示成本统计
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "手续费成本",
                            f"¥{commission_cost:,.0f}",
                            help=f"费率 {commission_rate:.4%} × 换手率 × 初始资金"
                        )
                    with col2:
                        st.metric(
                            "滑点成本",
                            f"¥{slippage_cost:,.0f}",
                            help=f"滑点 {slippage:.4%} × 换手率 × 初始资金 × 0.5"
                        )
                    with col3:
                        st.metric(
                            "总交易成本",
                            f"¥{total_cost:,.0f}",
                            help="手续费 + 滑点"
                        )
                    with col4:
                        st.metric(
                            "成本占比",
                            f"{cost_ratio:.2%}",
                            help="总成本占初始资金的比例"
                        )

                    # 成本分解饼图
                    st.markdown("##### 成本构成")

                    fig_cost = go.Figure()
                    fig_cost.add_trace(go.Pie(
                        labels=['手续费成本', '滑点成本'],
                        values=[commission_cost, slippage_cost],
                        marker=dict(colors=['#FF6B6B', '#4ECDC4']),
                        textinfo='percent+label',
                        hovertemplate='%{label}: ¥%{value:,.0f}<extra></extra>'
                    ))

                    fig_cost.update_layout(
                        title="交易成本构成",
                        height=400,
                        margin=dict(l=60, r=60, t=40, b=60)
                    )

                    st.plotly_chart(fig_cost, use_container_width=True)

                    # 成本走势（累计）
                    st.markdown("##### 累计成本走势")

                    # 计算每日成本
                    daily_cost = (
                        position_changes * commission_rate * initial_capital +
                        (position_changes * slippage * initial_capital * 0.5 if slippage > 0 else 0)
                    )
                    cumulative_cost = daily_cost.cumsum()

                    fig_cumulative = go.Figure()
                    fig_cumulative.add_trace(go.Scatter(
                        x=cumulative_cost.index,
                        y=cumulative_cost.values,
                        mode='lines',
                        name='累计成本',
                        line=dict(color='#FF6B6B', width=2),
                        fill='tozeroy'
                    ))

                    fig_cumulative.update_layout(
                        title="累计交易成本走势",
                        xaxis_title="日期",
                        yaxis_title="累计成本（元）",
                        height=400,
                        hovermode='x unified',
                        margin=dict(l=60, r=60, t=40, b=60)
                    )

                    st.plotly_chart(fig_cumulative, use_container_width=True)

                    # 成本解释和建议
                    with st.expander("💡 成本分析与优化建议", expanded=False):
                        st.markdown(f"""
                        ### 交易成本构成

                        **1. 手续费成本**
                        - 费率：{commission_rate:.4%}（万{commission_rate*10000:.0f}）
                        - 计算：换手率 × 费率 × 初始资金
                        - 影响：每1%换手率产生 {commission_rate*100:.2%} 的成本

                        **2. 滑点成本**
                        - 滑点率：{slippage:.4%}
                        - 计算：换手率 × 滑点率 × 初始资金 × 0.5
                        - 说明：假设买入和卖出各承担一半滑点

                        **成本评估**：
                        - 总成本：¥{total_cost:,.0f}
                        - 成本占比：{cost_ratio:.2%}（占初始资金）
                        - 年化成本：¥{annualized_cost:,.0f}

                        ### 优化建议

                        {'⚠️ 高成本警告：换手率较高，交易成本占比超过5%！建议降低调仓频率。' if cost_ratio > 0.05 else '✅ 成本合理：交易成本在可接受范围内。' if cost_ratio > 0.02 else '✅ 低成本：交易成本控制良好。'}

                        **降低成本的方法**：
                        1. **降低调仓频率**：从高频调整为中频或低频
                        2. **优化调仓策略**：只在因子信号强时调仓
                        3. **批量交易**：减少小额频繁交易
                        4. **选择低费率券商**：降低佣金费率
                        5. **使用算法交易**：减少市场冲击成本
                        """)

                    st.markdown("---")

                    # 6. 持仓数量统计
                    st.markdown("#### 📈 持仓数量统计")

                    # 计算每日持仓数量（单股票为1或0）
                    position_count = (positions.abs() > 0).astype(int)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "平均持仓数",
                            f"{position_count.mean():.2f}",
                            help="每日平均持仓股票数量"
                        )
                    with col2:
                        st.metric(
                            "最大持仓数",
                            f"{position_count.max():.0f}",
                            help="最多持仓股票数量"
                        )
                    with col3:
                        st.metric(
                            "最小持仓数",
                            f"{position_count.min():.0f}",
                            help="最少持仓股票数量"
                        )

                    # 持仓数量走势图
                    st.markdown("##### 持仓数量走势")

                    fig_count = go.Figure()
                    fig_count.add_trace(go.Scatter(
                        x=position_count.index,
                        y=position_count.values,
                        mode='lines',
                        name='持仓数量',
                        line=dict(color='#95E1D3', width=2),
                        fill='tozeroy'
                    ))

                    fig_count.update_layout(
                        title="持仓数量随时间变化",
                        xaxis_title="日期",
                        yaxis_title="持仓股票数量",
                        height=400,
                        hovermode='x unified',
                        margin=dict(l=60, r=60, t=40, b=60)
                    )

                    st.plotly_chart(fig_count, use_container_width=True)

                    # 功能说明
                    with st.expander("📖 持仓分析指标说明", expanded=False):
                        st.markdown("""
                        ### 持仓分析指标解释

                        **基础持仓统计**：
                        - **平均持仓**：所有交易日持仓比例的平均值
                        - **最大持仓**：历史最大持仓比例
                        - **空仓天数占比**：空仓天数占总交易日的比例
                        - **满仓天数占比**：持仓>=90%的天数占比

                        **持仓时长统计**：
                        - **持仓时段数**：独立持仓时段的数量（从开仓到平仓为一个时段）
                        - **总持仓天数**：所有持仓时段的实际天数总和
                        - **平均持仓周期**：每个持仓时段平均持续的天数

                        **持仓集中度**：
                        - **HHI指数（赫芬达尔-赫希曼指数）**：
                          - 取值范围：0到1
                          - 1表示完全集中（单只股票满仓）
                          - 0表示完全分散（无限多股票均仓）
                          - 数值越大表示持仓越集中

                        **换手率**：
                        - **持仓变化**：当日持仓与昨日持仓的绝对差值
                        - **总换手率**：所有交易日持仓变化的累计总和
                        - **换手率越高**表示交易越频繁，交易成本越高

                        **注意**：当前为单股票回测，部分指标（如HHI、集中度）会显示极端值。
                        多股票组合回测时，这些指标将更有参考价值。
                        """)

                else:
                    st.info("""
                    **持仓数据未找到**

                    当前回测结果中未包含持仓数据。这可能是因为：
                    1. 回测结果不完整
                    2. 使用了不支持持仓分析的回测方法

                    **建议**：请重新运行回测，或查看其他分析标签。
                    """)
            else:
                st.info("""
                **请先运行回测**

                持仓分析需要基于回测结果进行。请先配置参数并运行回测。

                **操作步骤**：
                1. 在左侧配置回测参数
                2. 选择股票代码和因子
                3. 点击"运行回测"按钮
                4. 回测完成后，回到此标签查看持仓分析
                """)


# ============ 主应用 ============
def main():
    """主应用入口"""

    # 侧边栏导航
    with st.sidebar:
        st.title("FactorFlow")
        st.caption("股票因子分析系统")

        page = st.radio(
            "选择功能",
            ["因子管理", "因子分析", "因子挖掘", "组合分析", "策略回测"],
            label_visibility="collapsed",
        )

        st.divider()

        # 缓存统计（所有页面显示）
        with st.expander("💾 缓存统计", expanded=False):
            cache_stats = data_service.get_cache_stats()

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "缓存命中率",
                    f"{cache_stats.get('hit_rate', 0):.1%}",
                    help="缓存命中的比例"
                )
            with col2:
                st.metric(
                    "缓存数量",
                    cache_stats.get('active_count', 0),
                    help="当前有效的缓存数量"
                )

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "总大小",
                    f"{cache_stats.get('total_size_mb', 0):.2f} MB",
                    help="所有缓存文件的总大小"
                )
            with col2:
                st.metric(
                    "过期数量",
                    cache_stats.get('expired_count', 0),
                    help="已过期的缓存数量"
                )

            # 缓存管理按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧹 清理过期", use_container_width=True):
                    cleaned = data_service.cleanup_cache()
                    st.success(f"已清理 {cleaned} 个过期缓存")
                    st.rerun()
            with col2:
                if st.button("🗑️ 清空全部", use_container_width=True):
                    cleared = data_service.clear_cache()
                    st.success(f"已清空 {cleared} 个缓存")
                    st.rerun()

            st.caption(f"命中: {cache_stats.get('hits', 0)} | 未中: {cache_stats.get('misses', 0)}")

        st.divider()

        # 因子管理页面显示统计
        if page == "因子管理":
            stats = factor_service.get_factor_stats()
            st.metric("预置因子", stats["preset_count"])
            st.metric("用户因子", stats["user_count"])
            st.metric("总因子数", stats["total_count"])

    # 因子管理页面
    if page == "因子管理":
        st.title("因子管理")

        tab1, tab2, tab3 = st.tabs(["因子列表", "新增因子", "批量生成"])

        # Tab1: 因子列表
        with tab1:
            # 顶部工具栏
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown("### 因子列表")
            with col2:
                factor_count = len(factor_service.get_all_factors())
                st.metric("因子总数", factor_count)
            with col3:
                if st.button("🔄 刷新", use_container_width=True, key="refresh_factor_list"):
                    st.rerun()

            st.markdown("---")

            factors = factor_service.get_all_factors()

            if not factors:
                st.info("暂无因子，请先在系统初始化时加载预置因子或添加自定义因子")
            else:
                # 按分类分组显示
                factor_df = pd.DataFrame(factors)
                if not factor_df.empty:
                    factor_df["来源"] = factor_df["source"].map({"preset": "预置", "user": "自定义"})

                    # 分类筛选
                    categories = ["全部"] + factor_df["category"].unique().tolist()
                    selected_category = st.selectbox("筛选分类", categories)

                    if selected_category != "全部":
                        display_df = factor_df[factor_df["category"] == selected_category]
                    else:
                        display_df = factor_df

                    # 来源筛选
                    source_filter = st.multiselect(
                        "筛选来源", ["预置", "自定义"], default=["预置", "自定义"]
                    )
                    if source_filter:
                        display_df = display_df[display_df["来源"].isin(source_filter)]

                    # 显示选项
                    show_formula = st.checkbox("在表格中显示公式", value=False, key="show_formula_in_table")

                    # 因子列表数据表格
                    if show_formula:
                        columns_to_show = ["name", "category", "来源", "code", "description"]
                        column_config = {
                            "name": st.column_config.TextColumn("因子名称", width="medium"),
                            "category": st.column_config.TextColumn("分类", width="small"),
                            "来源": st.column_config.TextColumn("来源", width="small"),
                            "code": st.column_config.TextColumn("公式", width="large"),
                            "description": st.column_config.TextColumn("说明", width="large"),
                        }
                    else:
                        columns_to_show = ["name", "category", "来源", "description"]
                        column_config = {
                            "name": st.column_config.TextColumn("因子名称", width="medium"),
                            "category": st.column_config.TextColumn("分类", width="small"),
                            "来源": st.column_config.TextColumn("来源", width="small"),
                            "description": st.column_config.TextColumn("说明", width="large"),
                        }

                    st.dataframe(
                        display_df[columns_to_show],
                        column_config=column_config,
                        hide_index=True,
                        use_container_width=True,
                    )

                    # 因子公式查看器
                    with st.expander("📝 查看因子公式", expanded=False):
                        selected_factor_name = st.selectbox(
                            "选择因子查看公式",
                            options=display_df["name"].tolist(),
                            key="view_formula_select",
                        )

                        if selected_factor_name:
                            factor_info = display_df[display_df["name"] == selected_factor_name].iloc[0]
                            st.markdown(f"**因子名称**: {factor_info['name']}")
                            st.markdown(f"**分类**: {factor_info['category']}")
                            st.markdown(f"**说明**: {factor_info['description']}")

                            # 显示公式代码
                            if "code" in factor_info and pd.notna(factor_info["code"]):
                                st.markdown("**计算公式**:")
                                st.code(
                                    factor_info["code"],
                                    language="python",
                                    line_numbers=False,
                                )
                            else:
                                st.warning("该因子没有公式代码")

                    # 删除因子（仅限用户自定义）
                    with st.expander("删除因子", expanded=False):
                        user_factors = factor_df[factor_df["source"] == "user"]
                        if not user_factors.empty:
                            factor_to_delete = st.selectbox(
                                "选择要删除的因子",
                                options=user_factors["name"].tolist(),
                                key="delete_factor_select",
                            )
                            delete_col1, delete_col2 = st.columns(2)
                            with delete_col1:
                                if st.button("确认删除", type="primary", key="btn_delete_factor"):
                                    try:
                                        # 查找因子ID
                                        factor_id = user_factors[user_factors["name"] == factor_to_delete]["id"].iloc[0]
                                        factor_service.delete_factor(factor_id)
                                        # 设置成功消息到session_state
                                        st.session_state["delete_success"] = f"因子 '{factor_to_delete}' 已删除"
                                        st.rerun()
                                    except Exception as e:
                                        st.session_state["delete_error"] = str(e)
                                        st.rerun()

                            # 显示操作结果
                            if "delete_success" in st.session_state:
                                st.success(st.session_state["delete_success"])
                                # 清除消息，避免重复显示
                                del st.session_state["delete_success"]
                            elif "delete_error" in st.session_state:
                                st.error(f"删除失败: {st.session_state['delete_error']}")
                                del st.session_state["delete_error"]
                        else:
                            st.info("没有可删除的用户自定义因子")

        # Tab2: 新增因子
        with tab2:
            st.subheader("创建自定义因子")

            # 代码模式选择
            code_mode = st.radio(
                "代码模式",
                ["简单表达式", "自定义函数"],
                horizontal=True,
                help="简单表达式：单行公式；自定义函数：支持复杂逻辑的Python函数"
            )

            # 自定义函数模式下的模板选择（在form外部）
            if code_mode == "自定义函数":
                st.markdown("**快速加载模板**")
                template_col1, template_col2, template_col3 = st.columns(3)

                # 定义模板代码
                templates = {
                    "trend": '''def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """多因子组合趋势指标"""
    # 计算RSI
    rsi = RSI(df["close"], timeperiod=14)

    # 计算MACD
    macd, signal, hist = MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)

    # 计算ADX
    adx = ADX(df["high"], df["low"], df["close"], timeperiod=14)

    # 组合信号：RSI > 50 且 MACD > signal 且 ADX > 25
    trend_score = (
        (rsi > 50).astype(int) +
        (macd > signal).astype(int) +
        (adx > 25).astype(int)
    )

    return trend_score''',
                    "volatility": '''def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """动态波动率因子"""
    # 计算对数收益率
    log_ret = np.log(df["close"] / df["close"].shift(1))

    # 计算短期波动率（10日）
    vol_short = log_ret.rolling(window=10).std()

    # 计算长期波动率（60日）
    vol_long = log_ret.rolling(window=60).std()

    # 波动率相对水平
    vol_ratio = vol_short / vol_long

    # 标记高波动期
    high_vol_threshold = vol_ratio.rolling(window=252).quantile(0.7)
    is_high_vol = (vol_ratio > high_vol_threshold).astype(float)

    return is_high_vol''',
                    "pattern": '''def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """K线形态识别因子"""
    # 计算实体大小
    body = abs(df["close"] - df["open"])

    # 计算上下影线
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]

    # 识别十字星（实体很小，上下影线较长）
    avg_body = body.rolling(window=20).mean()
    is_doji = (
        (body < 0.3 * avg_body) &
        (upper_shadow > 0.5 * body) &
        (lower_shadow > 0.5 * body)
    ).astype(float)

    # 识别大阳线
    avg_range = (df["high"] - df["low"]).rolling(window=20).mean()
    is_bullish = (
        (df["close"] > df["open"]) &
        (body > 1.5 * avg_body) &
        ((df["close"] - df["low"]) / (df["high"] - df["low"]) > 0.6)
    ).astype(float)

    # 组合信号
    return is_doji * 0.5 + is_bullish * 1.0'''
                }

                with template_col1:
                    if st.button("📈 加载趋势模板", use_container_width=True):
                        st.session_state["factor_code_template"] = templates["trend"]
                        st.rerun()

                with template_col2:
                    if st.button("📊 加载波动率模板", use_container_width=True):
                        st.session_state["factor_code_template"] = templates["volatility"]
                        st.rerun()

                with template_col3:
                    if st.button("🕯️ 加载K线形态模板", use_container_width=True):
                        st.session_state["factor_code_template"] = templates["pattern"]
                        st.rerun()

                st.markdown("---")

            with st.form("create_factor"):
                factor_name = st.text_input(
                    "因子名称",
                    placeholder="例如：my_custom_factor",
                    help="请使用英文名称，仅包含字母、数字和下划线",
                )

                if code_mode == "简单表达式":
                    st.markdown("**因子公式代码**")
                    st.markdown("""
                    可用变量：
                    - `open`, `high`, `low`, `close`, `volume`: OHLCV数据
                    - `np`: numpy模块
                    - `SMA`, `EMA`, `RSI`, `MACD`, `ADX`, `CCI`, `ATR`, `BBANDS`, `OBV`: TA-Lib函数

                    示例：
                    ```python
                    # 简单移动平均比率
                    close / SMA(close, timeperiod=20)

                    # 价格动量
                    (close - close.shift(10)) / close.shift(10)

                    # 波动率
                    np.log(close / close.shift(1)).rolling(window=20).std()
                    ```
                    """)

                    factor_code = st.text_area(
                        "因子计算公式",
                        placeholder="close / SMA(close, timeperiod=20)",
                        height=100,
                    )

                else:  # 自定义函数模式
                    st.markdown("**因子计算函数**")
                    st.markdown("""
                    可用变量：
                    - `df`: 包含OHLCV数据的DataFrame，包含列 `open`, `high`, `low`, `close`, `volume`
                    - `np`: numpy模块
                    - `pd`: pandas模块
                    - `SMA`, `EMA`, `RSI`, `MACD`, `ADX`, `CCI`, `ATR`, `BBANDS`, `OBV`: TA-Lib函数

                    **函数签名**：
                    ```python
                    def calculate_factor(df: pd.DataFrame) -> pd.Series:
                        \"\"\"
                        df: 包含OHLCV数据的DataFrame
                        返回: 与df长度相同的因子值Series
                        \"\"\"
                        # 你的计算逻辑
                        return factor_values
                    ```
                    """)

                    # 显示代码编辑器
                    # 使用 session_state 直接控制值
                    if "factor_code_input" not in st.session_state:
                        # 初始默认代码
                        st.session_state.factor_code_input = """def calculate_factor(df: pd.DataFrame) -> pd.Series:
    \"\"\"
    自定义因子计算函数

    Args:
        df: 包含OHLCV数据的DataFrame

    Returns:
        因子值Series，长度与df相同
    \"\"\"
    # 示例：计算价格相对于20日均线的偏离度
    sma20 = SMA(df["close"], timeperiod=20)
    deviation = (df["close"] - sma20) / sma20

    return deviation"""

                    # 如果有模板加载，更新值
                    if "factor_code_template" in st.session_state:
                        st.session_state.factor_code_input = st.session_state.factor_code_template
                        # 清除模板标记，避免重复加载
                        del st.session_state["factor_code_template"]

                    factor_code = st.text_area(
                        "因子计算函数代码",
                        height=300,
                        help="完整的Python函数，必须返回pd.Series对象",
                        key="factor_code_input"
                    )

                factor_description = st.text_area(
                    "因子说明",
                    placeholder="简要描述因子的含义和用途...",
                    height=80,
                )

                col1, col2 = st.columns(2)
                with col1:
                    validate_btn = st.form_submit_button("验证代码", type="secondary")
                with col2:
                    create_btn = st.form_submit_button("创建因子", type="primary")

                # 验证按钮
                if validate_btn:
                    if not factor_name or not factor_code:
                        st.warning("请填写因子名称和计算公式")
                    else:
                        is_valid, message = factor_service.validate_factor_code(factor_code)
                        if is_valid:
                            st.success(message)
                        else:
                            st.error(message)

                # 创建按钮
                if create_btn:
                    if not factor_name or not factor_code:
                        st.warning("请填写因子名称和计算公式")
                    else:
                        try:
                            factor_service.create_factor(
                                name=factor_name,
                                code=factor_code,
                                description=factor_description,
                            )
                            # 设置成功消息到session_state
                            st.session_state["create_success"] = f"因子 '{factor_name}' 创建成功！切换到「因子列表」标签页查看。"
                            st.rerun()
                        except ValueError as e:
                            st.session_state["create_error"] = str(e)
                            st.rerun()

            # 显示创建操作的反馈消息（在form外部）
            if "create_success" in st.session_state:
                st.success(st.session_state["create_success"])
                # 清除消息
                del st.session_state["create_success"]
            elif "create_error" in st.session_state:
                st.error(st.session_state["create_error"])
                del st.session_state["create_error"]

        # Tab3: 批量生成因子
        with tab3:
            st.subheader("批量生成因子")
            st.info("💡 基于现有因子通过组合、统计变换等方式批量生成新因子")

            # 侧边栏配置
            with st.sidebar:
                st.markdown("### 生成配置")

                # 选择基础因子
                all_factors_list = factor_service.get_all_factors()
                base_factor_names = [f["name"] for f in all_factors_list]

                selected_base_factors = st.multiselect(
                    "选择基础因子",
                    options=base_factor_names,
                    default=base_factor_names[:5] if len(base_factor_names) >= 5 else base_factor_names,
                    help="选择用于生成新因子的基础因子",
                    key="gen_base_factors"
                )

                # 数据配置（用于预筛选）
                st.markdown("### 数据配置（预筛选用）")
                stock_code_for_test = st.text_input(
                    "测试股票代码",
                    value="000001",
                    help="用于因子预筛选计算的测试股票代码"
                )

                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "开始日期",
                        value=datetime.now() - timedelta(days=365),
                        key="gen_start_date"
                    )
                with col2:
                    end_date = st.date_input(
                        "结束日期",
                        value=datetime.now(),
                        key="gen_end_date"
                    )

                # 生成方式
                st.markdown("### 生成方式")
                gen_binary = st.checkbox("二元运算组合", value=True, help="如: close + sma_20")
                gen_statistical = st.checkbox("统计变换", value=True, help="如: rank(close), zscore(volume)")
                gen_technical = st.checkbox("技术指标", value=False, help="如: SMA, EMA, RSI")

                # 生成参数
                st.markdown("### 生成参数")
                max_depth = st.slider("最大嵌套层数", 1, 3, 2, help="表达式嵌套的最大深度")
                max_combinations = st.slider("最大生成数量", 10, 200, 50, help="最多生成的因子数量")

                # 生成按钮
                generate_btn = st.button("🚀 开始生成", type="primary", use_container_width=True, key="generate_factors")

            # 主内容区
            if not selected_base_factors:
                st.warning("⚠️ 请至少选择一个基础因子")
            else:
                st.info(f"👈 配置生成参数后点击「开始生成」")

                # 显示预计生成的因子数量
                estimated_count = 0
                if gen_binary:
                    estimated_count += min(len(selected_base_factors) * (len(selected_base_factors) - 1) * 4, max_combinations)
                if gen_statistical:
                    estimated_count += min(len(selected_base_factors) * 6, max_combinations // 2)
                if gen_technical:
                    estimated_count += min(len(selected_base_factors) * 4, max_combinations // 4)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("预计生成数量", min(estimated_count, max_combinations))
                with col2:
                    st.metric("基础因子数", len(selected_base_factors))

            # 执行生成
            if generate_btn and selected_base_factors:
                with st.spinner("正在生成因子，请稍候..."):
                    try:
                        from backend.services.factor_generator_service import factor_generator_service

                        generated_factors = []

                        # 1. 二元运算组合
                        if gen_binary:
                            st.info("🔄 生成二元运算组合...")
                            binary_exprs = factor_generator_service.generate_binary_combinations(
                                selected_base_factors,
                                max_depth=max_depth,
                                max_combinations=max_combinations // (2 if gen_statistical else 1)
                            )

                            for i, expr in enumerate(binary_exprs):
                                generated_factors.append({
                                    "name": f"binary_{i+1}",
                                    "expression": expr,
                                    "type": "二元运算"
                                })

                        # 2. 统计变换
                        if gen_statistical:
                            st.info("🔄 生成统计变换因子...")
                            stat_exprs = factor_generator_service.generate_statistical_combinations(
                                selected_base_factors,
                                max_combinations=max_combinations // 2
                            )

                            for i, expr in enumerate(stat_exprs):
                                generated_factors.append({
                                    "name": f"stat_{i+1}",
                                    "expression": expr,
                                    "type": "统计变换"
                                })

                        # 3. 技术指标
                        if gen_technical:
                            st.info("🔄 生成技术指标因子...")
                            tech_exprs = factor_generator_service.generate_indicator_combinations(
                                selected_base_factors,
                                max_combinations=max_combinations // 4
                            )

                            for i, expr in enumerate(tech_exprs):
                                generated_factors.append({
                                    "name": f"tech_{i+1}",
                                    "expression": expr,
                                    "type": "技术指标"
                                })

                        st.success(f"✅ 成功生成 {len(generated_factors)} 个因子！")

                        # 因子预筛选功能
                        st.markdown("---")
                        st.subheader("🔍 因子预筛选")
                        st.info("💡 基于IC、IR等指标筛选有潜力的因子")

                        # 筛选参数配置
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            ic_threshold = st.number_input(
                                "IC绝对值阈值",
                                min_value=0.01,
                                max_value=0.1,
                                value=0.03,
                                step=0.01,
                                help="选择IC绝对值大于等于此阈值的因子"
                            )
                        with col2:
                            ir_threshold = st.number_input(
                                "IR阈值",
                                min_value=0.1,
                                max_value=5.0,
                                value=0.5,
                                step=0.1,
                                help="选择IR大于等于此阈值的因子"
                            )
                        with col3:
                            min_valid_ratio = st.number_input(
                                "最小有效数据比例",
                                min_value=0.5,
                                max_value=1.0,
                                value=0.7,
                                step=0.05,
                                help="选择有效数据比例大于等于此阈值的因子"
                            )

                        # 执行预筛选按钮
                        if st.button("🚀 执行预筛选", type="primary", key="preselect_factors"):
                            try:
                                from backend.services.factor_generator_service import factor_generator_service

                                # 获取用于计算的股票数据
                                with st.spinner("正在获取测试数据..."):
                                    test_data = data_service.get_stock_data(
                                        stock_code_for_test,
                                        start_date.strftime("%Y-%m-%d"),
                                        end_date.strftime("%Y-%m-%d")
                                    )
                                    if test_data.empty or "close" not in test_data.columns:
                                        st.error("❌ 无法获取测试数据，请检查股票代码和日期范围")
                                        display_selected = False
                                    else:
                                        returns = test_data["close"].pct_change().dropna()  # 计算收益率
                                        st.success(f"✅ 数据获取成功：{len(test_data)} 条记录")

                                        # 准备因子数据
                                        factor_data_map = {}
                                        valid_factors = []
                                        for factor_info in generated_factors:
                                            expr = factor_info["expression"]
                                            try:
                                                factor_result = factor_service.calculate_factor(
                                                    factor_name="temp",
                                                    data=test_data,
                                                    custom_code=expr  # 使用自定义表达式计算
                                                )
                                                if factor_result and "factor_values" in factor_result:
                                                    factor_data_map[expr] = factor_result["factor_values"]["factor_value"].dropna()
                                                    valid_factors.append(factor_info)
                                            except Exception as e:
                                                st.warning(f"⚠️ 因子 {expr} 计算失败，跳过: {str(e)}")
                                                continue

                                        if factor_data_map:
                                            # 执行预筛选
                                            selected_factors = factor_generator_service.preselect_factors(
                                                factors=valid_factors,
                                                factor_data_map=factor_data_map,
                                                return_data=returns,
                                                ic_threshold=ic_threshold,
                                                ir_threshold=ir_threshold,
                                                min_valid_ratio=min_valid_ratio
                                            )

                                            st.success(f"✅ 筛选完成！从 {len(generated_factors)} 个因子中筛选出 {len(selected_factors)} 个优质因子")

                                            # 存储筛选后的因子到session_state
                                            st.session_state.selected_factors = selected_factors

                                            # 显示筛选结果
                                            display_selected = True
                                        else:
                                            st.warning("⚠️ 没有有效因子可筛选")
                                            display_selected = False

                            except Exception as e:
                                st.error(f"❌ 预筛选失败: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())
                                display_selected = False
                        else:
                            display_selected = False

                        # 显示筛选结果（如果有）
                        if "selected_factors" in st.session_state and len(st.session_state.selected_factors) > 0:
                            if not display_selected:
                                st.markdown("---")
                                st.subheader("筛选结果")
                                st.info(f"💡 已筛选出 {len(st.session_state.selected_factors)} 个优质因子")

                            # 显示筛选后的因子
                            selected_df = pd.DataFrame(st.session_state.selected_factors)

                            # 分页显示
                            page_size = 20
                            total_pages = (len(selected_df) + page_size - 1) // page_size

                            if total_pages > 1:
                                page = st.number_input("页码", min_value=1, max_value=total_pages, value=1, key="selected_page")
                                start_idx = (page - 1) * page_size
                                end_idx = min(start_idx + page_size, len(selected_df))
                                display_df = selected_df.iloc[start_idx:end_idx]
                            else:
                                display_df = selected_df

                            st.dataframe(
                                display_df,
                                column_config={
                                    "name": st.column_config.TextColumn("名称", width="medium"),
                                    "expression": st.column_config.TextColumn("表达式", width="large"),
                                    "type": st.column_config.TextColumn("类型", width="small"),
                                    "ic": st.column_config.NumberColumn("IC", format="%.4f"),
                                    "ir": st.column_config.NumberColumn("IR", format="%.4f"),
                                    "valid_ratio": st.column_config.NumberColumn("有效数据", format="%.2%")
                                },
                                use_container_width=True,
                                hide_index=True
                            )

                            # 筛选结果的批量操作
                            col_save, col_export, col_clear = st.columns(3)
                            with col_save:
                                if st.button("💾 保存筛选结果", type="primary", key="save_selected_factors"):
                                    saved_count = 0
                                    failed_count = 0
                                    for factor in st.session_state.selected_factors:
                                        try:
                                            factor_service.create_factor(
                                                name=f"sel_{factor['name']}",
                                                code=factor['expression'],
                                                description=f"筛选出的{factor['type']}因子: {factor['expression']}"
                                            )
                                            saved_count += 1
                                        except Exception as e:
                                            failed_count += 1

                                    st.success(f"✅ 成功保存 {saved_count} 个因子，失败 {failed_count} 个")

                            with col_export:
                                if st.button("📥 导出筛选结果为CSV", key="export_selected_csv"):
                                    import io
                                    output = io.StringIO()
                                    export_df = pd.DataFrame(st.session_state.selected_factors)
                                    export_df.to_csv(output, index=False, encoding='utf-8-sig')
                                    st.download_button(
                                        label="⬇️ 下载CSV",
                                        data=output.getvalue(),
                                        file_name=f"selected_factors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        key="download_selected_csv"
                                    )

                            with col_clear:
                                if st.button("🗑️ 清空筛选结果", key="clear_selected"):
                                    if "selected_factors" in st.session_state:
                                        del st.session_state.selected_factors
                                    st.rerun()

                        # 显示生成的因子（完整列表）
                        st.markdown("---")
                        st.subheader("📋 所有生成的因子")

                        # 使用session_state存储生成的因子
                        if "generated_factors" not in st.session_state:
                            st.session_state.generated_factors = []

                        st.session_state.generated_factors = generated_factors

                        # 显示因子列表
                        if generated_factors:
                            # 创建DataFrame显示
                            gen_df = pd.DataFrame(generated_factors)

                            # 分页显示
                            page_size = 20
                            total_pages = (len(gen_df) + page_size - 1) // page_size

                            if total_pages > 1:
                                page = st.number_input("页码", min_value=1, max_value=total_pages, value=1)
                                start_idx = (page - 1) * page_size
                                end_idx = min(start_idx + page_size, len(gen_df))
                                display_df = gen_df.iloc[start_idx:end_idx]
                            else:
                                display_df = gen_df

                            st.dataframe(
                                display_df,
                                column_config={
                                    "name": st.column_config.TextColumn("名称", width="medium"),
                                    "expression": st.column_config.TextColumn("表达式", width="large"),
                                    "type": st.column_config.TextColumn("类型", width="small")
                                },
                                use_container_width=True,
                                hide_index=True
                            )

                            # 批量操作
                            st.markdown("---")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                if st.button("💾 全部保存到因子库", type="primary", key="save_all_generated"):
                                    saved_count = 0
                                    failed_count = 0
                                    for factor in generated_factors:
                                        try:
                                            factor_service.create_factor(
                                                name=f"gen_{factor['name']}",
                                                code=factor['expression'],
                                                description=f"批量生成的{factor['type']}因子: {factor['expression']}"
                                            )
                                            saved_count += 1
                                        except Exception as e:
                                            failed_count += 1

                                    st.success(f"✅ 成功保存 {saved_count} 个因子，失败 {failed_count} 个")

                            with col2:
                                if st.button("📥 导出为CSV", key="export_generated_csv"):
                                    import io
                                    output = io.StringIO()
                                    pd.DataFrame(generated_factors)[["name", "expression", "type"]].to_csv(output, index=False, encoding='utf-8-sig')
                                    st.download_button(
                                        label="⬇️ 下载CSV",
                                        data=output.getvalue(),
                                        file_name=f"generated_factors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        key="download_gen_csv"
                                    )

                            with col3:
                                if st.button("🗑️ 清空", key="clear_generated"):
                                    st.session_state.generated_factors = []
                                    st.rerun()
                        else:
                            st.warning("⚠️ 未生成任何因子，请检查配置")

                    except Exception as e:
                        st.error(f"❌ 生成失败: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())

    # 因子分析页面
    elif page == "因子分析":
        st.title("因子分析")

        # 侧边栏配置
        with st.sidebar:
            st.subheader("分析配置")

            # 数据模式
            data_mode = st.radio("数据模式", ["单股票", "股票池"])

            # 股票代码输入
            if data_mode == "单股票":
                stock_codes_input = st.text_input(
                    "股票代码",
                    value="000001",
                    help="支持格式：000001 或 000001.SZ（深圳）, 600000（上海）",
                )
                stock_codes = [stock_codes_input.strip()]
            else:
                stock_codes_input = st.text_area(
                    "股票代码（每行一个）",
                    value="000001\n600000",
                    height=100,
                )
                stock_codes = [code.strip() for code in stock_codes_input.strip().split("\n") if code.strip()]

            # 选择因子
            all_factors = factor_service.get_all_factors()
            factor_options = {f["name"]: f["description"] for f in all_factors}
            selected_factors = st.multiselect(
                "选择因子",
                options=list(factor_options.keys()),
                format_func=lambda x: f"{x} - {factor_options.get(x, '')}",
                default=[],
            )

            # 时间范围
            default_end = datetime.now()
            default_start = default_end - timedelta(days=365)

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("开始日期", value=default_start)
            with col2:
                end_date = st.date_input("结束日期", value=default_end)

            # 因子中性化选项
            st.markdown("---")
            st.markdown("### 因子中性化")
            enable_neutralization = st.checkbox("启用因子中性化", value=False, help="去除市值或行业对因子的影响")
            neutralization_method = st.selectbox(
                "中性化方法",
                ["无", "市值中性化", "行业中性化"],
                index=0,
                disabled=not enable_neutralization
            )

            # 分析按钮
            analyze_btn = st.button("📊 开始分析", type="primary", use_container_width=True)

        # 执行分析
        if analyze_btn:
            if not stock_codes or not stock_codes[0]:
                st.error("请输入股票代码")
            elif not selected_factors:
                st.error("请至少选择一个因子")
            else:
                with st.spinner("正在执行因子分析，请稍候..."):
                    try:
                        results = analysis_service.analyze(
                            stock_codes=stock_codes,
                            factor_names=selected_factors,
                            start_date=start_date.strftime("%Y-%m-%d"),
                            end_date=end_date.strftime("%Y-%m-%d"),
                            use_cache=True,
                        )

                        # 应用因子中性化
                        if enable_neutralization and neutralization_method != "无":
                            from backend.services.factor_neutralization_service import factor_neutralization_service

                            st.info(f"🔄 正在应用{neutralization_method}...")

                            # 对每个因子应用中性化
                            for factor_name in selected_factors:
                                if factor_name in results.get("factor_data", {}):
                                    factor_df = results["factor_data"][factor_name]

                                    try:
                                        if neutralization_method == "市值中性化":
                                            # 检查是否有市值列
                                            if "market_cap" in factor_df.columns:
                                                neutralized = factor_neutralization_service.neutralize_market_cap(
                                                    factor_df,
                                                    factor_name,
                                                    "market_cap"
                                                )
                                                factor_df["factor_value"] = neutralized
                                                st.info(f"✅ 因子 {factor_name} 已应用市值中性化")
                                            else:
                                                st.warning(f"⚠️ 因子 {factor_name} 缺少市值数据，跳过中性化")

                                        elif neutralization_method == "行业中性化":
                                            # 检查是否有行业列
                                            if "industry" in factor_df.columns:
                                                neutralized = factor_neutralization_service.neutralize_industry(
                                                    factor_df,
                                                    factor_name,
                                                    "industry"
                                                )
                                                factor_df["factor_value"] = neutralized
                                                st.info(f"✅ 因子 {factor_name} 已应用行业中性化")
                                            else:
                                                st.warning(f"⚠️ 因子 {factor_name} 缺少行业数据，跳过中性化")

                                    except Exception as e:
                                        st.warning(f"⚠️ 因子 {factor_name} 中性化失败: {str(e)}")

                        # 保存到session state供其他页面使用
                        st.session_state["analysis_results"] = results
                        st.session_state["analysis_stock_codes"] = stock_codes
                        st.session_state["analysis_factors"] = selected_factors

                        st.success("✅ 分析完成！")

                    except Exception as e:
                        st.error(f"❌ 分析失败: {e}")
                        import traceback
                        st.error(traceback.format_exc())

        # 显示分析结果
        if "analysis_results" in st.session_state:
            results = st.session_state["analysis_results"]
            factor_data = results.get("factor_data", {})
            stock_codes_result = results["metadata"]["stock_codes"]
            selected_factors_result = results["metadata"]["factor_names"]

            # Tab1: 行情预览, Tab2: SHAP分析, Tab3: 统计分析, Tab4: 导出报告, Tab5: 增强分析
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["行情预览", "SHAP分析", "统计分析", "导出报告", "增强分析"])

            with tab1:
                st.subheader("股票行情")

                # 价格走势图
                plot_price_chart(factor_data, stock_codes_result)

                # 因子叠加图（每个股票分tab显示）
                if len(stock_codes_result) > 1:
                    stock_tabs = st.tabs(stock_codes_result)
                    for stock_code, stock_tab in zip(stock_codes_result, stock_tabs):
                        with stock_tab:
                            if stock_code in factor_data:
                                st.write(f"**{stock_code}**")

                                # 因子选择
                                selected_plot_factor = st.selectbox(
                                    "选择要显示的因子",
                                    options=selected_factors_result,
                                    key=f"plot_factor_{stock_code}",
                                )
                                if selected_plot_factor:
                                    plot_factor_with_price(
                                        factor_data[stock_code],
                                        selected_plot_factor,
                                        stock_code,
                                    )

                                # 因子数据表格
                                with st.expander("查看因子数据"):
                                    display_cols = ["close", "open", "high", "low"] + selected_factors_result
                                    available_cols = [col for col in display_cols if col in factor_data[stock_code].columns]
                                    st.dataframe(
                                        factor_data[stock_code][available_cols].tail(100),
                                        use_container_width=True,
                                    )
                else:
                    stock_code = stock_codes_result[0]
                    if stock_code in factor_data:
                        selected_plot_factor = st.selectbox(
                            "选择要显示的因子",
                            options=selected_factors_result,
                            key="plot_factor_single",
                        )
                        if selected_plot_factor:
                            plot_factor_with_price(
                                factor_data[stock_code],
                                selected_plot_factor,
                                stock_code,
                            )

                        with st.expander("查看因子数据"):
                            display_cols = ["close", "open", "high", "low"] + selected_factors_result
                            available_cols = [col for col in display_cols if col in factor_data[stock_code].columns]
                            st.dataframe(
                                factor_data[stock_code][available_cols].tail(100),
                                use_container_width=True,
                            )

            with tab2:
                st.subheader("SHAP分析")

                # 检查SHAP是否可用
                try:
                    import shap
                    shap_available = True
                except ImportError:
                    shap_available = False

                if not shap_available:
                    st.error("❌ SHAP库未安装")
                    st.markdown("""
                    ### 安装SHAP库

                    SHAP是一个用于解释机器学习模型的Python库。

                    **安装方法**：

                    1. 使用pip安装：
                    ```bash
                    pip install shap
                    ```

                    2. 使用conda安装：
                    ```bash
                    conda install -c conda-forge shap
                    ```

                    **安装后刷新页面即可使用SHAP分析功能**
                    """)
                    st.info("💡 SHAP分析可以解释每个因子对收益率预测的贡献度，帮助识别最重要的因子。")
                else:
                    # SHAP分析说明
                    with st.expander("📖 SHAP分析方法说明", expanded=False):
                        st.markdown("""
                        ### SHAP (SHapley Additive exPlanations) 分析逻辑

                    **什么是SHAP？**
                    SHAP 是一种博弈论方法，用于解释机器学习模型的输出。它将预测结果分解为每个特征的贡献度。

                    ---

                    ### 📊 不同场景下的SHAP分析

                    #### 1️⃣ 单股票 × 单因子场景

                    **计算逻辑**：
                    ```
                    时间序列: t=1, 2, ..., T
                    因子值: f₁, f₂, ..., fₜ (已计算)
                    未来收益率: rₜ = close(t+5)/close(t) - 1

                    训练数据:
                    X = [f₁, f₂, ..., fₜ]  # 特征矩阵
                    y = [r₁, r₂, ..., rₜ]  # 目标变量

                    模型: XGBoost(X, y)
                    SHAP值: 贡献度(fᵢ → rₜ)
                    ```

                    **解释**：
                    - SHAP值表示该因子在每个时点对未来5日收益的贡献
                    - 适合分析单个因子对收益的预测能力
                    - 时序依赖性强，适合个股择时策略

                    **适用场景**：
                    - 评估单个因子的有效性
                    - 选择最有效的预测因子
                    - 构建单因子择时策略

                    #### 2️⃣ 单股票 × 多因子场景

                    **计算逻辑**：
                    ```
                    多个因子: f₁, f₂, ..., fₙ (例如：RSI, MACD, ADX)

                    训练数据:
                    X = [f₁, f₂, ..., fₙ]  # 多个因子作为特征
                    y = 未来5日收益率

                    模型: XGBoost(X, y)
                    SHAP值: 每个因子对预测的贡献度
                    ```

                    **解释**：
                    - 比较多个因子的相对重要性
                    - 识别冗余因子（SHAP值接近0）
                    - 发现因子之间的交互作用

                    **适用场景**：
                    - 构建多因子选股模型
                    - 因子组合优化
                    - 识别最佳因子组合

                    #### 3️⃣ 股票池 × 单因子场景

                    **计算逻辑**：
                    ```
                    横截面: t时刻，股票池 {s₁, s₂, ..., sₙ}
                    因子值: f(s₁), f(s₂), ..., f(sₙ)
                    收益率: r(s₁), r(s₂), ..., r(sₙ)

                    训练数据（合并所有股票和时刻）:
                    X = [f(sᵢ, tₖ)]  # (股票×时间) × 1
                    y = r(sᵢ, tₖ)     # (股票×时间) × 1

                    模型: XGBoost(X, y)
                    SHAP值: 该因子在所有股票和时刻的平均贡献度
                    ```

                    **解释**：
                    - 评估因子在横截面上的选股能力
                    - SHAP值高说明该因子能有效区分股票表现
                    - 与时序IC互补，综合评估因子有效性

                    **适用场景**：
                    - 因子的横截面有效性验证
                    - 选股因子筛选
                    - 股票组合构建

                    #### 4️⃣ 股票池 × 多因子场景（最复杂）

                    **计算逻辑**：
                    ```
                    多个因子: f₁, f₂, ..., fₙ
                    股票池: s₁, s₂, ..., sₙ
                    时间: t₁, t₂, ..., tₜ

                    训练数据:
                    X = [f₁(sᵢ, tₖ), f₂(sᵢ, tₖ), ..., fₙ(sᵢ, tₖ)]  # (股票×时间) × 因子数
                    y = r(sᵢ, tₖ)                                       # (股票×时间) × 1

                    模型: XGBoost(X, y)
                    SHAP值: 每个因子在所有股票和时刻的平均贡献度
                    ```

                    **解释**：
                    - 综合评估多因子组合的预测能力
                    - 识别最重要的因子
                    - 发现因子在不同股票和时间的稳定性

                    **适用场景**：
                    - 多因子选股模型
                    - 因子权重配置
                    - 稳健的量化策略构建

                    ---

                    ### 📈 输出指标说明

                    **1. 全局特征重要性**
                    - `importance = mean(|SHAP值|)`
                    - 值越大，因子越重要
                    - 可以识别出最重要的因子

                    **2. 模型R²得分**
                    - R² = 1 - RSS/TSS
                    - 衡量模型对收益率变动的解释比例
                    - R² > 0.05 说明因子有效
                    - R² > 0.1 说明因子很强

                    **3. 使用建议**
                    - **因子筛选**：选择SHAP重要性>阈值的因子
                    - **因子组合**：使用SHAP正相关的因子
                    - **风险评估**：注意SHAP不稳定的因子
                    - **策略构建**：结合IC/IR综合评估
                    """)

                    shap_data = results.get("shap", {})

                    if "error" in shap_data:
                        st.warning(shap_data["error"])
                    else:
                        # 特征重要性
                        if "feature_importance" in shap_data:
                            st.markdown("### 全局特征重要性")
                            plot_shap_importance(shap_data["feature_importance"])

                        # 模型得分
                        if "model_score" in shap_data:
                            st.metric("XGBoost模型R²得分", f"{shap_data['model_score']:.4f}")

            with tab3:
                st.subheader("统计分析")

                # 统计分析说明
                with st.expander("📖 IC/IR统计分析说明", expanded=False):
                    st.markdown("""
                    ### IC (Information Coefficient) 信息系数分析

                    **什么是IC？**
                    IC 是衡量因子值与未来收益率相关性的指标，反映因子的预测能力。

                    ---

                    ### 📊 不同场景下的IC/IR计算

                    #### 1️⃣ 单股票时序IC (Time-Series IC)

                    **计算公式**：
                    ```
                    股票 i，时间 t
                    因子值: fₜ
                    未来收益率: rₜ = close(t+5)/close(t) - 1

                    ICₜ = Corr(fₜ, rₜ)
                          = Cov(fₜ, rₜ) / (σ(fₜ) × σ(rₜ))
                    ```

                    **计算逻辑**：
                    - 使用滚动窗口（默认252天）计算时序IC
                    - 每个时点计算因子值与未来收益率的斯皮尔曼相关系数
                    - 得到IC序列：IC₁, IC₂, ..., ICₜ

                    **统计指标**：
                    - **IC均值**：`mean(IC)` - 衡量因子平均预测能力
                    - **IC标准差**：`std(IC)` - 衡量IC的波动程度
                    - **IR (信息比率)**：`mean(IC) / std(IC)` - 风险调整后的收益能力
                    - **IC>0占比**：`count(IC>0) / count(IC)` - 因子方向性

                    **解释**：
                    - IC均值绝对值 > 0.03 → 因子有效
                    - IC均值绝对值 > 0.05 → 因子很强
                    - IR > 0.5 → 优秀因子
                    - IR > 0.3 → 良好因子

                    **适用场景**：
                    - 评估因子的择时能力
                    - 单因子策略回测
                    - 时序策略因子选择

                    #### 2️⃣ 股票池横截面IC (Cross-Sectional IC)

                    **计算公式**：
                    ```
                    时间 t，股票池 {s₁, s₂, ..., sₙ}
                    因子值: f(s₁), f(s₂), ..., f(sₙ)
                    收益率: r(s₁), r(s₂), ..., r(sₙ)

                    ICₜ = Corr(f(s), r(s))
                          = Cov(f(s), r(s)) / (σ(f(s)) × σ(r(s)))
                    ```

                    **计算逻辑**：
                    - 在每个时点t，计算所有股票因子值与收益率的相关系数
                    - 得到IC序列：IC₁, IC₂, ..., ICₜ

                    **统计指标**：
                    - 计算方式与时序IC相同
                    - 但意义不同：衡量横截面选股能力

                    **解释**：
                    - IC高 → 因子能有效区分股票表现
                    - 适合选股策略，而非择时
                    - 横截面IC与时序IC互补

                    **适用场景**：
                    - 多因子选股模型
                    - 股票组合构建
                    - 因子横截面有效性验证

                    ---

                    ### 🔍 单因子 vs 多因子场景

                    #### 单因子场景
                    ```
                    计算某个因子 f₁ 的 IC/IR

                    时序IC（单股票）：
                      IC = Corr(f₁, future_return)
                      分析：因子f₁的预测能力

                    横截面IC（股票池）：
                      IC = Corr(f₁(s), future_return(s))
                      分析：因子f₁的选股能力
                    ```

                    #### 多因子场景
                    ```
                    计算多个因子 {f₁, f₂, ..., fₙ} 的IC/IR

                    方法1：分别计算每个因子的IC
                      IC(f₁), IC(f₂), ..., IC(fₙ)
                      比较哪个因子预测能力最强

                    方法2：计算因子组合的IC
                      先组合：F = w₁·f₁ + w₂·f₂ + ... + wₙ·fₙ
                      再计算：IC = Corr(F, future_return)
                      分析：因子组合是否比单因子更好
                    ```

                    ---

                    ### 📈 统计图表说明

                    **1. IC统计表格**
                    - 显示每个因子的IC均值、标准差、IR
                    - 用于快速比较因子性能

                    **2. 月度IC热力图**
                    - X轴：月份
                    - Y轴：因子
                    - 颜色：IC值（红色负值，绿色正值）
                    - 用于观察因子在不同时期的表现

                    **3. 滚动IR曲线**
                    - X轴：时间
                    - Y轴：滚动IR值
                    - 用于识别因子稳定性变化

                    **4. 因子相关性矩阵**
                    - 显示因子之间的相关系数
                    - 用于识别冗余因子
                    - 相关性高(>0.7)的因子可考虑去重

                    ---

                    ### 💡 使用建议

                    1. **因子筛选**
                       - 优先选择IC均值高且IR>0.5的因子
                       - IC<0.02的因子建议剔除

                    2. **因子组合**
                       - 选择相关性低、IC都高的因子
                       - 避免使用高度相关的因子

                    3. **稳定性分析**
                       - 观察滚动IR，识别因子失效期
                       - 月度IC热力图观察周期性

                    4. **策略构建**
                       - 时序IC高的因子适合择时策略
                       - 横截面IC高的因子适合选股策略
                       - 结合SHAP分析综合评估
                    """)

                ic_ir_data = results.get("ic_ir", {})

                # 相关性矩阵
                st.markdown("### 因子相关性矩阵")
                plot_correlation_matrix(factor_data, selected_factors_result)

                if not ic_ir_data:
                    st.info("暂无IC/IR统计数据")
                else:
                    # IC统计表格
                    if "ic_stats" in ic_ir_data:
                        st.markdown("### IC统计表格")
                        ic_stats = ic_ir_data["ic_stats"]

                        # 准备表格数据
                        table_data = []
                        for factor_name, stats in ic_stats.items():
                            table_data.append({
                                "因子名称": factor_name,
                                "IC均值": stats['IC均值'],
                                "IC标准差": stats['IC标准差'],
                                "IR": stats['IR'],
                                "IC>0占比": stats['IC>0占比'],
                                "IC绝对值均值": stats['IC绝对值均值'],
                            })

                        st.dataframe(
                            pd.DataFrame(table_data),
                            column_config={
                                "因子名称": st.column_config.TextColumn("因子名称", width="medium"),
                                "IC均值": st.column_config.NumberColumn(
                                    "IC均值",
                                    format="%.4f",
                                    help="📖 **IC均值**：衡量因子预测能力的平均水平\n\n- 绝对值越大，因子预测能力越强\n- >0.03 为有效因子，>0.05 为优秀因子\n- 负值表示反向预测"
                                ),
                                "IC标准差": st.column_config.NumberColumn(
                                    "IC标准差",
                                    format="%.4f",
                                    help="📖 **IC标准差**：IC值的波动程度\n\n- 标准差越小，因子表现越稳定\n- 用于计算IR（信息比率）\n- 过大的波动表明因子不稳定"
                                ),
                                "IR": st.column_config.NumberColumn(
                                    "IR (信息比率)",
                                    format="%.4f",
                                    help="📖 **IR (Information Ratio)**：IC均值与标准差的比值\n\n- 衡量因子的风险调整后收益能力\n- >0.5 为优秀因子，>0.3 为良好因子\n- 公式：IR = IC均值 / IC标准差"
                                ),
                                "IC>0占比": st.column_config.NumberColumn(
                                    "IC>0占比",
                                    format="%.2%",
                                    help="📖 **IC>0占比**：因子预测方向正确的比例\n\n- 衡量因子方向性预测的准确度\n- >50% 表示方向预测准确\n- 接近100%表示几乎总是方向正确"
                                ),
                                "IC绝对值均值": st.column_config.NumberColumn(
                                    "IC绝对值均值",
                                    format="%.4f",
                                    help="📖 **IC绝对值均值**：|IC|的平均值\n\n- 忽略正负方向，只看预测强度\n- 用于评估因子的整体预测能力\n- 比IC均值更能反映因子的真实预测力"
                                ),
                            },
                            use_container_width=True,
                            hide_index=True,
                        )

                    # IC时间序列
                    if "ic_stats" in ic_ir_data:
                        st.markdown("### IC时间序列")
                        ic_series_dict = {
                            name: stats["IC序列"]
                            for name, stats in ic_ir_data["ic_stats"].items()
                        }
                        plot_ic_time_series(ic_series_dict)

                    # 月度IC热力图
                    if "monthly_ic" in ic_ir_data:
                        st.markdown("### 月度IC热力图")
                        monthly_ic = ic_ir_data["monthly_ic"]
                        if not monthly_ic:
                            st.info("暂无月度IC数据")
                        elif len(monthly_ic) == 1:
                            plot_monthly_ic_heatmap(list(monthly_ic.values())[0], list(monthly_ic.keys())[0])
                        else:
                            monthly_tabs = st.tabs(list(monthly_ic.keys()))
                            for factor_name, monthly_tab in zip(monthly_ic.keys(), monthly_tabs):
                                with monthly_tab:
                                    plot_monthly_ic_heatmap(monthly_ic[factor_name], factor_name)

                    # 滚动窗口IR
                    if "rolling_ir" in ic_ir_data:
                        st.markdown("### 滚动窗口IR")
                        plot_rolling_ir(ic_ir_data["rolling_ir"])

            with tab4:
                st.subheader("导出报告")

                report = analysis_service.generate_report(results)

                st.markdown(report)

                if st.button("保存报告到文件", type="primary"):
                    output_path = analysis_service.export_report(results)
                    st.success(f"报告已保存到: {output_path}")

            with tab5:
                st.subheader("增强分析")
                st.info("💡 IC显著性检验、因子分布分析、因子衰减分析")

                from backend.services.enhanced_analysis_service import enhanced_analysis_service

                # IC显著性检验
                st.markdown("### IC显著性检验")

                with st.expander("📖 IC显著性检验说明", expanded=False):
                    st.markdown("""
                    **什么是IC显著性检验？**

                    IC显著性检验用于判断因子与收益率之间的相关性是否统计显著，而非偶然现象。

                    **检验方法**：
                    - **t检验**：检验IC均值是否显著不为0
                    - **p值**：p < 0.05 表示显著

                    **解释**：
                    - p < 0.01：高度显著（***）
                    - p < 0.05：显著（**）
                    - p < 0.1：边际显著（*）
                    """)

                try:
                    st.info(f"💡 数据结构信息：")
                    st.caption(f"因子数据 keys: {list(results.keys())}")

                    if "factor_data" in results:
                        st.caption(f"因子数量: {len(list(results['factor_data'].keys()))}")
                    if "ic_ir" in results:
                        ic_ir = results["ic_ir"]
                        st.caption(f"IC数据结构: {list(ic_ir.keys()) if ic_ir else 'None'}")
                        if "ic_stats" in ic_ir:
                            st.caption(f"IC统计中的因子: {list(ic_ir['ic_stats'].keys())}")

                    # 执行IC显著性检验
                    if "ic_ir" in results and results["ic_ir"] and "ic_stats" in results["ic_ir"]:
                        ic_stats = results["ic_ir"]["ic_stats"]
                        for factor_name in selected_factors_result:
                            if factor_name in ic_stats:
                                factor_stats = ic_stats[factor_name]

                                # 从所有股票的DataFrame中收集因子和收益率数据
                                all_factor_data = []
                                for stock_code, stock_df in results.get("factor_data", {}).items():
                                    if factor_name in stock_df.columns and "future_return_1" in stock_df.columns:
                                        temp_df = stock_df[[factor_name, "future_return_1"]].copy()
                                        temp_df = temp_df.dropna()
                                        if len(temp_df) > 0:
                                            all_factor_data.append(temp_df)

                                if all_factor_data:
                                    # 合并所有股票的数据
                                    combined_df = pd.concat(all_factor_data)
                                    factor_values = combined_df[factor_name]
                                    return_values = combined_df["future_return_1"]

                                    sig_result = enhanced_analysis_service.calculate_ic_significance(
                                        factor_values=factor_values,
                                        return_values=return_values
                                    )

                                    if "error" not in sig_result:
                                        # 显示结果
                                        col1, col2, col3, col4 = st.columns(4)

                                        with col1:
                                            st.metric(f"{factor_name} IC", f"{sig_result['ic']:.4f}")
                                        with col2:
                                            st.metric("t统计量", f"{sig_result['t_statistic']:.4f}")
                                        with col3:
                                            st.metric("p值", f"{sig_result['p_value']:.4f}")
                                        with col4:
                                            significance = (
                                                "***" if sig_result['p_value'] < 0.01 else
                                                "**" if sig_result['p_value'] < 0.05 else
                                                "*" if sig_result['p_value'] < 0.1 else
                                                ""
                                            )
                                            st.metric("显著性", significance)

                                        # 显示IC均值（从ic_stats获取）
                                        ic_mean = factor_stats.get("IC均值", 0)
                                        st.caption(f"IC均值: {ic_mean:.4f}")

                                    # IC分布直方图（使用IC序列）
                                    ic_series_dict = factor_stats.get("IC序列", {})
                                    if ic_series_dict:
                                        ic_series = pd.Series(ic_series_dict)
                                        ic_series.index = pd.to_datetime(ic_series.index)

                                        st.markdown(f"**{factor_name} - IC分布**")
                                        fig_ic_hist = go.Figure(data=[go.Histogram(
                                            x=ic_series.dropna(),
                                            nbinsx=30,
                                            marker_color='skyblue'
                                        )])
                                        fig_ic_hist.update_layout(
                                            title=f"{factor_name} IC值分布",
                                            xaxis_title="IC值",
                                            yaxis_title="频数",
                                            height=300
                                        )
                                        st.plotly_chart(fig_ic_hist, use_container_width=True)

                                        st.markdown("---")

                except Exception as e:
                    st.warning(f"⚠️ IC显著性检验失败: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

                # 因子分布分析
                st.markdown("### 因子分布分析")

                try:
                    st.info(f"💡 调试信息：selected_factors_result = {selected_factors_result}")
                    st.info(f"💡 调试信息：factor_data keys = {list(results.get('factor_data', {}).keys())}")

                    # factor_data 结构是 {股票代码: DataFrame}
                    # 每个DataFrame包含所有因子的列
                    for factor_name in selected_factors_result:
                        st.info(f"💡 处理因子：{factor_name}")

                        # 从所有股票的DataFrame中收集该因子的值
                        all_factor_values = []
                        for stock_code, stock_df in results.get("factor_data", {}).items():
                            if factor_name in stock_df.columns:
                                factor_values = stock_df[factor_name].dropna()
                                all_factor_values.append(factor_values)
                                st.caption(f"股票 {stock_code}: {len(factor_values)} 个有效值")

                        if all_factor_values:
                            # 合并所有股票的因子值
                            combined_factor_values = pd.concat(all_factor_values)
                            st.info(f"💡 合并后的因子值数量: {len(combined_factor_values)}")

                            # 分布统计
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("均值", f"{combined_factor_values.mean():.4f}")
                            with col2:
                                st.metric("标准差", f"{combined_factor_values.std():.4f}")
                            with col3:
                                st.metric("偏度", f"{combined_factor_values.skew():.4f}")
                            with col4:
                                st.metric("峰度", f"{combined_factor_values.kurtosis():.4f}")

                            # 分布直方图
                            fig_dist = go.Figure(data=[go.Histogram(
                                x=combined_factor_values,
                                nbinsx=50,
                                marker_color='lightblue',
                                name='因子值分布'
                            )])
                            fig_dist.update_layout(
                                title=f"{factor_name} - 因子值分布",
                                xaxis_title="因子值",
                                yaxis_title="频数",
                                height=400
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)

                            st.markdown("---")
                        else:
                            st.warning(f"⚠️ 因子 {factor_name} 在所有股票的数据中都未找到")

                except Exception as e:
                    st.warning(f"⚠️ 因子分布分析失败: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

                # 因子衰减分析
                st.markdown("### 因子衰减分析")
                st.info("💡 分析因子预测能力随时间衰减的情况")

                try:
                    for factor_name in selected_factors_result:
                        # 从所有股票的DataFrame中收集因子和收益率数据
                        all_factor_data = []

                        for stock_code, stock_df in results.get("factor_data", {}).items():
                            if factor_name in stock_df.columns and "future_return_1" in stock_df.columns:
                                # 合并因子和收益率数据
                                temp_df = stock_df[[factor_name, "future_return_1"]].copy()
                                temp_df = temp_df.dropna()
                                if len(temp_df) > 20:  # 确保有足够数据
                                    all_factor_data.append(temp_df)

                        if all_factor_data:
                            # 合并所有股票的数据
                            combined_df = pd.concat(all_factor_data)

                            # 计算不同滞后的IC
                            lags = [1, 2, 3, 5, 10, 20]
                            ic_by_lag = []

                            for lag in lags:
                                # 计算滞后收益
                                factor_df_aligned = combined_df.copy()
                                factor_df_aligned["future_return"] = factor_df_aligned["future_return_1"].shift(-lag)

                                # 计算IC
                                aligned = factor_df_aligned[[factor_name, "future_return"]].dropna()
                                if len(aligned) > 10:
                                    ic = aligned[factor_name].corr(aligned["future_return"])
                                    ic_by_lag.append({
                                        "滞后天数": lag,
                                        "IC": ic
                                    })

                            if ic_by_lag:
                                decay_df = pd.DataFrame(ic_by_lag)

                                # 衰减曲线
                                fig_decay = go.Figure()
                                fig_decay.add_trace(go.Scatter(
                                    x=decay_df["滞后天数"],
                                    y=decay_df["IC"],
                                    mode='lines+markers',
                                    name='IC衰减',
                                    line=dict(color='red', width=2)
                                ))
                                fig_decay.update_layout(
                                    title=f"{factor_name} - IC衰减分析",
                                    xaxis_title="滞后天数",
                                    yaxis_title="IC",
                                    height=400
                                )
                                st.plotly_chart(fig_decay, use_container_width=True)

                                # 衰减表格
                                st.dataframe(
                                    decay_df,
                                    column_config={
                                        "滞后天数": st.column_config.TextColumn("滞后天数"),
                                        "IC": st.column_config.TextColumn("IC值")
                                    },
                                    use_container_width=True,
                                    hide_index=True
                                )

                                st.markdown("---")
                            else:
                                st.warning(f"⚠️ 因子 {factor_name} 数据不足，无法进行衰减分析")
                        else:
                            st.warning(f"⚠️ 因子 {factor_name} 在所有股票的数据中都未找到")

                except Exception as e:
                    st.warning(f"⚠️ 因子衰减分析失败: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

                # 因子稳定性检验
                st.markdown("### 因子稳定性检验")
                st.info("💡 分析因子在不同时间段和市场环境下的稳定性")

                from backend.services.factor_stability_service import factor_stability_service

                with st.expander("📖 因子稳定性检验说明", expanded=False):
                    st.markdown("""
                    **什么是因子稳定性检验？**

                    因子稳定性检验用于评估因子在不同时间段和市场环境下的表现是否一致。

                    **检验方法**：
                    - **分布稳定性**：使用KS检验比较不同时间段的因子分布
                    - **时间序列稳定性**：使用ADF检验判断IC序列是否平稳
                    - **滚动窗口稳定性**：在不同窗口下计算IC统计
                    - **市场环境表现**：分析因子在牛市、熊市、震荡市的表现

                    **解释**：
                    - 稳定性得分越高，因子在不同时期表现越一致
                    - p值>0.05表示分布无显著差异，稳定性好
                    """)

                try:
                    for factor_name in selected_factors_result:
                        # 从所有股票的DataFrame中收集因子和收益率数据
                        all_factor_data = []

                        for stock_code, stock_df in results.get("factor_data", {}).items():
                            if factor_name in stock_df.columns and "future_return_1" in stock_df.columns:
                                temp_df = stock_df[[factor_name, "future_return_1"]].copy()
                                temp_df = temp_df.dropna()
                                if len(temp_df) > 20:  # 确保有足够数据
                                    all_factor_data.append(temp_df)

                        if all_factor_data:
                            # 合并所有股票的数据
                            combined_df = pd.concat(all_factor_data)
                            combined_df = combined_df.sort_index()

                            st.markdown(f"#### 📊 {factor_name} - 稳定性分析")

                            # 1. 分布稳定性检验
                            st.markdown("##### 分布稳定性（KS检验）")

                            try:
                                stability_result = factor_stability_service.calculate_distribution_stability(
                                    factor_series=combined_df[factor_name],
                                    window=min(252, len(combined_df) // 3),
                                    method="ks"
                                )

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("稳定性得分", f"{stability_result['stable_ratio']:.2%}")
                                with col2:
                                    st.metric("平均p值", f"{stability_result['avg_p_value']:.4f}")
                                with col3:
                                    st.metric("比较次数", stability_result['n_comparisons'])

                                # 解释
                                if stability_result['stable_ratio'] >= 0.8:
                                    st.success("✅ 因子分布高度稳定，不同时间段分布一致")
                                elif stability_result['stable_ratio'] >= 0.6:
                                    st.info("ℹ️ 因子分布中等稳定")
                                else:
                                    st.warning("⚠️ 因子分布不稳定，可能存在结构性变化")

                            except Exception as e:
                                st.warning(f"⚠️ 分布稳定性检验失败: {str(e)}")

                            # 2. 时间序列稳定性（ADF检验）
                            st.markdown("##### IC序列平稳性（ADF检验）")

                            try:
                                if "ic_ir" in results and "ic_stats" in results["ic_ir"]:
                                    ic_stats = results["ic_ir"]["ic_stats"]
                                    if factor_name in ic_stats:
                                        factor_stats = ic_stats[factor_name]
                                        ic_series_dict = factor_stats.get("IC序列", {})

                                        if ic_series_dict:
                                            ic_series = pd.Series(ic_series_dict)
                                            ic_series.index = pd.to_datetime(ic_series.index)

                                            if len(ic_series) > 20:
                                                adf_result = factor_stability_service.calculate_time_series_stability(
                                                    ic_series=ic_series,
                                                    maxlag=10
                                                )

                                                if "error" not in adf_result:
                                                    col1, col2, col3, col4 = st.columns(4)
                                                    with col1:
                                                        is_stationary = adf_result['is_stationary']
                                                        st.metric(
                                                            "是否平稳",
                                                            "是" if is_stationary else "否",
                                                            help="平稳序列更稳定"
                                                        )
                                                    with col2:
                                                        st.metric("ADF统计量", f"{adf_result['adf_statistic']:.4f}")
                                                    with col3:
                                                        st.metric("p值", f"{adf_result['p_value']:.4f}")
                                                    with col4:
                                                        st.metric("滞后阶数", adf_result['used_lag'])

                                                    # 解释
                                                    st.info(f"💡 {adf_result['interpretation']}")

                                                    # 临界值对比
                                                    with st.expander("查看临界值对比", expanded=False):
                                                        st.markdown("""
                                                        **ADF检验临界值**：
                                                        - 1%: {:.4f}
                                                        - 5%: {:.4f}
                                                        - 10%: {:.4f}

                                                        **判断规则**：
                                                        - ADF统计量 < 临界值 → 拒绝原假设 → 序列平稳
                                                        - p值 < 0.05 → 拒绝原假设 → 序列平稳
                                                        """.format(
                                                            adf_result['critical_values']['1%'],
                                                            adf_result['critical_values']['5%'],
                                                            adf_result['critical_values']['10%']
                                                        ))

                            except Exception as e:
                                st.warning(f"⚠️ 平稳性检验失败: {str(e)}")

                            # 3. 滚动窗口稳定性
                            st.markdown("##### 滚动窗口IC稳定性")

                            try:
                                rolling_stability = factor_stability_service.calculate_rolling_stability(
                                    factor_data=combined_df,
                                    factor_name=factor_name,
                                    return_col="return",
                                    windows=[20, 60, 120]
                                )

                                if rolling_stability:
                                    # 准备表格数据
                                    window_data = []
                                    for key, metrics in rolling_stability.items():
                                        window_data.append({
                                            "窗口": metrics['window'],
                                            "平均IC": f"{metrics['mean_ic']:.4f}",
                                            "IC标准差": f"{metrics['std_ic']:.4f}",
                                            "IR": f"{metrics['ir']:.4f}" if not pd.isna(metrics['ir']) else "N/A",
                                            "变异系数": f"{metrics['cv']:.4f}" if not pd.isna(metrics['cv']) else "N/A"
                                        })

                                    st.dataframe(
                                        pd.DataFrame(window_data),
                                        column_config={
                                            "窗口": st.column_config.TextColumn("窗口（天）"),
                                            "平均IC": st.column_config.TextColumn("平均IC"),
                                            "IC标准差": st.column_config.TextColumn("IC标准差"),
                                            "IR": st.column_config.TextColumn("信息比率"),
                                            "变异系数": st.column_config.TextColumn("变异系数")
                                        },
                                        use_container_width=True,
                                        hide_index=True
                                    )

                            except Exception as e:
                                st.warning(f"⚠️ 滚动窗口分析失败: {str(e)}")

                            # 4. 变异系数分析
                            st.markdown("##### IC变异系数")

                            try:
                                if "ic_ir" in results and "ic_stats" in results["ic_ir"]:
                                    ic_stats = results["ic_ir"]["ic_stats"]
                                    if factor_name in ic_stats:
                                        factor_stats = ic_stats[factor_name]
                                        ic_series_dict = factor_stats.get("IC序列", {})

                                        if ic_series_dict:
                                            ic_series = pd.Series(ic_series_dict)
                                            ic_series.index = pd.to_datetime(ic_series.index)

                                            cv_result = factor_stability_service.calculate_coefficient_of_variation(
                                                ic_series=ic_series
                                            )

                                            if "error" not in cv_result:
                                                col1, col2, col3 = st.columns(3)
                                                with col1:
                                                    st.metric("IC均值", f"{cv_result['mean']:.4f}")
                                                with col2:
                                                    st.metric("IC标准差", f"{cv_result['std']:.4f}")
                                                with col3:
                                                    st.metric("变异系数", f"{cv_result['cv']:.4f}")

                                                st.info(f"💡 {cv_result['interpretation']}")

                            except Exception as e:
                                st.warning(f"⚠️ 变异系数计算失败: {str(e)}")

                            st.markdown("---")
                        else:
                            st.warning(f"⚠️ 因子 {factor_name} 在所有股票的数据中都未找到")

                except Exception as e:
                    st.warning(f"⚠️ 因子稳定性检验失败: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

                # 因子多周期分析
                st.markdown("### 因子多周期分析")
                st.info("💡 分析因子在不同时间周期（日、周、月）下的表现差异")

                with st.expander("📖 多周期分析说明", expanded=False):
                    st.markdown("""
                    **什么是因子多周期分析？**

                    多周期分析用于评估因子在不同时间尺度下的预测能力和稳定性。

                    **分析维度**：
                    - **日周期**：原始日度数据的IC表现
                    - **周周期**：每周最后一个交易日的IC表现
                    - **月周期**：每月最后一个交易日的IC表现
                    - **季度周期**：每季最后一个交易日的IC表现

                    **应用场景**：
                    - 判断因子适合短期还是长期持有
                    - 识别因子的最佳调仓频率
                    - 评估因子在不同周期下的衰减速度
                    """)

                try:
                    for factor_name in selected_factors_result:
                        # 从所有股票的DataFrame中收集因子和收益率数据
                        all_factor_data = []

                        for stock_code, stock_df in results.get("factor_data", {}).items():
                            if factor_name in stock_df.columns and "future_return_1" in stock_df.columns:
                                temp_df = stock_df[[factor_name, "future_return_1"]].copy()
                                temp_df = temp_df.dropna()
                                if len(temp_df) > 20:  # 确保有足够数据
                                    all_factor_data.append(temp_df)

                        if all_factor_data:
                            # 合并所有股票的数据
                            combined_df = pd.concat(all_factor_data)
                            combined_df = combined_df.sort_index()

                            # 确保数据有日期索引
                            if not isinstance(combined_df.index, pd.DatetimeIndex):
                                try:
                                    combined_df.index = pd.to_datetime(combined_df.index)
                                except:
                                    st.warning(f"⚠️ {factor_name} 数据索引不是日期格式，无法进行多周期分析")
                                    continue

                            st.markdown(f"#### 📊 {factor_name} - 多周期分析")

                            # 计算不同周期的IC
                            periods_data = {}

                            # 1. 日周期（原始数据）
                            daily_ic = []
                            for i in range(len(combined_df) - 1):
                                if pd.notna(combined_df[factor_name].iloc[i]) and pd.notna(combined_df['future_return_1'].iloc[i]):
                                    # 使用当前因子值和下一期收益
                                    if i + 1 < len(combined_df):
                                        factor_val = combined_df[factor_name].iloc[i]
                                        return_val = combined_df['future_return_1'].iloc[i]
                                        if pd.notna(factor_val) and pd.notna(return_val):
                                            daily_ic.append((combined_df.index[i], factor_val, return_val))

                            if daily_ic:
                                daily_df = pd.DataFrame(daily_ic, columns=['date', 'factor', 'return'])
                                daily_ic_corr = daily_df['factor'].corr(daily_df['return'])
                                periods_data['日频'] = {
                                    'period': '日频',
                                    'n_obs': len(daily_df),
                                    'ic': float(daily_ic_corr) if not pd.isna(daily_ic_corr) else 0,
                                    'abs_ic': float(abs(daily_ic_corr)) if not pd.isna(daily_ic_corr) else 0
                                }

                            # 2. 周周期
                            weekly_data = combined_df.resample('W').last()
                            if len(weekly_data) > 10:
                                weekly_ic_series = []
                                for i in range(len(weekly_data) - 1):
                                    if pd.notna(weekly_data[factor_name].iloc[i]) and pd.notna(weekly_data['future_return_1'].iloc[i]):
                                        factor_val = weekly_data[factor_name].iloc[i]
                                        return_val = weekly_data['future_return_1'].iloc[i]
                                        if pd.notna(factor_val) and pd.notna(return_val):
                                            weekly_ic_series.append((factor_val, return_val))

                                if weekly_ic_series:
                                    weekly_df = pd.DataFrame(weekly_ic_series, columns=['factor', 'return'])
                                    weekly_ic_corr = weekly_df['factor'].corr(weekly_df['return'])
                                    periods_data['周频'] = {
                                        'period': '周频',
                                        'n_obs': len(weekly_df),
                                        'ic': float(weekly_ic_corr) if not pd.isna(weekly_ic_corr) else 0,
                                        'abs_ic': float(abs(weekly_ic_corr)) if not pd.isna(weekly_ic_corr) else 0
                                    }

                            # 3. 月周期
                            monthly_data = combined_df.resample('M').last()
                            if len(monthly_data) > 6:
                                monthly_ic_series = []
                                for i in range(len(monthly_data) - 1):
                                    if pd.notna(monthly_data[factor_name].iloc[i]) and pd.notna(monthly_data['future_return_1'].iloc[i]):
                                        factor_val = monthly_data[factor_name].iloc[i]
                                        return_val = monthly_data['return'].iloc[i]
                                        if pd.notna(factor_val) and pd.notna(return_val):
                                            monthly_ic_series.append((factor_val, return_val))

                                if monthly_ic_series:
                                    monthly_df = pd.DataFrame(monthly_ic_series, columns=['factor', 'return'])
                                    monthly_ic_corr = monthly_df['factor'].corr(monthly_df['return'])
                                    periods_data['月频'] = {
                                        'period': '月频',
                                        'n_obs': len(monthly_df),
                                        'ic': float(monthly_ic_corr) if not pd.isna(monthly_ic_corr) else 0,
                                        'abs_ic': float(abs(monthly_ic_corr)) if not pd.isna(monthly_ic_corr) else 0
                                    }

                            # 4. 季度周期
                            quarterly_data = combined_df.resample('Q').last()
                            if len(quarterly_data) > 4:
                                quarterly_ic_series = []
                                for i in range(len(quarterly_data) - 1):
                                    if pd.notna(quarterly_data[factor_name].iloc[i]) and pd.notna(quarterly_data['return'].iloc[i]):
                                        factor_val = quarterly_data[factor_name].iloc[i]
                                        return_val = quarterly_data['return'].iloc[i]
                                        if pd.notna(factor_val) and pd.notna(return_val):
                                            quarterly_ic_series.append((factor_val, return_val))

                                if quarterly_ic_series:
                                    quarterly_df = pd.DataFrame(quarterly_ic_series, columns=['factor', 'return'])
                                    quarterly_ic_corr = quarterly_df['factor'].corr(quarterly_df['return'])
                                    periods_data['季频'] = {
                                        'period': '季频',
                                        'n_obs': len(quarterly_df),
                                        'ic': float(quarterly_ic_corr) if not pd.isna(quarterly_ic_corr) else 0,
                                        'abs_ic': float(abs(quarterly_ic_corr)) if not pd.isna(quarterly_ic_corr) else 0
                                    }

                            # 显示结果
                            if periods_data:
                                # 定义评级函数
                                def get_period_rating(abs_ic):
                                    if abs_ic >= 0.05:
                                        return "优秀"
                                    elif abs_ic >= 0.03:
                                        return "良好"
                                    elif abs_ic >= 0.01:
                                        return "一般"
                                    else:
                                        return "较差"

                                # 创建对比表格
                                periods_df = pd.DataFrame([
                                    {
                                        "周期": data['period'],
                                        "观测数": data['n_obs'],
                                        "IC": f"{data['ic']:.4f}",
                                        "|IC|": f"{data['abs_ic']:.4f}",
                                        "评级": get_period_rating(data['abs_ic'])
                                    }
                                    for data in periods_data.values()
                                ])

                                st.dataframe(
                                    periods_df,
                                    column_config={
                                        "周期": st.column_config.TextColumn("周期"),
                                        "观测数": st.column_config.NumberColumn("观测数"),
                                        "IC": st.column_config.TextColumn("IC值"),
                                        "|IC|": st.column_config.TextColumn("|IC|"),
                                        "评级": st.column_config.TextColumn("评级")
                                    },
                                    use_container_width=True,
                                    hide_index=True
                                )

                                # IC对比柱状图
                                fig_period = go.Figure()
                                periods_list = list(periods_data.keys())
                                ic_values = [periods_data[p]['ic'] for p in periods_list]

                                colors = ['green' if ic > 0 else 'red' for ic in ic_values]

                                fig_period.add_trace(go.Bar(
                                    x=periods_list,
                                    y=ic_values,
                                    marker_color=colors,
                                    text=[f"{ic:.4f}" for ic in ic_values],
                                    textposition='outside'
                                ))

                                fig_period.update_layout(
                                    title=f"{factor_name} - 不同周期IC对比",
                                    xaxis_title="周期",
                                    yaxis_title="IC值",
                                    height=400,
                                )

                                st.plotly_chart(fig_period, use_container_width=True)

                                # 推荐调仓周期
                                best_period = max(periods_data.items(), key=lambda x: x[1]['abs_ic'])
                                st.info(f"💡 **推荐调仓周期**: {best_period[1]['period']}（|IC| = {best_period[1]['abs_ic']:.4f}）")

                                st.markdown("---")
                            else:
                                st.warning(f"⚠️ 因子 {factor_name} 数据不足，无法进行多周期分析")
                        else:
                            st.warning(f"⚠️ 因子 {factor_name} 在所有股票的数据中都未找到")

                except Exception as e:
                    st.warning(f"⚠️ 因子多周期分析失败: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

    # 因子挖掘页面
    elif page == "因子挖掘":
        genetic_mining_main()

    # 组合分析页面
    elif page == "组合分析":
        portfolio_analysis_main()

    # 策略回测页面
    elif page == "策略回测":
        backtest_main()


def backtest_comparison():
    """多策略对比"""
    st.markdown("### 多策略对比回测")
    st.info("💡 同时对比多个策略的表现，找出最优策略")

    # 侧边栏配置
    with st.sidebar:
        st.subheader("策略对比配置")

        # 数据配置
        data_mode = st.radio("数据模式", ["单股票", "股票池"], key="comp_data_mode")

        if data_mode == "单股票":
            stock_codes_input = st.text_input("股票代码", value="000001", key="comp_stock_input")
            stock_codes = [stock_codes_input.strip()]
        else:
            stock_codes_input = st.text_area("股票代码（每行一个）", value="000001\n600000", height=100, key="comp_stock_area")
            stock_codes = [code.strip() for code in stock_codes_input.strip().split("\n") if code.strip()]

        # 时间范围
        default_end = datetime.now()
        default_start = default_end - timedelta(days=365)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("开始日期", value=default_start, key="comp_start_date")
        with col2:
            end_date = st.date_input("结束日期", value=default_end, key="comp_end_date")

        # 选择要对比的策略
        st.markdown("### 选择策略（至少2个）")

        # 获取可用因子
        all_factors = factor_service.get_all_factors()
        factor_options = {f["name"]: f["description"] for f in all_factors}

        # 策略配置列表
        if "strategy_configs" not in st.session_state:
            st.session_state.strategy_configs = []

        # 添加策略按钮
        if st.button("➕ 添加策略", use_container_width=True, key="add_strategy"):
            st.session_state.strategy_configs.append({
                "name": f"策略{len(st.session_state.strategy_configs) + 1}",
                "factor": None,
                "top_pct": 20,
                "direction": "long"
            })

        # 显示已配置的策略
        st.markdown("**已配置的策略**:")
        for i, config in enumerate(st.session_state.strategy_configs):
            with st.expander(f"策略 {i+1}", expanded=True):
                config["name"] = st.text_input(
                    "策略名称",
                    value=config["name"],
                    key=f"strategy_name_{i}"
                )

                config["factor"] = st.selectbox(
                    "选择因子",
                    options=list(factor_options.keys()),
                    index=list(factor_options.keys()).index(config["factor"]) if config["factor"] in factor_options else 0,
                    format_func=lambda x: f"{x}",
                    key=f"strategy_factor_{i}"
                )

                config["top_pct"] = st.slider(
                    "选择前N%",
                    min_value=5,
                    max_value=50,
                    value=config["top_pct"],
                    key=f"strategy_top_{i}"
                )

                config["direction"] = st.selectbox(
                    "交易方向",
                    ["long", "short"],
                    index=0 if config["direction"] == "long" else 1,
                    key=f"strategy_direction_{i}"
                )

            col_del, col_spacer = st.columns([1, 4])
            with col_del:
                if st.button("🗑️", key=f"delete_strategy_{i}"):
                    st.session_state.strategy_configs.pop(i)
                    st.rerun()

        # 回测参数
        st.markdown("### 回测参数")
        initial_capital = st.number_input("初始资金", value=1000000, min_value=10000, step=10000, key="comp_initial")
        commission_rate = st.number_input("费率(%)", value=0.03, min_value=0.0, max_value=1.0, step=0.01, key="comp_comm") / 100
        rebalance_freq = st.selectbox("调仓频率", ["日频", "周频", "月频"], index=2, key="comp_rebalance")

        # 运行对比按钮
        run_comparison = st.button("🚀 运行策略对比", type="primary", use_container_width=True, key="run_comparison")

        # 清空配置按钮
        if st.button("清空所有策略", type="secondary", key="clear_strategies"):
            st.session_state.strategy_configs = []
            st.rerun()

    # 主内容区
    if len(st.session_state.strategy_configs) < 2:
        st.warning("⚠️ 请至少配置2个策略进行对比")
        st.info("👈 在左侧添加策略并配置参数")
        return

    if not run_comparison:
        st.info("👈 配置完成后点击「运行策略对比」按钮")
        return

    # 验证配置
    valid_configs = [c for c in st.session_state.strategy_configs if c.get("factor")]
    if len(valid_configs) < 2:
        st.error("❌ 至少需要2个有效配置的策略（每个策略必须选择因子）")
        return

    # 执行策略对比
    try:
        with st.spinner("正在执行策略对比，请稍候..."):
            # 获取数据
            all_data = {}
            for stock_code in stock_codes:
                data = data_service.get_stock_data(
                    stock_code,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )
                if not data.empty:
                    all_data[stock_code] = data

            if not all_data:
                st.error("❌ 未获取到任何数据")
                return

            # 合并所有股票数据
            merged_data = pd.concat(all_data.values(), names=["stock_code"]).reset_index()
            merged_data = merged_data.sort_values(["date", "stock_code"])

            # 计算所有策略的收益
            results = {}
            for config in valid_configs:
                strategy_name = config["name"]

                # 计算因子值
                factor_result = factor_service.calculate_factor(
                    factor_name=config["factor"],
                    data=merged_data
                )

                if factor_result and "factor_values" in factor_result:
                    factor_df = factor_result["factor_values"]

                    # 按日期分组，选择top N股票
                    selected_returns = []
                    for date, group in factor_df.groupby("date"):
                        if config["direction"] == "long":
                            top_stocks = group.nlargest(int(len(group) * config["top_pct"] / 100), "factor_value")
                        else:
                            top_stocks = group.nsmallest(int(len(group) * config["top_pct"] / 100), "factor_value")

                        # 计算下期收益
                        next_returns = merged_data[
                            (merged_data["date"] > date) &
                            (merged_data["stock_code"].isin(top_stocks["stock_code"]))
                        ].groupby("stock_code")["return"].first()

                        if len(next_returns) > 0:
                            selected_returns.append(next_returns.mean())

                    # 构建策略收益序列
                    if selected_returns:
                        returns_series = pd.Series(selected_returns)
                        results[strategy_name] = {
                            "returns": returns_series,
                            "total_return": (1 + returns_series).prod() - 1,
                            "annual_return": (1 + returns_series).mean() * 252,
                            "volatility": returns_series.std() * np.sqrt(252),
                        }

                        # 计算夏普比率
                        if results[strategy_name]["volatility"] > 0:
                            results[strategy_name]["sharpe_ratio"] = results[strategy_name]["annual_return"] / results[strategy_name]["volatility"]
                        else:
                            results[strategy_name]["sharpe_ratio"] = 0

        st.success("✅ 策略对比完成！")

        # 显示对比结果
        st.markdown("---")
        st.subheader("📊 策略对比结果")

        # 1. 指标对比表
        st.markdown("### 指标对比")

        metrics_data = []
        for name, result in results.items():
            metrics_data.append({
                "策略名称": name,
                "累计收益率": f"{result['total_return']:.2%}",
                "年化收益率": f"{result['annual_return']:.2%}",
                "年化波动率": f"{result['volatility']:.2%}",
                "夏普比率": f"{result['sharpe_ratio']:.4f}",
            })

        st.dataframe(
            pd.DataFrame(metrics_data),
            use_container_width=True,
            hide_index=True,
        )

        # 2. 收益率排名
        st.markdown("### 收益率排名")
        sorted_results = sorted(results.items(), key=lambda x: x[1]["total_return"], reverse=True)
        for i, (name, result) in enumerate(sorted_results, 1):
            st.metric(f"#{i} {name}", f"{result['total_return']:.2%}")

        # 3. 可视化对比
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 年化收益率对比")
            yield_data = {name: result["annual_return"] * 100 for name, result in results.items()}
            fig_yield = go.Figure(data=[
                go.Bar(
                    x=list(yield_data.keys()),
                    y=list(yield_data.values()),
                    marker_color=['green' if v > 0 else 'red' for v in yield_data.values()]
                )
            ])
            fig_yield.update_layout(
                xaxis_title="策略",
                yaxis_title="年化收益率(%)",
                height=400
            )
            st.plotly_chart(fig_yield, use_container_width=True)

        with col2:
            st.markdown("### 夏普比率对比")
            sharpe_data = {name: result["sharpe_ratio"] for name, result in results.items()}
            fig_sharpe = go.Figure(data=[
                go.Bar(
                    x=list(sharpe_data.keys()),
                    y=list(sharpe_data.values()),
                    marker_color=['blue' if v > 1 else 'orange' for v in sharpe_data.values()]
                )
            ])
            fig_sharpe.update_layout(
                xaxis_title="策略",
                yaxis_title="夏普比率",
                height=400
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)

        # 4. 风险收益散点图
        st.markdown("### 风险-收益分布")
        scatter_data = []
        for name, result in results.items():
            scatter_data.append({
                "x": result["volatility"] * 100,
                "y": result["annual_return"] * 100,
                "name": name,
                "sharpe": result["sharpe_ratio"]
            })

        fig_scatter = go.Figure()

        for item in scatter_data:
            fig_scatter.add_trace(go.Scatter(
                x=[item["x"]],
                y=[item["y"]],
                mode='markers+text',
                name=item["name"],
                text=[item["name"]],
                textposition='top center',
                marker=dict(
                    size=20,
                    color=item["sharpe"],
                    colorscale='RdYlGn',
                    colorbar=dict(title="夏普比率"),
                    showscale=True
                ),
            ))

        fig_scatter.update_layout(
            xaxis_title="年化波动率(%)",
            yaxis_title="年化收益率(%)",
            height=500,
            hovermode='closest'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 策略对比失败: {str(e)}")
        import traceback
        st.error(traceback.format_exc())


def portfolio_analysis_main():
    """组合分析主页面"""
    st.title("📊 组合分析")
    st.info("💡 多因子组合分析与投资组合构建")

    # 侧边栏配置
    with st.sidebar:
        st.subheader("组合配置")

        # 选择因子
        all_factors = factor_service.get_all_factors()
        factor_options = {f["name"]: f["description"] for f in all_factors}
        selected_factors = st.multiselect(
            "选择因子（多因子组合）",
            options=list(factor_options.keys()),
            format_func=lambda x: f"{x} - {factor_options.get(x, '')}",
            default=[],
            key="portfolio_factors"
        )

        # 股票池
        stock_codes_input = st.text_area(
            "股票池（每行一个）",
            value="000001\n600000\n000002",
            height=100,
            key="portfolio_stocks"
        )
        stock_codes = [code.strip() for code in stock_codes_input.strip().split("\n") if code.strip()]

        # 时间范围
        default_end = datetime.now()
        default_start = default_end - timedelta(days=365)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("开始日期", value=default_start, key="portfolio_start")
        with col2:
            end_date = st.date_input("结束日期", value=default_end, key="portfolio_end")

        # 组合参数
        st.markdown("### 组合参数")
        rebalance_freq = st.selectbox(
            "调仓频率",
            ["日频", "周频", "月频"],
            index=2,
            key="portfolio_rebalance"
        )

        top_pct = st.slider("选择前N%股票", 5, 50, 20, key="portfolio_top_pct")
        weight_method = st.selectbox(
            "权重分配",
            ["等权重", "IC加权", "风险平价", "最大夏普", "最小方差"],
            index=0,
            key="portfolio_weight",
            help="""权重优化方法：
- 等权重：所有因子权重相同
- IC加权：基于因子历史IC/IR加权
- 风险平价：风险均衡分配
- 最大夏普：最大化夏普比率
- 最小方差：最小化组合波动率"""
        )

        # 运行分析按钮
        run_analysis = st.button("🚀 运行组合分析", type="primary", use_container_width=True, key="run_portfolio")

    # 主内容区
    if run_analysis:
        if not selected_factors:
            st.error("❌ 请至少选择一个因子")
        elif not stock_codes:
            st.error("❌ 请输入股票代码")
        else:
            with st.spinner("正在执行组合分析，请稍候..."):
                try:
                    from backend.services.portfolio_analysis_service import portfolio_analysis_service

                    # 获取数据
                    all_data = {}
                    factor_returns_list = []  # 用于权重优化的因子收益率

                    for stock_code in stock_codes:
                        data = data_service.get_stock_data(
                            stock_code,
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d")
                        )
                        if not data.empty:
                            all_data[stock_code] = data
                            # 计算收益率用于权重优化
                            if "close" in data.columns:
                                returns = data["close"].pct_change().dropna()
                                factor_returns_list.append(returns)

                    if not all_data:
                        st.error("❌ 未获取到任何数据")
                        return

                    # 合并数据
                    merged_data = pd.concat(all_data.values(), names=["stock_code"]).reset_index()
                    merged_data = merged_data.sort_values(["date", "stock_code"])

                    # 计算每个因子的值和收益率
                    factor_scores = pd.DataFrame()
                    factor_scores["date"] = merged_data["date"].unique()
                    factor_scores.set_index("date", inplace=True)

                    # 用于权重优化的因子收益率DataFrame
                    factor_returns_df = pd.DataFrame()

                    for factor_name in selected_factors:
                        try:
                            factor_result = factor_service.calculate_factor(
                                factor_name=factor_name,
                                data=merged_data
                            )

                            if factor_result and "factor_values" in factor_result:
                                factor_df = factor_result["factor_values"]
                                # 按日期聚合因子得分
                                daily_scores = factor_df.groupby("date")["factor_value"].mean()
                                factor_scores[factor_name] = daily_scores

                                # 计算因子收益率（用于权重优化）
                                factor_return = daily_scores.pct_change().dropna()
                                if len(factor_return) > 0:
                                    factor_returns_df[factor_name] = factor_return

                        except Exception as e:
                            st.warning(f"⚠️ 因子 {factor_name} 计算失败: {str(e)}")

                    # 权重优化
                    factor_scores = factor_scores.dropna()
                    optimization_result = None
                    weights_dict = None

                    if len(factor_scores) > 0 and len(factor_returns_df) > 1:
                        # 映射前端选项到后端方法名
                        method_map = {
                            "等权重": "equal_weight",
                            "IC加权": "ic_weight",
                            "风险平价": "risk_parity",
                            "最大夏普": "max_sharpe",
                            "最小方差": "min_variance"
                        }
                        backend_method = method_map.get(weight_method, "equal_weight")

                        # 执行权重优化
                        optimization_result = portfolio_analysis_service.optimize_weights(
                            factor_returns=factor_returns_df,
                            method=backend_method
                        )

                        if "error" not in optimization_result:
                            weights_dict = optimization_result["weights"]
                            st.success(f"✅ 权重优化完成（{weight_method}）")

                    # 如果优化失败，使用等权重
                    if weights_dict is None:
                        weights_dict = {factor: 1.0 / len(selected_factors) for factor in selected_factors}
                        st.warning("⚠️ 使用等权重（权重优化不可用）")

                    # 计算综合得分
                    if len(factor_scores) > 0:
                        # 标准化每个因子
                        factor_scores_normalized = (factor_scores - factor_scores.mean()) / (factor_scores.std() + 1e-8)

                        # 使用优化后的权重计算综合得分
                        composite_score = portfolio_analysis_service.calculate_combined_factor_score(
                            factor_data={name: factor_scores[name] for name in selected_factors},
                            weights=weights_dict,
                            normalize=True
                        )

                        st.success("✅ 组合分析完成！")

                        # 显示权重分配
                        st.markdown("---")
                        st.subheader("⚖️ 权重分配")

                        weight_df = pd.DataFrame({
                            "因子": list(weights_dict.keys()),
                            "权重": list(weights_dict.values())
                        }).sort_values("权重", ascending=False)

                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.dataframe(
                                weight_df,
                                column_config={
                                    "因子": st.column_config.TextColumn("因子"),
                                    "权重": st.column_config.NumberColumn("权重", format="%.2%")
                                },
                                hide_index=True,
                                use_container_width=True
                            )

                        with col2:
                            fig_weight = go.Figure(data=[
                                go.Pie(
                                    labels=weight_df["因子"],
                                    values=weight_df["权重"],
                                    textinfo='percent+label'
                                )
                            ])
                            fig_weight.update_layout(
                                title="因子权重分配",
                                height=300
                            )
                            st.plotly_chart(fig_weight, use_container_width=True)

                        # 显示综合得分走势
                        st.markdown("---")
                        st.subheader("📈 综合得分走势")

                        fig_score = go.Figure()
                        fig_score.add_trace(go.Scatter(
                            x=composite_score.index,
                            y=composite_score.values,
                            mode='lines',
                            name='综合得分',
                            line=dict(color='blue', width=2)
                        ))
                        fig_score.update_layout(
                            title=f"多因子综合得分走势（{weight_method}）",
                            xaxis_title="日期",
                            yaxis_title="得分",
                            height=400
                        )
                        st.plotly_chart(fig_score, use_container_width=True)

                        # 因子贡献度分析
                        st.markdown("---")
                        st.subheader("📊 因子贡献度")

                        # 计算每个因子的加权贡献
                        weighted_contribution = pd.Series({
                            factor: factor_scores_normalized[factor].mean() * weight
                            for factor, weight in weights_dict.items()
                        })
                        contribution_df = pd.DataFrame({
                            "因子": weighted_contribution.index,
                            "贡献度": weighted_contribution.values
                        }).sort_values("贡献度", ascending=False)

                        fig_contrib = go.Figure(data=[
                            go.Bar(
                                x=contribution_df["因子"],
                                y=contribution_df["贡献度"],
                                marker_color=['green' if v > 0 else 'red' for v in contribution_df["贡献度"]]
                            )
                        ])
                        fig_contrib.update_layout(
                            title="因子贡献度（加权）",
                            xaxis_title="因子",
                            yaxis_title="加权平均贡献",
                            height=400
                        )
                        st.plotly_chart(fig_contrib, use_container_width=True)

                        # 因子相关性热力图
                        st.markdown("---")
                        st.subheader("🔥 因子相关性")

                        correlation_matrix = factor_scores_normalized.corr()

                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=correlation_matrix.values,
                            x=correlation_matrix.columns,
                            y=correlation_matrix.index,
                            colorscale='RdBu',
                            zmid=0
                        ))
                        fig_heatmap.update_layout(
                            title="因子相关性矩阵",
                            height=500
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)

                        # 权重方法对比（如果有优化结果）
                        if optimization_result is not None:
                            st.markdown("---")
                            st.subheader("📊 权重方法对比")

                            # 比较所有方法
                            comparison_df = portfolio_analysis_service.compare_weight_methods(
                                factor_returns=factor_returns_df,
                                methods=["equal_weight", "ic_weight", "risk_parity", "max_sharpe", "min_variance"]
                            )

                            if not comparison_df.empty:
                                st.dataframe(
                                    comparison_df,
                                    column_config={
                                        "方法": st.column_config.TextColumn("方法"),
                                        "预期收益": st.column_config.NumberColumn("预期收益", format="%.6f"),
                                        "预期波动率": st.column_config.NumberColumn("预期波动率", format="%.6f"),
                                        "收益波动比": st.column_config.NumberColumn("收益波动比", format="%.4f"),
                                    },
                                    use_container_width=True
                                )

                        # 统计摘要
                        st.markdown("---")
                        st.subheader("📋 统计摘要")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("因子数量", len(selected_factors))
                        with col2:
                            st.metric("数据天数", len(composite_score))
                        with col3:
                            st.metric("平均得分", f"{composite_score.mean():.4f}")
                        with col4:
                            st.metric("得分标准差", f"{composite_score.std():.4f}")

                        # 优化结果摘要（如果有）
                        if optimization_result is not None and "expected_return" in optimization_result:
                            st.markdown("---")
                            st.subheader("🎯 优化指标")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("预期收益", f"{optimization_result['expected_return']:.6f}")
                            with col2:
                                st.metric("预期波动率", f"{optimization_result['expected_volatility']:.6f}")
                            with col3:
                                if optimization_result['expected_volatility'] > 0:
                                    sharpe = optimization_result['expected_return'] / optimization_result['expected_volatility']
                                    st.metric("预期夏普比率", f"{sharpe:.4f}")

                    else:
                        st.warning("⚠️ 没有有效的因子数据")

                except Exception as e:
                    st.error(f"❌ 分析失败: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    else:
        st.info("👈 配置完成后点击「运行组合分析」按钮")


if __name__ == "__main__":
    main()
