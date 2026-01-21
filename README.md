# FactorFlow - 股票因子分析系统

基于 Streamlit 的量化因子分析平台，提供因子管理、计算、IC/IR分析和SHAP特征解释功能。

## 功能特点

### 因子管理
- 预置30+常用技术因子（价格、动量、波动率、成交量等）
- 支持用户自定义因子
- 因子代码验证和测试

### 因子分析
- 支持单股票和股票池分析
- 自动滚动窗口标准化
- IC/IR 统计分析
- SHAP 特征重要性分析
- 可视化图表展示
- Markdown 报告导出

## 技术栈

- **前端**: Streamlit
- **数据源**: akshare（中国A股数据）
- **数据库**: SQLite
- **计算**: numpy, pandas, TA-Lib
- **分析**: xgboost, scikit-learn, shap
- **图表**: plotly

## 安装

### 1. 克隆项目

```bash
git clone <repository-url>
cd FactorFlow
```

### 2. 使用 uv 安装依赖

```bash
# 安装 uv（如果还没安装）
pip install uv

# 创建虚拟环境并安装依赖
uv sync
```

### 3. 安装 TA-Lib

**Windows:**
```bash
# 从以下地址下载对应Python版本的whl文件
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl
```

**Linux:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
make install
cd ..
pip install TA-Lib
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

## 使用

### 启动应用

```bash
# 使用 uv 运行
uv run streamlit run frontend/app.py

# 或者激活虚拟环境后直接运行
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows
streamlit run frontend/app.py
```

### 因子管理

1. 查看"因子列表"了解所有可用因子
2. 在"新增因子"页面创建自定义因子
3. 编写因子代码并验证后保存

### 因子分析

1. 选择"因子分析"页面
2. 配置分析参数：
   - 选择数据模式（单股票/股票池）
   - 输入股票代码
   - 选择要分析的因子
   - 设置时间区间
3. 点击"开始分析"
4. 查看分析结果：
   - 行情预览：价格走势与因子叠加
   - SHAP分析：特征重要性
   - 统计分析：IC/IR指标、月度热力图、滚动IR
   - 导出报告：生成Markdown格式报告

## 因子语法

在编写自定义因子时，可使用以下变量和函数：

### 变量
- `open`, `high`, `low`, `close`, `volume`: OHLCV数据
- `np`: numpy模块

### TA-Lib函数
- `SMA`, `EMA`: 移动平均
- `RSI`: 相对强弱指标
- `MACD`: 平滑异同移动平均
- `ADX`: 平均趋向指数
- `CCI`: 顺势指标
- `ATR`: 平均真实波幅
- `BBANDS`: 布林带
- `OBV`: 能量潮

### 示例

```python
# 价格相对20日均线位置
close / SMA(close, timeperiod=20)

# 5日收益率
close / close.shift(5) - 1

# 10日波动率
np.log(close / close.shift(1)).rolling(window=10).std()

# RSI指标
RSI(close, timeperiod=14)

# 布林带位置
(close - BBANDS(close, timeperiod=20)[2]) / (BBANDS(close, timeperiod=20)[0] - BBANDS(close, timeperiod=20)[2])
```

## 项目结构

```
FactorFlow/
├── backend/
│   ├── core/          # 核心配置和数据库
│   ├── models/        # 数据模型
│   ├── repositories/  # 数据访问层
│   └── services/      # 业务逻辑层
├── frontend/
│   └── app.py         # Streamlit应用
├── config/            # 配置文件
├── data/              # 数据目录
│   ├── cache/         # 数据缓存
│   ├── db/            # SQLite数据库
│   └── reports/       # 导出的报告
├── tests/             # 测试文件
└── pyproject.toml     # 项目配置
```

## 数据缓存

- akshare数据会自动缓存到 `data/cache/akshare/`
- 分析结果会缓存到SQLite数据库
- 下次相同参数的分析会直接读取缓存

## 注意事项

1. **股票代码格式**：
   - 支持 `000001` 或 `000001.SZ`（深圳）
   - 支持 `600000` 或 `600000.SH`（上海）

2. **数据获取**：
   - 首次获取数据会从网络下载，之后会使用缓存
   - 建议时间范围不超过5年，避免数据量过大

3. **因子计算**：
   - 因子值会自动进行滚动窗口标准化（默认252天）
   - 会自动添加时间特征（星期、月份、季度）

## License

MIT License
