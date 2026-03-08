# FactorFlow 前端系统

基于原生 HTML/CSS/JavaScript 的股票因子分析系统前端。

## 📁 项目结构

```
frontend/web/
├── index.html                    # 首页/仪表盘
├── factor-management.html        # 因子管理页面
├── factor-analysis.html          # 因子分析页面
├── factor-mining.html            # 因子挖掘页面
├── portfolio-analysis.html       # 组合分析页面
├── backtesting.html              # 策略回测页面
├── css/
│   └── common.css               # 通用样式
├── js/
│   ├── api.js                   # API客户端封装
│   ├── common.js                # 通用工具函数
│   └── charts.js                # 图表工具（待实现）
└── assets/                       # 静态资源
    └── images/
```

## 🚀 快速开始

### 1. 启动后端服务

首先启动 FastAPI 后端服务：

```bash
cd h:/pythonwork/FactorFlow
python -m backend.api.main
```

后端服务将在 http://localhost:8000 启动

- API文档：http://localhost:8000/docs
- Swagger UI：http://localhost:8000/docs
- ReDoc：http://localhost:8000/redoc

### 2. 启动前端

有两种方式：

**方式1：直接打开**
```bash
# 在浏览器中打开
start frontend/web/index.html
```

**方式2：使用本地服务器（推荐）**
```bash
cd frontend/web
python -m http.server 8080
```

然后访问 http://localhost:8080

## 📋 功能页面

### 1. 首页 (index.html)
- 系统概览
- 快速导航
- 实时统计（因子数量、缓存状态）

### 2. 因子管理 (factor-management.html)
- **因子列表**：查看所有因子，支持筛选和搜索
- **新增因子**：创建自定义因子，支持公式验证
- **批量生成**：基于基础因子批量生成新因子

### 3. 因子分析 (factor-analysis.html)
- **概览**：IC均值、标准差、IR比率等关键指标
- **IC分析**：IC序列和分布图
- **稳定性检验**：统计显著性测试
- **多周期分析**：不同周期的IC对比
- **增强分析**：中性化、去极值等预处理效果

### 4. 因子挖掘 (factor-mining.html)
- 遗传算法自动挖掘因子表达式
- 实时显示进化进度和适应度曲线
- 自动保存发现的优质因子

### 5. 组合分析 (portfolio-analysis.html)
- **权重优化**：等权重、IC加权、最大夏普等多种方法
- **综合得分**：计算多因子综合得分
- **方法对比**：对比不同权重方法的效果

### 6. 策略回测 (backtesting.html)
- **单策略回测**：完整的回测流程和绩效分析
- **多策略对比**：对比多个因子的表现

## 🎨 技术栈

- **HTML5**：语义化标签
- **CSS3**：原生CSS + Tailwind CSS (CDN)
- **JavaScript (ES6+)**：原生JavaScript，无框架
- **Chart.js 4.4.0**：图表库
- **Tailwind CSS**：实用工具类CSS框架

## 🎯 设计规范

基于 `web_demo.md` 规范：

- **主题色**：红色系 (#ef4444)
- **响应式设计**：支持桌面、平板、手机
- **交互风格**：现代化、简洁、直观
- **无构建工具**：可直接在浏览器运行

## 📦 API接口

所有API请求通过 `js/api.js` 封装：

```javascript
// 因子管理
api.getFactors(category, source)
api.createFactor(data)
api.deleteFactor(id)

// 因子分析
api.calculateFactor(data)
api.calculateIC(data)
api.calculateStability(data)

// 因子挖掘
api.runGeneticMining(data)
api.getMiningStatus(taskId)

// 组合分析
api.optimizeWeights(data)

// 策略回测
api.runSingleBacktest(data)
api.runStrategyComparison(data)

// 数据管理
api.getCacheStats()
```

## 🔧 配置

### API地址配置

在 `js/api.js` 中修改：

```javascript
const API_BASE_URL = 'http://localhost:8000';
```

### 主题色配置

在 `css/common.css` 中修改CSS变量：

```css
:root {
    --primary-500: #ef4444;
    --primary-600: #dc2626;
    /* ... */
}
```

## 📝 开发指南

### 添加新页面

1. 创建HTML文件，参考现有页面结构
2. 引入通用CSS和JS：
```html
<link rel="stylesheet" href="css/common.css">
<script src="js/api.js"></script>
<script src="js/common.js"></script>
```
3. 实现页面逻辑

### 调用后端API

```javascript
async function loadData() {
    try {
        const result = await api.getFactors();
        if (result.success) {
            // 处理数据
            displayData(result.data);
        }
    } catch (error) {
        console.error('加载失败:', error);
        showToast('加载失败', 'error');
    }
}
```

### 显示通知

```javascript
showToast('操作成功', 'success');  // success, error, warning, info
```

### 显示加载状态

```javascript
showLoading('处理中...');
// ... 执行操作
hideLoading();
```

## 🐛 已知问题

- 部分高级分析功能还在开发中
- 图表导出功能待实现
- 部分API接口需要完善

## 🔄 与Streamlit对比

| 功能 | Streamlit | 新前端 |
|------|-----------|--------|
| UI/UX | 组件化，但有限制 | 完全自定义 |
| 性能 | 每次交互重新渲染 | 前端渲染，更流畅 |
| 部署 | 需要Python环境 | 静态文件，简单部署 |
| 交互 | 有延迟 | 即时响应 |
| 定制 | 受限于Streamlit | 完全自由 |

## 📈 未来计划

- [ ] 实现实时数据推送（WebSocket）
- [ ] 添加更多图表类型
- [ ] 支持策略保存和历史记录
- [ ] 优化移动端体验
- [ ] 添加暗色主题
- [ ] 实现离线功能（PWA）

## 📄 许可

MIT License
