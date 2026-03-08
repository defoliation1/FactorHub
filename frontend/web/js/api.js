/**
 * FactorFlow API 客户端
 * 封装所有后端API调用
 */

class FactorFlowAPI {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
    }

    /**
     * 通用请求方法
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        console.log(`[API] 请求: ${options.method || 'GET'} ${url}`);

        try {
            const response = await fetch(url, { ...defaultOptions, ...options });

            console.log(`[API] 响应状态: ${response.status} ${response.statusText}`);

            if (!response.ok) {
                const error = await response.json();
                console.error('[API] 错误响应:', error);
                throw new Error(error.detail || error.message || '请求失败');
            }

            const data = await response.json();
            console.log('[API] 成功响应:', data);
            return data;
        } catch (error) {
            console.error(`[API] 请求失败: ${endpoint}`, error);
            throw error;
        }
    }

    // ========== 因子管理 API ==========

    /**
     * 获取因子列表
     */
    async getFactors(params = {}) {
        let endpoint = '/api/factors';
        const query = new URLSearchParams(params).toString();
        if (query) endpoint += `?${query}`;
        return this.request(endpoint);
    }

    /**
     * 获取因子详情
     */
    async getFactor(id) {
        return this.request(`/api/factors/${id}`);
    }

    /**
     * 获取因子统计
     */
    async getFactorStats() {
        return this.request('/api/factors/stats');
    }

    /**
     * 创建因子
     */
    async createFactor(data) {
        return this.request('/api/factors', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    /**
     * 更新因子
     */
    async updateFactor(id, data) {
        return this.request(`/api/factors/${id}`, {
            method: 'PUT',
            body: JSON.stringify(data),
        });
    }

    /**
     * 删除因子
     */
    async deleteFactor(id) {
        return this.request(`/api/factors/${id}`, {
            method: 'DELETE',
        });
    }

    /**
     * 批量生成因子
     */
    async batchGenerateFactors(data) {
        return this.request('/api/factors/batch-generate', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    /**
     * 验证因子公式
     */
    async validateFactor(data) {
        return this.request('/api/factors/validate', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    // ========== 数据管理 API ==========

    /**
     * 获取股票数据
     */
    async getStockData(code, startDate, endDate) {
        return this.request(
            `/api/data/stock/${code}?start_date=${startDate}&end_date=${endDate}`
        );
    }

    /**
     * 获取缓存统计
     */
    async getCacheStats() {
        return this.request('/api/data/cache/stats');
    }

    /**
     * 清理缓存
     */
    async cleanupCache() {
        return this.request('/api/data/cache/cleanup', {
            method: 'POST',
        });
    }

    /**
     * 清空缓存
     */
    async clearCache() {
        return this.request('/api/data/cache/clear', {
            method: 'POST',
        });
    }

    // ========== 因子分析 API ==========

    /**
     * 计算因子值
     */
    async calculateFactor(data) {
        return this.request('/api/analysis/calculate', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    /**
     * 计算IC/IR
     */
    async calculateIC(data) {
        return this.request('/api/analysis/ic', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    /**
     * 稳定性检验
     */
    async stabilityTest(data) {
        return this.request('/api/analysis/stability', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    /**
     * 多周期分析
     */
    async multiPeriodAnalysis(data) {
        return this.request('/api/analysis/multi-period', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    /**
     * 因子衰减分析
     */
    async calculateDecay(data) {
        return this.request('/api/analysis/decay', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    /**
     * 因子衰减分析
     */
    async decayAnalysis(data) {
        return this.request('/api/analysis/decay', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    // ========== 因子挖掘 API ==========

    /**
     * 启动遗传算法挖掘
     */
    async startGeneticMining(data) {
        return this.request('/api/mining/genetic', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    /**
     * 获取挖掘状态
     */
    async getMiningStatus(taskId) {
        return this.request(`/api/mining/status/${taskId}`);
    }

    /**
     * 获取挖掘结果
     */
    async getMiningResults(taskId) {
        return this.request(`/api/mining/results/${taskId}`);
    }

    // ========== 组合分析 API ==========

    /**
     * 优化权重
     */
    async optimizeWeights(data) {
        return this.request('/api/portfolio/optimize-weights', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    /**
     * 计算综合得分
     */
    async calculateCompositeScore(data) {
        return this.request('/api/portfolio/composite-score', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    /**
     * 对比权重方法
     */
    async compareWeightMethods(data) {
        return this.request('/api/portfolio/compare-methods', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    // ========== 策略回测 API ==========

    /**
     * 运行单策略回测
     */
    async runSingleBacktest(data) {
        return this.request('/api/backtest/single', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    /**
     * 运行策略对比
     */
    async runStrategyComparison(data) {
        return this.request('/api/backtest/comparison', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    /**
     * 获取回测历史
     */
    async getBacktestHistory(limit = 10) {
        return this.request(`/api/backtest/history?limit=${limit}`);
    }

    /**
     * 删除回测历史
     */
    async deleteBacktestHistory(id) {
        return this.request(`/api/backtest/history/${id}`, {
            method: 'DELETE',
        });
    }
}

// 创建全局API实例
const api = new FactorFlowAPI();
