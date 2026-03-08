/**
 * FactorFlow 图表统一配置
 * 基于 web_demo.md 规范
 */

/**
 * Chart.js 通用配置（符合 web_demo.md 规范）
 */
const COMMON_CHART_OPTIONS = {
    // 响应式配置
    responsive: true,
    maintainAspectRatio: false,

    // 交互模式
    interaction: {
        mode: 'index',
        intersect: false,
    },

    // 插件配置
    plugins: {
        legend: {
            position: 'top',
            labels: {
                font: {
                    family: "'Noto Sans SC', sans-serif",
                    size: 12
                },
                padding: 15,
                usePointStyle: true
            }
        },
        tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            padding: 12,
            titleFont: {
                family: "'Noto Sans SC', sans-serif",
                size: 13
            },
            bodyFont: {
                family: "'JetBrains Mono', monospace",
                size: 12
            },
            cornerRadius: 6,
            displayColors: true
        }
    },

    // 坐标轴配置
    scales: {
        x: {
            ticks: {
                maxTicksLimit: 12,
                font: {
                    size: 11,
                    family: "'Noto Sans SC', sans-serif"
                },
                color: '#64748b'
            },
            grid: {
                color: 'rgba(0, 0, 0, 0.05)',
                drawBorder: false
            }
        },
        y: {
            ticks: {
                font: {
                    size: 11,
                    family: "'JetBrains Mono', monospace"
                },
                color: '#64748b',
                padding: 8
            },
            grid: {
                color: 'rgba(0, 0, 0, 0.05)',
                drawBorder: false
            }
        }
    },

    // 元素样式配置
    elements: {
        line: {
            tension: 0.3,        // 平滑曲线
            spanGaps: false,     // 不连接空值
            borderWidth: 2
        },
        point: {
            radius: 0,          // 隐藏数据点
            hitRadius: 10,      // 扩大点击区域
            hoverRadius: 6      // 悬停时显示点
        }
    }
};

/**
 * 折线图专用配置
 */
function getLineChartOptions(customOptions = {}) {
    return mergeDeep(COMMON_CHART_OPTIONS, {
        ...customOptions
    });
}

/**
 * 柱状图专用配置
 */
function getBarChartOptions(customOptions = {}) {
    return mergeDeep(COMMON_CHART_OPTIONS, {
        elements: {
            bar: {
                borderRadius: 4,
                borderWidth: 1
            }
        },
        ...customOptions
    });
}

/**
 * 不显示图例的配置
 */
function getNoLegendChartOptions(customOptions = {}) {
    return mergeDeep(COMMON_CHART_OPTIONS, {
        plugins: {
            ...COMMON_CHART_OPTIONS.plugins,
            legend: {
                display: false
            }
        },
        ...customOptions
    });
}

/**
 * 工具函数：深度合并对象
 */
function mergeDeep(target, ...sources) {
    const isObject = (obj) => obj && typeof obj === 'object';

    if (!sources.length) return target;
    const source = sources.shift();

    if (isObject(target) && isObject(source)) {
        for (const key in source) {
            if (isObject(source[key])) {
                if (!target[key]) Object.assign(target, { [key]: {} });
                mergeDeep(target[key], source[key]);
            } else {
                Object.assign(target, { [key]: source[key] });
            }
        }
    }

    return mergeDeep(target, ...sources);
}

/**
 * 颜色辅助函数（符合中国股市习惯）
 */
const ChartColors = {
    // 主色红色（上涨/正数）
    primary: '#ef4444',
    primaryBg: 'rgba(239, 68, 68, 0.1)',

    // 绿色（下跌/负数）
    positive: '#22c55e',
    positiveBg: 'rgba(34, 197, 94, 0.1)',

    // 辅助色
    blue: '#3b82f6',
    blueBg: 'rgba(59, 130, 246, 0.1)',
    yellow: '#eab308',
    yellowBg: 'rgba(234, 179, 8, 0.1)',
    purple: '#8b5cf6',
    purpleBg: 'rgba(139, 92, 246, 0.1)',

    // 根据数值获取颜色（正数红色，负数绿色）
    getValueColor(value) {
        return value > 0 ? this.primary : this.positive;
    },

    getValueBgColor(value, alpha = 0.1) {
        const color = value > 0 ? this.primary : this.positive;
        return `rgba(${hexToRgb(color)}, ${alpha})`;
    },

    // 生成多数据集颜色序列
    getDatasetColors(count) {
        const colors = [this.primary, this.blue, this.positive, this.yellow, this.purple];
        return Array.from({ length: count }, (_, i) => colors[i % colors.length]);
    },

    getDatasetBgColors(count, alpha = 0.1) {
        const hexColors = this.getDatasetColors(count);
        return hexColors.map(color => `rgba(${hexToRgb(color)}, ${alpha})`);
    }
};

/**
 * 工具函数：十六进制颜色转RGB
 */
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
        ? `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}`
        : '239, 68, 68';
}

/**
 * 图表容器配置
 */
const ChartContainer = {
    // 标准图表容器
    STANDARD_HEIGHT: '300px',

    // 小图表容器
    SMALL_HEIGHT: '150px',

    // 大图表容器
    LARGE_HEIGHT: '400px',

    /**
     * 获取图表容器样式
     */
    getContainerClass(size = 'standard') {
        const heightMap = {
            standard: '300px',
            small: '150px',
            large: '400px'
        };
        return {
            width: '100%',
            height: heightMap[size] || heightMap.standard,
            position: 'relative'
        };
    }
};
