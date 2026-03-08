/**
 * FactorFlow 通用工具函数
 */

// ========== UI 工具函数 ==========

/**
 * 显示Toast通知
 */
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="flex items-center gap-2">
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" class="ml-4 text-gray-400 hover:text-gray-600">&times;</button>
        </div>
    `;
    document.body.appendChild(toast);

    // 3秒后自动移除
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

/**
 * 显示加载遮罩
 */
function showLoading(message = '加载中...') {
    const overlay = document.createElement('div');
    overlay.id = 'loading-overlay';
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="flex flex-col items-center gap-4">
            <div class="loading-spinner"></div>
            <p class="text-slate-600">${message}</p>
        </div>
    `;
    document.body.appendChild(overlay);
}

/**
 * 隐藏加载遮罩
 */
function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

/**
 * 格式化日期
 */
function formatDate(date, format = 'YYYY-MM-DD') {
    const d = new Date(date);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');

    return format
        .replace('YYYY', year)
        .replace('MM', month)
        .replace('DD', day);
}

/**
 * 格式化数字（百分比）
 */
function formatPercent(value, decimals = 2) {
    if (value === null || value === undefined) return '-';
    return (value * 100).toFixed(decimals) + '%';
}

/**
 * 格式化数字（千分位）
 */
function formatNumber(value, decimals = 2) {
    if (value === null || value === undefined) return '-';
    return Number(value).toLocaleString('zh-CN', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    });
}

/**
 * 获取数值颜色类
 */
function getValueColor(value) {
    if (value > 0) return 'text-accent-red';
    if (value < 0) return 'text-accent-green';
    return 'text-slate-600';
}

/**
 * 获取分位数颜色
 */
function getPercentileColor(percentile) {
    if (percentile >= 75) return '#ff4d4f';
    if (percentile <= 25) return '#1890ff';
    return '#73d13d';
}

// ========== 数据处理 ==========

/**
 * 将DataFrame-like数据转换为数组
 */
function dataframeToArray(data) {
    if (!data) return [];
    const { index, columns, data: values } = data;
    return values.map((row, i) => {
        const obj = { date: index[i] };
        columns.forEach((col, j) => {
            obj[col] = row[j];
        });
        return obj;
    });
}

/**
 * 深拷贝对象
 */
function deepClone(obj) {
    return JSON.parse(JSON.stringify(obj));
}

/**
 * 防抖函数
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * 节流函数
 */
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// ========== DOM 操作 ==========

/**
 * 查询元素（安全版本）
 */
function $(selector) {
    return document.querySelector(selector);
}

/**
 * 查询所有元素
 */
function $$(selector) {
    return document.querySelectorAll(selector);
}

/**
 * 创建元素
 */
function createElement(tag, options = {}) {
    const el = document.createElement(tag);

    if (options.className) el.className = options.className;
    if (options.id) el.id = options.id;
    if (options.innerHTML) el.innerHTML = options.innerHTML;
    if (options.text) el.textContent = options.text;
    if (options.attributes) {
        Object.entries(options.attributes).forEach(([key, value]) => {
            el.setAttribute(key, value);
        });
    }

    return el;
}

// ========== 表单处理 ==========

/**
 * 收集表单数据
 */
function collectFormData(form) {
    const formData = new FormData(form);
    const data = {};
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    return data;
}

/**
 * 填充表单数据
 */
function fillFormData(form, data) {
    Object.entries(data).forEach(([key, value]) => {
        const input = form.querySelector(`[name="${key}"]`);
        if (input) {
            input.value = value;
        }
    });
}

// ========== 本地存储 ==========

/**
 * 保存到本地存储
 */
function saveToLocalStorage(key, value) {
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
        console.error('保存到本地存储失败:', e);
    }
}

/**
 * 从本地存储读取
 */
function loadFromLocalStorage(key) {
    try {
        const value = localStorage.getItem(key);
        return value ? JSON.parse(value) : null;
    } catch (e) {
        console.error('从本地存储读取失败:', e);
        return null;
    }
}

/**
 * 从本地存储删除
 */
function removeFromLocalStorage(key) {
    try {
        localStorage.removeItem(key);
    } catch (e) {
        console.error('从本地存储删除失败:', e);
    }
}

// ========== URL 工具 ==========

/**
 * 获取URL参数
 */
function getUrlParams() {
    const params = new URLSearchParams(window.location.search);
    const result = {};
    for (let [key, value] of params.entries()) {
        result[key] = value;
    }
    return result;
}

/**
 * 设置URL参数
 */
function setUrlParams(params) {
    const url = new URL(window.location);
    Object.entries(params).forEach(([key, value]) => {
        if (value === null || value === undefined) {
            url.searchParams.delete(key);
        } else {
            url.searchParams.set(key, value);
        }
    });
    window.history.pushState({}, '', url);
}

// ========== 错误处理 ==========

/**
 * 全局错误处理
 */
window.addEventListener('error', (event) => {
    console.error('全局错误:', event.error);
    showToast('发生错误，请刷新页面重试', 'error');
});

/**
 * 未捕获的Promise错误
 */
window.addEventListener('unhandledrejection', (event) => {
    console.error('未处理的Promise错误:', event.reason);
    showToast('请求失败，请稍后重试', 'error');
});
