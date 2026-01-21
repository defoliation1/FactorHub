"""
应用配置模块
"""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # 项目路径
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CACHE_DIR: Path = DATA_DIR / "cache"
    DB_DIR: Path = DATA_DIR / "db"
    CONFIG_DIR: Path = BASE_DIR / "config"
    REPORTS_DIR: Path = DATA_DIR / "reports"

    # 数据库配置
    DATABASE_URL: str = f"sqlite:///{DB_DIR}/factorflow.db"

    # akshare 配置
    AKSHARE_CACHE_ENABLED: bool = True
    AKSHARE_CACHE_DIR: Path = CACHE_DIR / "akshare"

    # 分析配置
    DEFAULT_START_DATE: str = "2020-01-01"
    DEFAULT_END_DATE: str = "2024-12-31"

    # 滚动窗口配置
    ROLLING_WINDOW: int = 252  # 一年交易日

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 创建必要的目录
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.DB_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.AKSHARE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# 全局配置实例
settings = Settings()
