"""数据引擎模块：基于本地 Parquet 和因子表实现动态前复权（QFQ）高速行情检索。"""

import configparser
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import duckdb
import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class DataEngine:
    """行情数据引擎：负责从本地 Parquet 和因子表读取并动态计算前复权 OHLCV。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.parquet_dir: Path = Path(settings.parquet_dir)
        self.factors_path: Path = Path(settings.factors_path)
        self.start_date: str = settings.start_date

        self._factors_df: Optional[pd.DataFrame] = None
        self._symbols_cache: Optional[list[str]] = None
        self._all_ohlcv_cache: Optional[dict[str, pd.DataFrame]] = None

    def _get_factors_df(self) -> pd.DataFrame:
        """加载因子表（全市场因子约 1.3MB，缓存到内存）。"""
        if self._factors_df is not None:
            return self._factors_df

        if self.factors_path.exists():
            df = pd.read_parquet(self.factors_path)
        else:
            df = self._load_factors_from_pg()

        if not df.empty:
            df["code"] = df["code"].astype(str).str.zfill(6)
            df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[us]")
            df = df.sort_values(["code", "date"]).reset_index(drop=True)

        self._factors_df = df
        return self._factors_df

    def _load_factors_from_pg(self) -> pd.DataFrame:
        """从 Postgres 数据库加载因子表。"""
        engine = None
        try:
            from sqlalchemy import create_engine, text

            db_ini = Path(self.settings.db_config_path)
            if not db_ini.exists():
                logger.warning(f"Database config not found: {db_ini}")
                return pd.DataFrame()

            parser = configparser.ConfigParser()
            parser.read(db_ini)
            section = self.settings.pg_section
            if not parser.has_section(section):
                logger.warning(f"Section [{section}] not found in {db_ini}")
                return pd.DataFrame()

            user = parser.get(section, "user").strip()
            password = parser.get(section, "password", fallback="").strip()
            host = parser.get(section, "host").strip()
            port = parser.get(section, "port", fallback="5432").strip()
            database = parser.get(section, "database", fallback=parser.get(section, "dbname", fallback="")).strip()

            auth = quote_plus(user)
            if password:
                auth = f"{auth}:{quote_plus(password)}"
            url = f"postgresql://{auth}@{host}:{port}/{database}"

            engine = create_engine(url)
            sql = text("SELECT code, date, qfq_factor FROM public.stock_adjustment_factors ORDER BY code, date ASC")
            with engine.connect() as conn:
                df = pd.read_sql(sql, conn)
            return df
        except Exception as e:
            logger.error(f"从 Postgres 读取因子失败: {e}")
            return pd.DataFrame()
        finally:
            if engine is not None:
                engine.dispose()

    def get_all_symbols(self) -> list[str]:
        """获取全市场所有股票代码列表。"""
        if self._symbols_cache is not None:
            return self._symbols_cache

        parquet_files = sorted(self.parquet_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning(f"未在 {self.parquet_dir} 找到 parquet 数据文件")
            return []

        # 优先读取最新年份的 parquet 提取股票代码
        latest_file = str(parquet_files[-1])
        try:
            symbols = duckdb.query(
                f"SELECT DISTINCT code FROM '{latest_file}' ORDER BY code"
            ).fetchall()
            self._symbols_cache = [str(r[0]).zfill(6) for r in symbols]
        except Exception as e:
            logger.error(f"提取股票列表失败: {e}")
            self._symbols_cache = []

        return self._symbols_cache

    def get_local_symbols(self) -> list[str]:
        """获取本地已有数据的股票代码列表（与 get_all_symbols 等价）。"""
        return self.get_all_symbols()

    def _preload_ohlcv_if_needed(self) -> None:
        """一次性加载全市场数据并完成全局向量化前复权（QFQ），毫秒级供各策略查询。"""
        if self._all_ohlcv_cache is not None:
            return

        parquet_pattern = str(self.parquet_dir / "*.parquet")
        try:
            sql = f"""
                SELECT
                    code as symbol,
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    amount as turnover
                FROM '{parquet_pattern}'
                WHERE date >= ?
                ORDER BY date ASC
            """
            df_raw = duckdb.execute(sql, [self.start_date]).df()
            if df_raw.empty:
                self._all_ohlcv_cache = {}
                return

            df_raw["symbol"] = df_raw["symbol"].astype(str).str.zfill(6)
            df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.normalize().astype("datetime64[us]")
            df_raw = df_raw.sort_values("date").reset_index(drop=True)

            # 获取因子表
            df_factors = self._get_factors_df()
            if not df_factors.empty and "qfq_factor" in df_factors.columns:
                df_factors_sorted = df_factors.sort_values("date").reset_index(drop=True)
                df_merged = pd.merge_asof(
                    df_raw,
                    df_factors_sorted[["code", "date", "qfq_factor"]],
                    on="date",
                    left_by="symbol",
                    right_by="code",
                    direction="backward",
                )
                df_merged["qfq_factor"] = df_merged.groupby("symbol")["qfq_factor"].ffill().bfill().fillna(1.0)
                f = df_merged["qfq_factor"].replace(0, 1.0)
                for col in ["open", "high", "low", "close"]:
                    df_merged[col] = (df_merged[col] / f).round(3)
                df_merged["volume"] = (df_merged["volume"] * f).round(0)
                df_all = df_merged
            else:
                df_all = df_raw

            df_all["date"] = df_all["date"].dt.strftime("%Y-%m-%d")
            self._all_ohlcv_cache = {
                symbol: grp.sort_values("date").reset_index(drop=True)
                for symbol, grp in df_all.groupby("symbol")
            }
            logger.info(f"全市场数据与前复权加载完成，共缓存 {len(self._all_ohlcv_cache)} 只股票")
        except Exception as e:
            logger.error(f"预加载全市场数据失败: {e}")
            self._all_ohlcv_cache = {}

    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        """获取单只股票的前复权（QFQ）历史日 K 线数据。

        Returns:
            DataFrame 包含列: ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        """
        symbol = str(symbol).zfill(6)
        self._preload_ohlcv_if_needed()

        if self._all_ohlcv_cache is not None and symbol in self._all_ohlcv_cache:
            return self._all_ohlcv_cache[symbol]

        parquet_pattern = str(self.parquet_dir / "*.parquet")
        try:
            sql = f"""
                SELECT
                    code as symbol,
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    amount as turnover
                FROM '{parquet_pattern}'
                WHERE code = ? AND date >= ?
                ORDER BY date ASC
            """
            df_raw = duckdb.execute(sql, [symbol, self.start_date]).df()
        except Exception as e:
            logger.error(f"[{symbol}] 读取 Parquet 失败: {e}")
            return pd.DataFrame()

        if df_raw.empty:
            return df_raw

        return self.apply_qfq(df_raw, symbol)

    def apply_qfq(self, df_raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """应用前复权计算。

        公式：
          前复权价格 = 原始价格 / qfq_factor
          成交量 = 原始成交量 * qfq_factor
        """
        if df_raw.empty:
            return df_raw

        df = df_raw.copy()
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)

        df_factors = self._get_factors_df()
        if df_factors.empty or "qfq_factor" not in df_factors.columns:
            return df

        stock_factors = df_factors[df_factors["code"] == symbol]
        if stock_factors.empty:
            return df

        df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[us]")
        df = df.sort_values("date").reset_index(drop=True)

        stock_factors = stock_factors.copy()
        stock_factors["date"] = pd.to_datetime(stock_factors["date"]).dt.normalize().astype("datetime64[us]")
        stock_factors = stock_factors.sort_values("date").reset_index(drop=True)

        df = pd.merge_asof(
            df,
            stock_factors[["date", "qfq_factor"]],
            on="date",
            direction="backward",
        )
        df["qfq_factor"] = df["qfq_factor"].ffill().bfill().fillna(1.0)

        f = df["qfq_factor"].replace(0, 1.0)
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = (df[col] / f).round(3)

        if "volume" in df.columns:
            df["volume"] = (df["volume"] * f).round(0)

        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df

    def sync_today_bulk(self) -> int:
        """盘后/日常数据同步状态检查。"""
        symbols = self.get_all_symbols()
        logger.info(f"Parquet 数据源已就绪，当前市场涵盖 {len(symbols)} 只股票")
        return len(symbols)

    def backfill(self, symbols: list[str]) -> None:
        """回填模式。"""
        logger.info(f"本地 Parquet 数据源就绪，共 {len(symbols)} 只标的，无需网络拉取。")

