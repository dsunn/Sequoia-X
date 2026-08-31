"""数据引擎与复权属性测试。"""

import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
from hypothesis import given, settings as h_settings
from hypothesis import strategies as st

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine


def make_engine_in(tmp_dir: str) -> tuple[DataEngine, Settings]:
    """创建使用临时目录的 DataEngine 实例。"""
    settings = Settings(
        parquet_dir=tmp_dir,
        factors_path=str(Path(tmp_dir) / "factors.parquet"),
        start_date="2024-01-01",
        feishu_webhook_url="https://example.com/hook",
    )
    engine = DataEngine(settings)
    return engine, settings


# Property: 前复权计算价格 = raw / factor, 成交量 = volume * factor
@given(
    symbol=st.text(min_size=6, max_size=6, alphabet="0123456789"),
    trade_date=st.dates(min_value=date(2024, 1, 1), max_value=date(2025, 12, 31)),
    raw_price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
    qfq_factor=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
)
@h_settings(max_examples=50, deadline=None)
def test_qfq_factor_adjustment(symbol: str, trade_date: date, raw_price: float, qfq_factor: float) -> None:
    """前复权计算验证：收盘价除以因子，成交量乘以因子。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine, _ = make_engine_in(tmp_dir)

        # 模拟因子表
        df_factors = pd.DataFrame([{
            "code": symbol,
            "date": pd.to_datetime(trade_date),
            "qfq_factor": qfq_factor,
        }])
        engine._factors_df = df_factors

        raw_df = pd.DataFrame([{
            "symbol": symbol,
            "date": str(trade_date),
            "open": raw_price,
            "high": raw_price,
            "low": raw_price,
            "close": raw_price,
            "volume": 1000.0,
            "turnover": raw_price * 1000.0,
        }])

        adjusted_df = engine.apply_qfq(raw_df, symbol)
        expected_close = round(raw_price / qfq_factor, 3)
        expected_volume = round(1000.0 * qfq_factor, 0)

        assert abs(adjusted_df["close"].iloc[0] - expected_close) < 1e-3
        assert abs(adjusted_df["volume"].iloc[0] - expected_volume) < 1.0

