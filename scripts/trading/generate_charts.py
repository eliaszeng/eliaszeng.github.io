"""
从 trades.json / advice.json 读取交易记录，
用 akshare 获取K线数据，用 mplfinance 绘图，
输出到 public/trading/charts/。
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import matplotlib
import mplfinance as mpf
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent.parent
TRADES_FILE = ROOT / "public" / "trading" / "data" / "trades.json"
ADVICE_FILE = ROOT / "public" / "trading" / "data" / "advice.json"
CHARTS_DIR = ROOT / "public" / "trading" / "charts"


def load_json(path: Path) -> list:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: list):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_kline(stock_code: str, days: int = 60) -> pd.DataFrame | None:
    """获取最近N个交易日的日K线数据。"""
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(
            symbol=stock_code, period="daily",
            start_date=start_date, end_date=end_date, adjust="qfq",
        )
        if df.empty:
            return None
        df = df.rename(columns={
            "日期": "Date", "开盘": "Open", "收盘": "Close",
            "最高": "High", "最低": "Low", "成交量": "Volume",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        return df[["Open", "Close", "High", "Low", "Volume"]].tail(days)
    except Exception as e:
        print(f"  [WARN] 获取 {stock_code} K线失败: {e}", file=sys.stderr)
        return None


def plot_chart(df: pd.DataFrame, stock_code: str, output_path: Path):
    """绘制K线图并保存为 PNG。"""
    mc = mpf.make_marketcolors(
        up="#c2553a", down="#5a6b4a",
        edge="inherit", wick="inherit", volume="inherit",
    )
    style = mpf.make_mpf_style(
        marketcolors=mc, base_mpf_style="charles",
        gridstyle=":", gridcolor="#ede9e1",
        facecolor="#ffffff", edgecolor="#ede9e1",
        rc={"font.size": 9},
    )
    fig, axes = mpf.plot(
        df, type="candle", style=style, volume=True,
        title=f"\n{stock_code}", returnfig=True,
        figsize=(10, 6), tight_layout=True,
    )
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    matplotlib.pyplot.close(fig)


def process_records(records: list, date_field: str) -> bool:
    """处理记录列表，生成K线图并更新 chartPath。返回是否有更新。"""
    updated = False
    for rec in records:
        code = rec.get("stockCode", "")
        date_str = rec.get(date_field, "")[:10].replace("-", "")
        if not code or not date_str:
            continue

        chart_name = f"{code}_{date_str}.png"
        chart_path = CHARTS_DIR / chart_name
        rec["chartPath"] = f"/trading/charts/{chart_name}"

        if chart_path.exists():
            print(f"  [SKIP] {chart_name} 已存在")
            continue

        print(f"  [GEN]  {chart_name} ...")
        df = fetch_kline(code)
        if df is not None:
            plot_chart(df, code, chart_path)
            updated = True
        else:
            rec["chartPath"] = ""

    return updated


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 加载交易数据 ===")
    trades = load_json(TRADES_FILE)
    advice = load_json(ADVICE_FILE)
    print(f"  交易记录: {len(trades)} 条")
    print(f"  建议记录: {len(advice)} 条")

    print("\n=== 生成交易K线图 ===")
    trades_updated = process_records(trades, "tradeTime")

    print("\n=== 生成建议K线图 ===")
    advice_updated = process_records(advice, "createdAt")

    if trades_updated:
        save_json(TRADES_FILE, trades)
        print("\n  已更新 trades.json")
    if advice_updated:
        save_json(ADVICE_FILE, advice)
        print("\n  已更新 advice.json")

    print("\n完成！")


if __name__ == "__main__":
    main()
