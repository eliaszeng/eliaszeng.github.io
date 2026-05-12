# Trading Data

存放从交易日记页面导出的 JSON 数据文件。

## 使用方式

1. 在交易日记页面点击「导出」按钮，下载 JSON 文件
2. 将下载的 JSON 文件放到此目录
3. 运行 `python scripts/trading/generate_charts.py` 生成K线图

## 文件格式

每个 JSON 文件为交易记录数组，格式与交易日记页面的导出格式一致。
