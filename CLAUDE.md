# eliaszeng.github.io

个人仪表盘 — 项目进度跟踪 + 大模型科技资讯聚合。部署在 GitHub Pages。

## 技术栈

- Astro 5 + Tailwind CSS 3
- GitHub Actions 自动部署
- Python 脚本：Hacker News 资讯抓取 + 交易K线图生成 + 截图识别

## 本地开发

```bash
cd D:/claude_project/eliaszeng.github.io
npm install          # 安装依赖（需 npmmirror 或科学上网）
npm run dev          # 启动开发服务器 http://localhost:4321
npm run build        # 构建静态文件到 dist/
```

## 更新项目进度

编辑 `src/data/projects.json`，字段说明：

| 字段 | 说明 |
|------|------|
| id | 唯一标识 |
| name | 项目名称 |
| description | 简短描述 |
| status | `active` / `planned` / `paused` / `completed` |
| progress | 0-100 进度百分比 |
| tags | 技术标签数组 |
| link | 项目链接（可选） |
| startDate | 开始日期 |

## 更新资讯

资讯由 GitHub Actions 每天北京时间 10:00 自动从 Hacker News 抓取 AI/LLM 相关内容。

手动抓取：
```bash
python scripts/fetch_news.py
```

结果写入 `src/data/news.json`。

## 部署

推送到 `main` 分支自动触发 GitHub Actions 构建和部署。

```bash
git add -A
git commit -m "update: 描述"
git push origin main
```

## 仓库安全

- 仓库为 Public，但只有 owner (eliaszeng) 有 push 权限
- 外部贡献者只能 fork 后提 PR，需手动合并
- GitHub Actions deploy 环境绑定 owner 账号

## 交易日记

交易日记（trading.astro）是纯展示页面，数据由后端 Python 脚本生成。

### 数据文件

- `public/trading/data/trades.json` — 交易记录数组
- `public/trading/data/advice.json` — 建议记录数组
- `public/trading/charts/` — K线图 PNG（`{股票代码}_{日期}.png`）

前端通过 `fetch()` 读取 JSON 文件并渲染，无需 IndexedDB。

### 生成K线图

```bash
pip install -r scripts/trading/requirements.txt
python scripts/trading/generate_charts.py
```

从 JSON 读取记录，用 akshare 获取K线数据，用 mplfinance 绘图输出到 `public/trading/charts/`。

GitHub Actions (`update-trading.yml`) 支持手动触发，自动生成图表并提交。

### 截图识别

```bash
export VISION_API_KEY=your_key
export VISION_BASE_URL=https://api.siliconflow.cn
export VISION_MODEL=Qwen/Qwen2.5-VL-72B-Instruct
python scripts/trading/recognize_screenshot.py /path/to/screenshot.png
```

识别截图中的交易信息，输出结构化 JSON。可配合 `--output` 写入文件。

## 项目结构

```
├── .github/workflows/
│   ├── deploy.yml              # push 到 main 自动构建部署
│   ├── fetch-news.yml          # 每天定时抓取 HN 资讯
│   └── update-trading.yml      # 手动触发，生成交易K线图
├── scripts/
│   ├── fetch_news.py           # HN AI 资讯抓取脚本
│   └── trading/
│       ├── generate_charts.py      # 读取 JSON 生成K线图
│       ├── recognize_screenshot.py # 截图 AI 识别
│       └── requirements.txt        # Python 依赖
├── public/
│   └── trading/
│       ├── data/
│       │   ├── trades.json     # 交易记录（后端生成）
│       │   └── advice.json     # 建议记录（后端生成）
│       └── charts/             # K线图 PNG（后端生成）
├── src/
│   ├── components/
│   │   ├── ProjectCard.astro
│   │   └── StatsBar.astro
│   ├── data/
│   │   ├── projects.json       # 项目数据（手动维护）
│   │   └── news.json           # 资讯数据（自动更新）
│   ├── layouts/
│   │   └── BaseLayout.astro
│   └── pages/
│       ├── index.astro         # 首页
│       ├── projects.astro      # 项目看板
│       ├── news.astro          # 科技资讯
│       └── trading.astro       # 交易日记（纯展示）
├── astro.config.mjs
├── tailwind.config.mjs
└── package.json
```

## 访问地址

https://eliaszeng.github.io
