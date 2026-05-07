# eliaszeng.github.io

个人仪表盘 — 项目进度跟踪 + 大模型科技资讯聚合。部署在 GitHub Pages。

## 技术栈

- Astro 5 + Tailwind CSS 3
- GitHub Actions 自动部署
- Python 脚本抓取 Hacker News AI 资讯

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

## 项目结构

```
├── .github/workflows/
│   ├── deploy.yml          # push 到 main 自动构建部署
│   └── fetch-news.yml      # 每天定时抓取 HN 资讯
├── scripts/
│   └── fetch_news.py       # HN AI 资讯抓取脚本
├── src/
│   ├── components/
│   │   ├── ProjectCard.astro
│   │   └── StatsBar.astro
│   ├── data/
│   │   ├── projects.json   # 项目数据（手动维护）
│   │   └── news.json       # 资讯数据（自动更新）
│   ├── layouts/
│   │   └── BaseLayout.astro
│   └── pages/
│       ├── index.astro     # 首页
│       ├── projects.astro  # 项目看板
│       └── news.astro      # 科技资讯
├── astro.config.mjs
├── tailwind.config.mjs
└── package.json
```

## 访问地址

https://eliaszeng.github.io
