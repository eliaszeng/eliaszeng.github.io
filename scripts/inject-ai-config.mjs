/**
 * 从 ~/.claude/settings.json 读取 API 配置，注入到 public/ai-config.json
 * 用法: node scripts/inject-ai-config.mjs
 *
 * 注入后交易页面会自动加载配置，无需手动填写。
 * 注意: ai-config.json 会包含 API Key，仅适合个人私有部署。
 */

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const configPath = path.join(os.homedir(), '.claude', 'settings.json');
const outPath = path.join(process.cwd(), 'public', 'ai-config.json');

try {
  const raw = fs.readFileSync(configPath, 'utf-8');
  const settings = JSON.parse(raw);
  const env = settings.env || {};

  const config = {
    endpoint: env.ANTHROPIC_BASE_URL || '',
    model: env.ANTHROPIC_MODEL || '',
    key: env.ANTHROPIC_AUTH_TOKEN || '',
  };

  fs.writeFileSync(outPath, JSON.stringify(config, null, 2));
  console.log(`已注入 AI 配置到 public/ai-config.json`);
  console.log(`  endpoint: ${config.endpoint}`);
  console.log(`  model:    ${config.model}`);
  console.log(`  key:      ${config.key.slice(0, 8)}...`);
} catch (e) {
  console.error(`读取配置失败: ${e.message}`);
  process.exit(1);
}
