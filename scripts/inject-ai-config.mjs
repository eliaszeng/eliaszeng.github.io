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

  if (!config.endpoint || !config.key) {
    console.log('[inject-ai-config] 配置中缺少 endpoint 或 key，跳过注入');
    process.exit(0);
  }

  fs.writeFileSync(outPath, JSON.stringify(config, null, 2));
  console.log(`[inject-ai-config] 已注入到 public/ai-config.json`);
  console.log(`  endpoint: ${config.endpoint}`);
  console.log(`  model:    ${config.model}`);
  console.log(`  key:      ${config.key.slice(0, 8)}...`);
} catch (e) {
  // 配置文件不存在或读取失败，静默跳过
  console.log(`[inject-ai-config] 未找到 Claude 配置，跳过: ${e.message}`);
}
