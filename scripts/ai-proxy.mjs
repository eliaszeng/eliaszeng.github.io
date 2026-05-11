/**
 * 本地 AI 代理服务器
 * 读取 ~/.claude/settings.json 中的 API 配置，将 OpenAI 格式请求转换为 Anthropic 格式转发。
 * 用法: node scripts/ai-proxy.mjs
 * 交易页面设置 Base URL 为 http://localhost:3456 即可，无需填 API Key。
 */

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

// 读取 Claude Code 配置
function loadConfig() {
  const configPath = path.join(os.homedir(), '.claude', 'settings.json');
  try {
    const raw = fs.readFileSync(configPath, 'utf-8');
    const settings = JSON.parse(raw);
    const env = settings.env || {};
    return {
      baseUrl: env.ANTHROPIC_BASE_URL || 'https://api.anthropic.com',
      apiKey: env.ANTHROPIC_AUTH_TOKEN || process.env.ANTHROPIC_API_KEY || '',
      model: env.ANTHROPIC_MODEL || 'claude-sonnet-4-20250514',
    };
  } catch (e) {
    console.error(`无法读取配置 ${configPath}:`, e.message);
    process.exit(1);
  }
}

const config = loadConfig();
console.log(`配置已加载:`);
console.log(`  Base URL: ${config.baseUrl}`);
console.log(`  Model:    ${config.model}`);
console.log(`  API Key:  ${config.apiKey.slice(0, 8)}...`);
console.log();

// OpenAI → Anthropic 格式转换
function openaiToAnthropic(body) {
  const messages = body.messages || [];

  // 提取 system message
  let systemPrompt = '';
  const userMessages = [];
  for (const msg of messages) {
    if (msg.role === 'system') {
      systemPrompt += (systemPrompt ? '\n' : '') + msg.content;
    } else {
      userMessages.push(msg);
    }
  }

  // 转换 content 格式
  const anthropicMessages = userMessages.map(msg => {
    if (typeof msg.content === 'string') {
      return { role: msg.role, content: msg.content };
    }
    // 数组格式（包含 text + image_url）
    const content = [];
    for (const part of (msg.content || [])) {
      if (part.type === 'text') {
        content.push({ type: 'text', text: part.text });
      } else if (part.type === 'image_url') {
        const url = part.image_url?.url || '';
        if (url.startsWith('data:')) {
          // base64 data URL
          const match = url.match(/^data:(image\/\w+);base64,(.+)$/);
          if (match) {
            content.push({
              type: 'image',
              source: { type: 'base64', media_type: match[1], data: match[2] }
            });
          }
        } else {
          // 普通 URL
          content.push({
            type: 'image',
            source: { type: 'url', url }
          });
        }
      }
    }
    return { role: msg.role, content };
  });

  return {
    model: body.model || config.model,
    max_tokens: body.max_tokens || 1024,
    system: systemPrompt || undefined,
    messages: anthropicMessages,
  };
}

// Anthropic → OpenAI 响应格式转换
function anthropicToOpenai(anthropicResp) {
  const text = anthropicResp.content?.map(c => c.text || '').join('') || '';
  return {
    choices: [{
      message: { role: 'assistant', content: text },
      finish_reason: 'stop',
    }],
    model: anthropicResp.model,
    usage: anthropicResp.usage,
  };
}

const server = http.createServer(async (req, res) => {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // 健康检查
  if (req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', model: config.model }));
    return;
  }

  // 只处理 /v1/chat/completions 和 /chat/completions
  if (!req.url?.includes('/chat/completions')) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not Found. Use /v1/chat/completions' }));
    return;
  }

  try {
    // 读取请求体
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const bodyStr = Buffer.concat(chunks).toString();
    const openaiBody = JSON.parse(bodyStr);

    // 转换为 Anthropic 格式
    const anthropicBody = openaiToAnthropic(openaiBody);
    console.log(`[${new Date().toLocaleTimeString()}] 请求: ${openaiBody.messages?.length} 条消息, ${openaiBody.messages?.filter(m => Array.isArray(m.content) && m.content.some(p => p.type === 'image_url')).length} 条含图片`);

    // 转发到 Anthropic API
    const apiUrl = `${config.baseUrl}/v1/messages`;
    const apiRes = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': config.apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify(anthropicBody),
    });

    if (!apiRes.ok) {
      const errText = await apiRes.text();
      console.error(`API 错误 (${apiRes.status}):`, errText.slice(0, 200));
      res.writeHead(apiRes.status, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { message: `API error: ${apiRes.status}`, detail: errText.slice(0, 500) } }));
      return;
    }

    const anthropicResp = await apiRes.json();
    const openaiResp = anthropicToOpenai(anthropicResp);

    console.log(`[${new Date().toLocaleTimeString()}] 成功, tokens: ${anthropicResp.usage?.input_token_count || '?'}+${anthropicResp.usage?.output_token_count || '?'}`);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(openaiResp));
  } catch (e) {
    console.error('代理错误:', e.message);
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { message: e.message } }));
  }
});

const PORT = 3456;
server.listen(PORT, () => {
  console.log(`代理已启动: http://localhost:${PORT}`);
  console.log(`交易页面设置 Base URL 为 http://localhost:${PORT}，无需填 API Key`);
  console.log(`按 Ctrl+C 停止`);
});
