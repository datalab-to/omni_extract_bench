// Drive one headless Chrome over the DevTools protocol and write one PNG per frame.
import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { resolve } from 'node:path';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const HERE   = resolve(import.meta.dirname);
const OUT    = resolve(HERE, 'frames');
const PORT   = 9339;
const W = 1100, H = 560;

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

const chrome = spawn(CHROME, [
  '--headless=new', '--disable-gpu', '--hide-scrollbars', '--mute-audio',
  '--no-first-run', '--no-default-browser-check', '--disable-extensions',
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${resolve(HERE, 'chrome-profile')}`,
  `--window-size=${W},${H}`, '--force-device-scale-factor=1',
  `file://${resolve(HERE, 'film.html')}`,
], { stdio: ['ignore', 'ignore', 'ignore'] });

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function target() {
  for (let i = 0; i < 60; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = list.find(t => t.type === 'page' && t.webSocketDebuggerUrl);
      if (page) return page.webSocketDebuggerUrl;
    } catch {}
    await sleep(250);
  }
  throw new Error('Chrome never exposed a page target');
}

const ws = new WebSocket(await target());
await new Promise((ok, no) => { ws.onopen = ok; ws.onerror = no; });

let id = 0;
const waiting = new Map();
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) {
    const { ok, no } = waiting.get(m.id); waiting.delete(m.id);
    m.error ? no(new Error(JSON.stringify(m.error))) : ok(m.result);
  }
};
const send = (method, params = {}) => new Promise((ok, no) => {
  const n = ++id; waiting.set(n, { ok, no });
  ws.send(JSON.stringify({ id: n, method, params }));
});

const evaluate = async (expression) => (await send('Runtime.evaluate',
  { expression, awaitPromise: true, returnByValue: true })).result.value;

await send('Page.enable');
// --window-size includes browser chrome even in headless=new, so pin the viewport here.
await send('Emulation.setDeviceMetricsOverride',
  { width: W, height: H, deviceScaleFactor: 2, mobile: false });

// The page may still be loading when we attach; wait for FILM and its fonts.
for (let i = 0; i < 80; i++) {
  if (await evaluate('!!window.FILM')) break;
  await sleep(250);
}
await evaluate('window.FILM.ready');
await sleep(400);                       // one beat for the first paint to settle

const count  = await evaluate('window.FILM.count');
const delays = await evaluate('JSON.stringify(window.FILM.delays)');
writeFileSync(resolve(OUT, 'delays.json'), delays);

for (let i = 0; i < count; i++) {
  await evaluate(`window.FILM.seek(${i})`);
  const shot = await send('Page.captureScreenshot',
    { format: 'png', captureBeyondViewport: false, optimizeForSpeed: false });
  writeFileSync(resolve(OUT, `f${String(i).padStart(4, '0')}.png`),
                Buffer.from(shot.data, 'base64'));
  if (i % 20 === 0) process.stdout.write(`  ${i}/${count}\n`);
}

console.log(`captured ${count} frames -> ${OUT}`);
ws.close();
chrome.kill('SIGKILL');
process.exit(0);
