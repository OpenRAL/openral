// Record the live dashboard as a video (smooth, full-page) via playwright.
// Uses a tall viewport so the whole page is in-frame and records it for a
// duration while a policy drives the robot. Output: a .webm in <outDir>.
// Usage: node record_dashboard.js <url> <outDir> <durationMs>
const PW = '/home/allopart/.npm/_npx/e41f203b7505f1fb/node_modules/playwright';
const { chromium } = require(PW);

(async () => {
  const url = process.argv[2] || 'http://127.0.0.1:4318/';
  const outDir = process.argv[3] || '/tmp/dashvid';
  const durationMs = parseInt(process.argv[4] || '30000', 10);
  const vw = 1280;

  // Probe the page height first.
  const probe = await chromium.launch({ args: ['--disable-gpu', '--hide-scrollbars'] });
  const pctx = await probe.newContext({ viewport: { width: vw, height: 1000 } });
  const ppage = await pctx.newPage();
  await ppage.goto(url, { waitUntil: 'domcontentloaded' }).catch(() => {});
  await ppage.waitForTimeout(1800);
  let h = await ppage.evaluate(() => document.body.scrollHeight).catch(() => 2200);
  await probe.close();
  h = Math.min(Math.max(Math.round(h), 1000), 3800);

  // Record the full-height viewport for the duration.
  const browser = await chromium.launch({ args: ['--disable-gpu', '--hide-scrollbars'] });
  const context = await browser.newContext({
    viewport: { width: vw, height: h },
    recordVideo: { dir: outDir, size: { width: vw, height: h } },
  });
  const page = await context.newPage();
  await page.goto(url, { waitUntil: 'domcontentloaded' }).catch(() => {});
  await page.waitForTimeout(durationMs);
  await context.close();
  await browser.close();
  console.log('recorded height=' + h + ' duration=' + durationMs + 'ms dir=' + outDir);
})();
