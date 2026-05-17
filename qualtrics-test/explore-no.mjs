import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const URL = 'https://hkpuhealthandsocial.au1.qualtrics.com/jfe/form/SV_8ijwehsuQEADz4a';
const OUT = '/home/user/creative-skills/qualtrics-test/pages-no';
fs.mkdirSync(OUT, { recursive: true });

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function captureCurrentPage(page, idx) {
  await page.waitForLoadState('networkidle').catch(() => {});
  await sleep(900);
  const shot = path.join(OUT, `page-${String(idx).padStart(2, '0')}.png`);
  await page.screenshot({ path: shot, fullPage: true });

  const data = await page.evaluate(() => {
    const questions = [];
    document.querySelectorAll('.QuestionOuter').forEach((q) => {
      const qtxt = q.querySelector('.QuestionText')?.innerText?.trim();
      const choices = [...q.querySelectorAll('.QuestionBody label, .QuestionBody .LabelWrapper')].map(l => l.innerText.trim()).filter(Boolean);
      const inputs = [...q.querySelectorAll('input, textarea, select')].map(i => ({ type: i.type || i.tagName.toLowerCase(), name: i.name, id: i.id }));
      const errors = q.querySelector('.ValidationError')?.innerText?.trim() || '';
      if (qtxt) questions.push({ qtxt, choices, inputs, errors });
    });
    return { text: document.body.innerText, questions };
  });

  fs.writeFileSync(path.join(OUT, `page-${String(idx).padStart(2, '0')}.json`), JSON.stringify(data, null, 2));
  fs.writeFileSync(path.join(OUT, `page-${String(idx).padStart(2, '0')}.txt`), data.text || '');
  console.log(`Captured page ${idx} (${data.questions.length} questions)`);
  return data;
}

async function clickNext(page) {
  const candidates = ['#NextButton', 'button#NextButton', 'button[aria-label*="Next"]', 'button:has-text("Next")'];
  for (const sel of candidates) {
    const el = await page.$(sel);
    if (el && await el.isVisible().catch(() => false)) { await el.click().catch(() => {}); return true; }
  }
  return false;
}

async function answerCurrentPage(page, opts = {}) {
  await page.evaluate((opts) => {
    document.querySelectorAll('.QuestionOuter').forEach((q) => {
      const qtxt = q.querySelector('.QuestionText')?.innerText?.trim() || '';
      const radios = q.querySelectorAll('input[type="radio"]');
      if (radios.length) {
        // If the question is the gating "已完成問卷部分" and we want NO, pick the second radio (否)
        if (opts.firstPickNo && qtxt.includes('完成問卷部分')) {
          radios[1].click();
        } else {
          radios[0].click();
        }
        return;
      }
      const cbs = q.querySelectorAll('input[type="checkbox"]');
      if (cbs.length) { cbs[0].click(); return; }
      const sels = q.querySelectorAll('select');
      sels.forEach((s) => {
        const opt = [...s.options].find(o => o.value && !o.disabled);
        if (opt) { s.value = opt.value; s.dispatchEvent(new Event('change', { bubbles: true })); }
      });
      q.querySelectorAll('input[type="text"], textarea, input[type="email"], input[type="number"], input[type="tel"]').forEach((i) => {
        if (!i.value) {
          const label = (i.closest('.QuestionOuter')?.innerText || '') + ' ' + (i.placeholder || '') + ' ' + (i.name || '') + ' ' + qtxt;
          if (/年齡|age/i.test(label)) i.value = '45';
          else if (/電郵|email/i.test(label)) i.value = 'test@example.com';
          else if (/電話|phone|tel/i.test(label)) i.value = '51234567';
          else if (/收縮/.test(label)) i.value = '120';
          else if (/舒張/.test(label)) i.value = '80';
          else if (/脈搏|pulse/i.test(label)) i.value = '72';
          else if (/血糖/.test(label)) i.value = '5.2';
          else if (i.type === 'number' || /數字|數值|height|weight|身高|體重/i.test(label)) i.value = '60';
          else if (/英文姓氏|surname|last name/i.test(label)) i.value = 'Tester';
          else if (/英文名字|first name/i.test(label)) i.value = 'Test';
          else if (/姓氏|中文/.test(label)) i.value = '陳';
          else if (/名字/.test(label)) i.value = '大文';
          else i.value = 'Test';
          i.dispatchEvent(new Event('input', { bubbles: true }));
          i.dispatchEvent(new Event('change', { bubbles: true }));
        }
      });
    });
  }, opts);
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 1800 }, ignoreHTTPSErrors: true, locale: 'zh-HK' });
  const page = await ctx.newPage();
  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 60000 });

  let idx = 1;
  let firstPage = true;
  let stuckCount = 0;
  let prevQtxt = '';
  while (idx <= 80) {
    const data = await captureCurrentPage(page, idx);
    if (/問卷已成功完成|response has been recorded|2028年再見/i.test(data.text || '')) {
      console.log('Reached end.');
      break;
    }
    await answerCurrentPage(page, { firstPickNo: firstPage });
    firstPage = false;
    const advanced = await clickNext(page);
    if (!advanced) { console.log('No next button — done.'); break; }
    await sleep(1500);
    // Detect stuck-on-same-page
    const curQtxt = JSON.stringify(data.questions.map(q => q.qtxt));
    if (curQtxt === prevQtxt) {
      stuckCount++;
      if (stuckCount > 2) {
        console.log('Stuck on same page after multiple advance attempts; capturing and stopping.');
        break;
      }
    } else { stuckCount = 0; }
    prevQtxt = curQtxt;
    idx++;
  }

  await browser.close();
})().catch((e) => { console.error('FATAL', e); process.exit(1); });
