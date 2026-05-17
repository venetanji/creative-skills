import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const URL = 'https://hkpuhealthandsocial.au1.qualtrics.com/jfe/form/SV_8ijwehsuQEADz4a';
const OUT = '/home/user/creative-skills/qualtrics-test/pages';
fs.mkdirSync(OUT, { recursive: true });

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function captureCurrentPage(page, idx) {
  await page.waitForLoadState('networkidle').catch(() => {});
  await sleep(800);
  const shot = path.join(OUT, `page-${String(idx).padStart(2, '0')}.png`);
  await page.screenshot({ path: shot, fullPage: true });

  // Pull visible text from the survey body
  const data = await page.evaluate(() => {
    const root = document.querySelector('#Questions') || document.querySelector('.SurveyEngineBody') || document.body;
    const text = root ? root.innerText : document.body.innerText;
    // Identify each question block by .QuestionBody / .QuestionText
    const blocks = [...document.querySelectorAll('.QuestionOuter, .QuestionBody, [role="group"], .Skin .QuestionText')];
    const questions = [];
    document.querySelectorAll('.QuestionOuter').forEach((q) => {
      const qtxt = q.querySelector('.QuestionText')?.innerText?.trim();
      const choices = [...q.querySelectorAll('.QuestionBody label, .QuestionBody .LabelWrapper')].map(l => l.innerText.trim()).filter(Boolean);
      const inputs = [...q.querySelectorAll('input, textarea, select')].map(i => ({ type: i.type || i.tagName.toLowerCase(), name: i.name, id: i.id }));
      if (qtxt) questions.push({ qtxt, choices, inputs });
    });
    // header / progress
    const progress = document.querySelector('.ProgressBarFill')?.style?.width || '';
    const title = document.title;
    return { text, questions, progress, title };
  });

  const dump = path.join(OUT, `page-${String(idx).padStart(2, '0')}.json`);
  fs.writeFileSync(dump, JSON.stringify(data, null, 2));
  const txt = path.join(OUT, `page-${String(idx).padStart(2, '0')}.txt`);
  fs.writeFileSync(txt, data.text || '');
  console.log(`Captured page ${idx} (progress=${data.progress}, ${data.questions.length} questions)`);
  return data;
}

async function clickNext(page) {
  // Qualtrics next button selectors vary; try a few
  const candidates = [
    '#NextButton',
    'button#NextButton',
    'button[aria-label*="Next"]',
    'button:has-text("Next")',
    'input[id="NextButton"]',
    '#nextButton',
    'button:has-text(">>")',
  ];
  for (const sel of candidates) {
    const el = await page.$(sel);
    if (el && await el.isVisible().catch(() => false)) {
      await el.click().catch(() => {});
      return true;
    }
  }
  return false;
}

async function answerCurrentPage(page) {
  // Best-effort answering so we can advance through pages without triggering validation
  await page.evaluate(() => {
    // For each QuestionOuter, try to pick the first reasonable answer
    document.querySelectorAll('.QuestionOuter').forEach((q) => {
      // Pick first radio (single-answer multiple choice)
      const radios = q.querySelectorAll('input[type="radio"]');
      if (radios.length) {
        radios[0].click();
        return;
      }
      // Multi-select checkboxes — pick first
      const cbs = q.querySelectorAll('input[type="checkbox"]');
      if (cbs.length) {
        cbs[0].click();
        return;
      }
      // Dropdowns
      const sels = q.querySelectorAll('select');
      sels.forEach((s) => {
        const opt = [...s.options].find(o => o.value && !o.disabled);
        if (opt) { s.value = opt.value; s.dispatchEvent(new Event('change', { bubbles: true })); }
      });
      // Text inputs
      q.querySelectorAll('input[type="text"], textarea, input[type="email"], input[type="number"]').forEach((i) => {
        if (!i.value) {
          i.value = i.type === 'number' ? '30' : 'Test response';
          i.dispatchEvent(new Event('input', { bubbles: true }));
          i.dispatchEvent(new Event('change', { bubbles: true }));
        }
      });
    });
  });
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 1800 }, ignoreHTTPSErrors: true });
  const page = await ctx.newPage();
  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 60000 });

  let idx = 1;
  const maxPages = 30;
  let lastText = '';
  while (idx <= maxPages) {
    const data = await captureCurrentPage(page, idx);
    // Detect end-of-survey
    if (/thank you|response has been recorded|We thank you/i.test(data.text || '')) {
      console.log('Reached end of survey.');
      break;
    }
    // Answer questions on this page
    await answerCurrentPage(page);
    // Click next
    const advanced = await clickNext(page);
    if (!advanced) {
      console.log('No next button found — likely terminal page.');
      break;
    }
    await sleep(1500);
    // If page text didn't change, stop
    const newText = await page.evaluate(() => document.body.innerText);
    if (newText === lastText) {
      console.log('Page text unchanged — stopping.');
      break;
    }
    lastText = newText;
    idx++;
  }

  await browser.close();
})().catch((e) => { console.error('FATAL', e); process.exit(1); });
