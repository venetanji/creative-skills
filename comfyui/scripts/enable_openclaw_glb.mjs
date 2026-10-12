#!/usr/bin/env node
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

function findBundle(distDir, pattern, marker) {
  const matches = readdirSync(distDir)
    .filter((name) => pattern.test(name))
    .map((name) => join(distDir, name))
    .filter((path) => readFileSync(path, "utf8").includes(marker));
  if (matches.length !== 1) {
    throw new Error(`Expected one bundle for ${marker}, found ${matches.length}`);
  }
  return matches[0];
}

function replaceOnce(source, original, replacement) {
  if (source.includes(replacement)) {
    return source;
  }
  const count = source.split(original).length - 1;
  if (count !== 1) {
    throw new Error(`Expected one patch anchor, found ${count}: ${original}`);
  }
  return source.replace(original, replacement);
}

export function enableOpenClawGlb(distDir) {
  const rules = [
    {
      pattern: /^web-media-[^.]+\.mjs$/,
      marker: "const HOST_READ_ALLOWED_DOCUMENT_MIMES =",
      original: "const HOST_READ_ALLOWED_DOCUMENT_MIMES = /* @__PURE__ */ new Set([\n",
      replacement: "const HOST_READ_ALLOWED_DOCUMENT_MIMES = /* @__PURE__ */ new Set([\n\t\"model/gltf-binary\",\n",
    },
    {
      pattern: /^mime-[^.]+\.mjs$/,
      marker: "const EXT_BY_MIME = {",
      original: "const EXT_BY_MIME = {\n",
      replacement: "const EXT_BY_MIME = {\n\t\"model/gltf-binary\": \".glb\",\n",
    },
    {
      pattern: /^constants-[^.]+\.mjs$/,
      marker: "function mediaKindFromMime(mime) {",
      original: "function mediaKindFromMime(mime) {\n",
      replacement: "function mediaKindFromMime(mime) {\n\tif (mime === \"model/gltf-binary\") return \"document\";\n",
    },
  ];
  const updates = rules.map((rule) => {
    const path = findBundle(distDir, rule.pattern, rule.marker);
    const original = readFileSync(path, "utf8");
    return { path, original, patched: replaceOnce(original, rule.original, rule.replacement) };
  });
  const changed = [];
  for (const update of updates) {
    if (update.original !== update.patched) {
      writeFileSync(update.path, update.patched);
      changed.push(update.path);
    }
  }
  return changed;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const distDir = process.argv[2] ?? "/usr/local/lib/node_modules/openclaw/dist";
  const changed = enableOpenClawGlb(distDir);
  console.log(`Enabled content-sniffed GLB document attachments (${changed.length} bundles updated).`);
}
