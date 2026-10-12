import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import { enableOpenClawGlb } from "./enable_openclaw_glb.mjs";

function fixture(context) {
  const directory = mkdtempSync(join(tmpdir(), "openclaw-glb-"));
  context.after(() => rmSync(directory, { recursive: true, force: true }));
  writeFileSync(join(directory, "web-media-test.mjs"),
    "const HOST_READ_ALLOWED_DOCUMENT_MIMES = /* @__PURE__ */ new Set([\n\t\"application/pdf\"\n]);\nexport { HOST_READ_ALLOWED_DOCUMENT_MIMES };\n");
  writeFileSync(join(directory, "mime-test.mjs"),
    "const EXT_BY_MIME = {\n\t\"application/pdf\": \".pdf\"\n};\nexport { EXT_BY_MIME };\n");
  writeFileSync(join(directory, "constants-test.mjs"),
    "function mediaKindFromMime(mime) {\n\tif (mime === \"application/pdf\") return \"document\";\n}\nexport { mediaKindFromMime };\n");
  return directory;
}

test("GLB is an allowed document with a .glb MIME mapping", async (context) => {
  const directory = fixture(context);
  assert.equal(enableOpenClawGlb(directory).length, 3);
  const policy = await import(pathToFileURL(join(directory, "web-media-test.mjs")));
  const mime = await import(pathToFileURL(join(directory, "mime-test.mjs")));
  const kinds = await import(pathToFileURL(join(directory, "constants-test.mjs")));
  assert.equal(policy.HOST_READ_ALLOWED_DOCUMENT_MIMES.has("model/gltf-binary"), true);
  assert.equal(policy.HOST_READ_ALLOWED_DOCUMENT_MIMES.has("application/pdf"), true);
  assert.equal(policy.HOST_READ_ALLOWED_DOCUMENT_MIMES.has("application/octet-stream"), false);
  assert.equal(policy.HOST_READ_ALLOWED_DOCUMENT_MIMES.has("model/gltf+json"), false);
  assert.equal(mime.EXT_BY_MIME["model/gltf-binary"], ".glb");
  assert.equal(kinds.mediaKindFromMime("model/gltf-binary"), "document");
  assert.equal(kinds.mediaKindFromMime("application/pdf"), "document");
  assert.equal(kinds.mediaKindFromMime("model/unknown"), undefined);
});

test("reapplying the patch leaves files unchanged", (context) => {
  const directory = fixture(context);
  const changed = enableOpenClawGlb(directory);
  const sources = changed.map((path) => readFileSync(path, "utf8"));
  assert.deepEqual(enableOpenClawGlb(directory), []);
  assert.deepEqual(changed.map((path) => readFileSync(path, "utf8")), sources);
});

test("the newer classifier does not need a separate PDF branch", async (context) => {
  const directory = fixture(context);
  writeFileSync(join(directory, "constants-test.mjs"),
    "function mediaKindFromMime(mime) {\n\tif (!mime) return;\n\tif (mime.startsWith(\"application/\")) return \"document\";\n}\nexport { mediaKindFromMime };\n");
  assert.equal(enableOpenClawGlb(directory).length, 3);
  const kinds = await import(pathToFileURL(join(directory, "constants-test.mjs")));
  assert.equal(kinds.mediaKindFromMime("model/gltf-binary"), "document");
  assert.equal(kinds.mediaKindFromMime("application/pdf"), "document");
  assert.equal(kinds.mediaKindFromMime("model/unknown"), undefined);
  assert.equal(kinds.mediaKindFromMime(undefined), undefined);
});

test("an unknown bundle format fails before writing any files", (context) => {
  const directory = fixture(context);
  const path = join(directory, "constants-test.mjs");
  writeFileSync(path, "function mediaKindFromMime(mime) { return mime; }\n");
  const policyPath = join(directory, "web-media-test.mjs");
  const original = readFileSync(policyPath, "utf8");
  assert.throws(() => enableOpenClawGlb(directory), /patch anchor/);
  assert.equal(readFileSync(policyPath, "utf8"), original);
});

test("ambiguous bundle names fail without weakening the allowlist", (context) => {
  const directory = fixture(context);
  writeFileSync(join(directory, "web-media-duplicate.mjs"),
    readFileSync(join(directory, "web-media-test.mjs"), "utf8"));
  assert.throws(() => enableOpenClawGlb(directory), /found 2/);
});

const liveDistDir = process.env.OPENCLAW_DIST_DIR;
const liveGlbFile = process.env.GLB_TEST_FILE;

async function liveBundle(pattern, marker) {
  const paths = readdirSync(liveDistDir)
    .filter((name) => pattern.test(name))
    .map((name) => join(liveDistDir, name))
    .filter((path) => readFileSync(path, "utf8").includes(marker));
  assert.equal(paths.length, 1);
  return import(pathToFileURL(paths[0]));
}

test("the real GLB passes OpenClaw's host-local attachment gate", {
  skip: !liveDistDir || !liveGlbFile,
}, async () => {
  const media = await liveBundle(/^web-media-[^.]+\.mjs$/, "const HOST_READ_ALLOWED_DOCUMENT_MIMES =");
  const mime = await liveBundle(/^mime-[^.]+\.mjs$/, "const EXT_BY_MIME = {");
  const result = await media.r(liveGlbFile, {
    hostReadCapability: true,
    localRoots: [dirname(liveGlbFile)],
  });
  assert.equal(result.contentType, "model/gltf-binary");
  assert.equal(result.kind, "document");
  assert.equal(result.buffer.length, readFileSync(liveGlbFile).length);
  assert.equal(mime.u(liveGlbFile), "model/gltf-binary");
  assert.equal(mime.r("model/gltf-binary"), ".glb");
});

test("renaming unsupported bytes to .glb does not bypass the gate", {
  skip: !liveDistDir,
}, async (context) => {
  const directory = fixture(context);
  const media = await liveBundle(/^web-media-[^.]+\.mjs$/, "const HOST_READ_ALLOWED_DOCUMENT_MIMES =");
  const path = join(directory, "spoof.glb");
  for (const bytes of [Buffer.from([0, 1, 2, 3]), Buffer.from("This is plain text, not a GLB.")]) {
    writeFileSync(path, bytes);
    await assert.rejects(media.r(path, {
      hostReadCapability: true,
      localRoots: [directory],
    }), /buffer-verified/);
  }
});
