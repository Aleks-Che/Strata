// Actual About/Monitor formatting functions with reported INFO fixtures.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, 'web/app.js'), 'utf8').replace(/\r\n/g, '\n');
const start = source.indexOf('function expertCacheText(');
const end = source.indexOf('\ndocument.addEventListener', start);
assert.ok(start >= 0 && end > start);
let rows = {};
const context = vm.createContext({$: id => id, fmt: n => String(n), gb: b => String(b / 1073741824),
  projectionText: () => null, location: {origin: 'http://127.0.0.1'},
  facts: (id, values) => rows[id] = Object.fromEntries(values.filter(x => x[1] != null && x[1] !== ''))});
vm.runInContext(source.slice(start, end), context);
function about(info) { rows = {}; context.renderAbout(info, {}, {}); return rows['facts-engine']; }
const minimal = about({architecture: 'glm5next', model: 'glm-test'});
assert.equal(minimal.Architecture, 'glm5next');
for (const key of ['Speculation', 'Expert placement', 'Expert transfer pipeline', 'Expert cache policy']) {
  assert.ok(!(key in minimal), key);
}
// Same reported runtime facts work for GLM, DeepSeek or a new architecture.
const engine = {expert_compute: 'gpu', expert_storage: 'mmap', gpu_expert_layers: 0,
  expert_pipeline: 1, expert_pipeline_slots: 4, expert_readers: 2, expert_read_mode: 'native',
  expert_cache_policy: 'frequency', expert_cached_matrices: 0, expert_cache_mib: 1024,
  spec: 0, speculative: 'none'};
for (const architecture of ['glm5next', 'deepseek4', 'future-model']) {
  const facts = about({...engine, architecture});
  assert.equal(facts.Speculation, 'Off');
  assert.equal(facts['Expert matrices in VRAM'], '0 matrices · 1 GiB budget');
  assert.match(facts['Expert placement'], /Compute: gpu; 0 resident expert layers; source: mmap/);
  assert.match(facts['Expert transfer pipeline'], /4 staging slots.*2 readers.*native/);
  assert.match(facts['Expert cache policy'], /frequently/);
}
assert.equal(context.expertCacheText({expert_slots: 2, expert_cache_live_bytes: 1073741824}), '2 experts · 1 GiB used');
assert.equal(context.expertCacheText({expert_slots: 0}), '0 experts');
assert.equal(context.expertCacheText({}), null);
assert.equal(context.expertPolicyText({}), null);
assert.equal(context.expertPolicyText({expert_cache_policy: 'custom'}), 'custom');
assert.equal(context.expertPipelineText({expert_pipeline: 0}), 'Off');
assert.equal(context.expertPipelineText({expert_pipeline: 1}), 'Enabled');
// Unknown speculation must not be guessed to be MTP, regardless of architecture.
assert.equal(context.speculationText({architecture: 'glm5next', spec: 4}), 'Enabled; draft strategy not reported');
assert.equal(context.speculationText({speculative: 'mtp', draft_tokens: 3}), 'Native MTP: up to 3 draft tokens');
assert.match(context.speculationText({speculative: 'dspark', spec: 3}), /^DSpark.*3 draft tokens/);
assert.equal(context.speculationText({mtp_max: 4, spec: 4, lookup: 1}), 'MTP drafts up to 3 tokens, prompt lookup on');
assert.equal(context.speculationText({mtp_max: 0, spec: 4}), 'MTP drafts up to 3 tokens');
console.log('Engine facts and About/Monitor formatting checks passed');
