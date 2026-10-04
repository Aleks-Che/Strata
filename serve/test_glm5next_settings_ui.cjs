// Execute the web app's real settings functions with a minimal DOM, no browser/GPU.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, 'web/app.js'), 'utf8');
const elements = {};
function element(id) {
  return elements[id] ||= {hidden: false, value: '', attrs: {},
    setAttribute(k, v) { this.attrs[k] = v; }, getAttribute(k) { return this.attrs[k]; }};
}
element('s-thinking').children = ['none', 'low', 'medium', 'high', 'max'].map(v => ({
  ...element('effort-' + v), dataset: {v}, attrs: {},
}));
let saved = null;
const context = vm.createContext({$: element, store: {get: () => saved},
  health: {}, settings: {thinking: 'high', temperature: 0.6}, sharedOn: false,
  projectionLoaded: () => false, outputs: () => {}});
const functions = source.slice(source.indexOf('function reasoningCapabilities()'),
                               source.indexOf('// "Use for other apps too"'));
const shared = source.slice(source.indexOf('function sharedDefaults(s)'), source.indexOf('async function saveShared'));
vm.runInContext(functions + shared, context);
const run = code => JSON.parse(JSON.stringify(vm.runInContext(code, context)));
// Fresh GLM settings use max; Off/Medium disappear and the clear switch appears.
context.health.reasoning = {efforts: ['low', 'high', 'max'], default: 'max', clear_thinking: true};
vm.runInContext('applyReasoningCapabilities()', context);
assert.equal(context.settings.thinking, 'max');
assert.deepEqual(element('s-thinking').children.filter(b => !b.hidden).map(b => b.dataset.v), ['low', 'high', 'max']);
assert.equal(element('clear-thinking-row').hidden, false);
assert.equal(element('s-clear-thinking').attrs['aria-checked'], 'false');
// Stale Qwen Off migrates to max; explicit saved High stays High.
saved = {thinking: 'none'};
context.settings.thinking = 'none';
vm.runInContext('applyReasoningCapabilities()', context);
assert.equal(context.settings.thinking, 'max');
context.settings.thinking = 'high';
vm.runInContext('applyReasoningCapabilities()', context);
assert.equal(context.settings.thinking, 'high');
assert.deepEqual(run('reasoningRequest({thinking:"low",clear_thinking:true})'),
                 {reasoning_effort: 'low', clear_thinking: true});
assert.deepEqual(run('sharedDefaults({thinking:"max",clear_thinking:false,temperature:0})'),
                 {reasoning_effort: 'max', clear_thinking: false, temperature: 0});
// An old server/Qwen keeps original choices and never receives clear_thinking.
context.health = {};
context.settings = {thinking: 'max', clear_thinking: true};
vm.runInContext('applyReasoningCapabilities()', context);
assert.equal(context.settings.thinking, 'high');
assert.equal(element('clear-thinking-row').hidden, true);
assert.deepEqual(element('s-thinking').children.filter(b => !b.hidden).map(b => b.dataset.v), ['none', 'low', 'medium', 'high']);
assert.deepEqual(run('reasoningRequest({thinking:"none",clear_thinking:true})'), {reasoning_effort: 'none'});
console.log('GLM and Qwen web settings checks passed');
