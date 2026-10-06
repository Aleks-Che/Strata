// Execute actual web settings functions: Hy3 choices and existing model regression.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, 'web/app.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, 'web/index.html'), 'utf8');
assert.match(html, /data-v="no_think" hidden>Off<\/button>/);
const elements = {};
function element(id) {
  return elements[id] ||= {hidden: false, value: '0', attrs: {},
    setAttribute(k,v) {this.attrs[k]=v;}, getAttribute(k) {return this.attrs[k];}};
}
element('s-thinking').children = ['none','no_think','low','medium','high','max'].map(v => ({
  ...element(v), dataset: {v}, attrs: {},
}));
const context = vm.createContext({$: element, store: {get: () => null}, health: {},
  settings: {thinking:'high',temperature:0}, sharedOn:false, projectionLoaded: () => false, outputs: () => {}});
vm.runInContext(source.slice(source.indexOf('function reasoningCapabilities()'), source.indexOf('// "Use for other apps too"')), context);
vm.runInContext(source.slice(source.indexOf('function outputs()'), source.indexOf('for (const b of $("s-thinking").children) b.onclick')), context);
for (const [caps,expected] of [
  [{efforts:['no_think','low','high'],default:'no_think',clear_thinking:false}, ['no_think','low','high']],
  [{efforts:['low','high','max'],default:'max',clear_thinking:true}, ['low','high','max']],
  [{efforts:['low','medium','high'],default:'high',clear_thinking:false}, ['low','medium','high']],
  [{efforts:['none','low','medium','high'],default:'high',clear_thinking:false}, ['none','low','medium','high']],
]) {
  context.health.reasoning=caps;
  vm.runInContext('applyReasoningCapabilities()',context);
  assert.deepEqual(element('s-thinking').children.filter(b=>!b.hidden).map(b=>b.dataset.v),expected);
  assert.equal(context.settings.thinking,caps.default);
  const request=JSON.parse(JSON.stringify(vm.runInContext('reasoningRequest(settings)',context)));
  assert.equal(request.reasoning_effort,caps.default);
  if(caps.default==='no_think') {
    assert.match(element('o-thinking').textContent,/answers right away.*default/);
    assert.equal(element('clear-thinking-row').hidden,true);
  }
}
console.log('Hy3/GLM/Step/Qwen web reasoning settings PASS');
