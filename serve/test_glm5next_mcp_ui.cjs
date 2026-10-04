// Replay MCP stream events through the actual web history functions.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, 'web/app.js'), 'utf8').replace(/\r\n/g, '\n');
const functions = ['onTool', 'apiMessages', 'assistantMessages', 'reasoningCapabilities'].map(name => {
  const start = source.indexOf(`function ${name}(`);
  assert.notEqual(start, -1);
  const end = source.indexOf('\n}\n', start);
  assert.ok(end > start);
  return source.slice(start, end + 3);
}).join('\n');
const context = vm.createContext({messages: [], userText: m => m.text,
                                 health: {reasoning: {replay_reasoning: true}}});
vm.runInContext(functions, context);
const m = {role: 'assistant', text: 'Before', reasoning: 'reason1'};
const event = x => context.onTool(m, x);
event({event: 'call', id: 'a', name: 'mock__echo', arguments: {text: 'A'}, round: 1});
event({event: 'result', id: 'a', text: 'result:A', ok: true});
event({event: 'call', id: 'b', name: 'mock__echo', arguments: {text: 'B'}, round: 1});
event({event: 'result', id: 'b', text: 'error:B', ok: false});
m.text += ' After';
m.reasoning += ' reason2';
// A parser emitting tool_start must retain the same identity at call/result.
event({event: 'start', id: 'c', name: 'mock__echo'});
event({event: 'call', id: 'c', name: 'mock__echo', arguments: {text: 'C'}, round: 2});
event({event: 'result', id: 'c', text: 'result:C', ok: true});
m.text += ' Final';
m.reasoning += ' reason3';
// Skipped and unfinished calls are not fabricated into completed history.
event({event: 'result', id: 'skipped', text: 'not run', skipped: true, ok: false});
event({event: 'start', id: 'unfinished', name: 'mock__echo'});
context.messages = [{role: 'user', text: 'Question'}, m];
const history = JSON.parse(JSON.stringify(context.apiMessages()));
assert.deepEqual(history.map(x => x.role), ['user', 'assistant', 'tool', 'tool', 'assistant', 'tool', 'assistant']);
assert.deepEqual(history[1].tool_calls.map(x => x.id), ['a', 'b']);
assert.deepEqual(history.slice(2, 4).map(x => x.tool_call_id), ['a', 'b']);
assert.equal(history[4].tool_calls[0].id, 'c');
assert.equal(history[5].tool_call_id, 'c');
assert.deepEqual(history.filter(x => x.role === 'tool').map(x => x.content), ['result:A', 'error:B', 'result:C']);
assert.deepEqual(history.filter(x => x.role === 'assistant').map(x => x.content), ['Before', 'After', 'Final']);
assert.deepEqual(history.filter(x => x.role === 'assistant').map(x => x.reasoning_content), ['reason1', 'reason2', 'reason3']);
assert.deepEqual(JSON.parse(history[1].tool_calls[0].function.arguments), {text: 'A'});
assert.equal(m.tools.filter(x => x.id === 'c').length, 1);
assert.ok(!JSON.stringify(history).includes('skipped'));
assert.ok(!JSON.stringify(history).includes('unfinished'));
// Persisting and reopening a chat must keep the same IDs and arguments.
context.messages = JSON.parse(JSON.stringify(context.messages));
assert.deepEqual(JSON.parse(JSON.stringify(context.apiMessages())), history);
// Plain answers and interrupted reasoning-only answers are also replayed.
let plain = context.assistantMessages({text: 'Answer', reasoning: 'Thought'});
assert.equal(plain[0].reasoning_content, 'Thought');
assert.equal(context.assistantMessages({text: '', reasoning: 'Thought'})[0].content, '');
// A final reasoning-only continuation must not be lost after a tool result.
const reasonOnly = JSON.parse(JSON.stringify(m));
reasonOnly.text = 'Before After';
assert.equal(context.assistantMessages(reasonOnly).at(-1).reasoning_content, 'reason3');
assert.equal(context.assistantMessages(reasonOnly).at(-1).content, '');
// Legacy/bad offsets: keep calls and text but never attach thoughts to guessed rounds.
for (const offset of [undefined, -1, 1.5, '7', 100000]) {
  const old = JSON.parse(JSON.stringify(m));
  old.tools[0].rat = offset;
  assert.ok(context.assistantMessages(old).every(x => !('reasoning_content' in x)));
}
const decreasing = JSON.parse(JSON.stringify(m));
decreasing.tools[2].rat = 0;
assert.ok(context.assistantMessages(decreasing).every(x => !('reasoning_content' in x)));
// Models without this capability keep their previous prompt behavior.
context.health = {};
assert.ok(context.apiMessages().every(x => !('reasoning_content' in x)));
assert.equal(context.assistantMessages({text: '', reasoning: 'Thought'}).length, 0);
if (process.argv.includes('--history')) console.log(JSON.stringify(history));
else console.log('MCP web history IDs, reasoning and round-trip checks passed');
