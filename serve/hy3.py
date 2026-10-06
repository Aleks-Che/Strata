"""Reviewed Hy3 template and incremental reasoning/tool output parser.

Selected only by an explicit hy_v3 profile. Tools become actionable only after the
whole tool_calls group is complete and valid. Invalid groups stay literal text.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from serve.frontend import ChatTemplate, Event, TemplateRequestError, ToolCall

TEMPLATE_SHA256 = '7fc351fee674c13754656ba7f33a3ca426bfb7231039f48444360a3f3c5ecf3e'
EFFORTS = ('no_think', 'low', 'high')
THINK_START, THINK_END = '<think:opensource>', '</think:opensource>'
GROUP_START, GROUP_END = '<tool_calls:opensource>', '</tool_calls:opensource>'
CALL_START, CALL_END = '<tool_call:opensource>', '</tool_call:opensource>'
SEPARATOR = '<tool_sep:opensource>'
KEY_START, KEY_END = '<arg_key:opensource>', '</arg_key:opensource>'
VALUE_START, VALUE_END = '<arg_value:opensource>', '</arg_value:opensource>'
NAME = re.compile(r'[^\s<>\x00-\x1f]+\Z')


def _name(value):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise TemplateRequestError('Hy3 tool/argument name must be nonempty and contain no whitespace or tags')
    return value


def _constant(value):
    raise ValueError('non-JSON constant: '+value)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON member: '+key)
        result[key] = value
    return result


def _json(text):
    value = json.loads(text, parse_constant=_constant, object_pairs_hook=_pairs)
    json.dumps(value, allow_nan=False)
    return value


def _tools(tools):
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise TemplateRequestError('Hy3 tools must be a list')
    result, names = [], set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise TemplateRequestError('Hy3 tool must be an object')
        function = tool.get('function', tool)
        if tool.get('type', 'function') != 'function' or not isinstance(function, dict):
            raise TemplateRequestError('Hy3 tools require function objects')
        name = _name(function.get('name'))
        if name in names:
            raise TemplateRequestError('duplicate Hy3 tool name')
        names.add(name)
        parameters = function.get('parameters', {})
        if not isinstance(parameters, dict) or not isinstance(parameters.get('properties', {}), dict):
            raise TemplateRequestError('Hy3 parameters/properties must be schema objects')
        result.append(deepcopy(tool) if 'function' in tool else {'type': 'function', 'function': deepcopy(function)})
    return result or None


def _arguments(value):
    try:
        args = _json(value) if isinstance(value, str) else deepcopy(value)
        if not isinstance(args, dict):
            raise ValueError('object required')
        json.dumps(args, allow_nan=False)
        for key, value in args.items():
            _name(key)
            # Native Jinja writes raw strings without XML escaping. This exact
            # delimiter/follower pair cannot round-trip as a literal value.
            if isinstance(value, str) and re.search(
                    re.escape(VALUE_END)+r'\s*(?:'+re.escape(KEY_START)+'|'+re.escape(CALL_END)+')', value):
                raise ValueError('ambiguous Hy3 raw argument delimiter')
        return args
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise TemplateRequestError('Hy3 arguments must be an unambiguous finite JSON object') from error


def _objects(value, field):
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
        raise TemplateRequestError('Hy3 '+field+' must be a list of objects')
    return value


def _identifier(value):
    if not isinstance(value, str) or not value:
        raise TemplateRequestError('Hy3 tool call/result IDs must be nonempty strings')
    return value


def _content(value):
    if value is None or isinstance(value, str):
        return value or ''
    parts = _objects(value, 'content')
    if any(p.get('type') not in ('text', 'input_text') or not isinstance(p.get('text'), str) for p in parts):
        raise TemplateRequestError('Hy3 supports text content only')
    return ''.join(p['text'] for p in parts)


def _options(req, api):
    if not isinstance(req, dict):
        raise TemplateRequestError('Hy3 request must be an object')
    stop_sequences(req)
    if 'enable_thinking' in req or 'clear_thinking' in req:
        raise TemplateRequestError('Hy3 uses reasoning_effort and preserved_thinking')
    if req.get('reasoning_budget_tokens') not in (None, 0):
        raise TemplateRequestError('Hy3 hard reasoning budgets are not supported')
    result = {}
    thinking = None
    if api == 'openai':
        reasoning = req.get('reasoning', {})
        if not isinstance(reasoning, dict):
            raise TemplateRequestError('reasoning must be an object')
        if 'effort' in reasoning:
            result['reasoning_effort'] = reasoning['effort']
        if 'reasoning_effort' in req:
            result['reasoning_effort'] = req['reasoning_effort']
    else:
        thinking = req.get('thinking')
        if thinking is not None:
            if not isinstance(thinking, dict) or thinking.get('type') not in ('disabled', 'enabled', 'adaptive'):
                raise TemplateRequestError('Hy3 thinking.type must be disabled, enabled or adaptive')
            if 'budget_tokens' in thinking:
                raise TemplateRequestError('Hy3 does not convert budget_tokens to effort')
            result['reasoning_effort'] = 'no_think' if thinking['type'] == 'disabled' else 'high'
        config = req.get('output_config', {})
        if not isinstance(config, dict):
            raise TemplateRequestError('output_config must be an object')
        if 'effort' in config:
            result['reasoning_effort'] = config['effort']
    if 'preserved_thinking' in req:
        result['preserved_thinking'] = req['preserved_thinking']
    overrides = req.get('chat_template_kwargs', {})
    if not isinstance(overrides, dict) or set(overrides)-{'reasoning_effort', 'preserved_thinking'}:
        raise TemplateRequestError('Hy3 chat_template_kwargs accepts only reasoning_effort and preserved_thinking')
    result.update(overrides)
    effort = result.get('reasoning_effort', 'no_think')
    if effort not in EFFORTS:
        raise TemplateRequestError('Hy3 reasoning_effort must be no_think, low or high')
    if thinking and (thinking['type'] == 'disabled') != (effort == 'no_think'):
        raise TemplateRequestError('Hy3 thinking.type conflicts with reasoning_effort')
    if 'preserved_thinking' in result and type(result['preserved_thinking']) is not bool:
        raise TemplateRequestError('Hy3 preserved_thinking must be boolean')
    return result


def openai_to_hy3_messages(req):
    options = _options(req, 'openai')
    messages = deepcopy(_objects(req.get('messages'), 'messages'))
    if not messages:
        raise TemplateRequestError('Hy3 requires at least one message')
    for message in messages:
        if message.get('role') == 'developer':
            message['role'] = 'system'
        if message.get('role') not in ('system', 'user', 'assistant', 'tool'):
            raise TemplateRequestError('unsupported Hy3 message role')
        message['content'] = _content(message.get('content'))
        if message['role'] == 'tool':
            _identifier(message.get('tool_call_id'))
        for call in _objects(message.get('tool_calls'), 'tool_calls'):
            if message['role'] != 'assistant' or call.get('type', 'function') != 'function':
                raise TemplateRequestError('Hy3 function calls belong to assistant messages')
            _identifier(call.get('id'))
            function = call.get('function')
            if not isinstance(function, dict):
                raise TemplateRequestError('Hy3 tool call requires a function')
            _name(function.get('name'))
            function['arguments'] = _arguments(function.get('arguments', {}))
    return messages, _tools(req.get('tools')), options


def anthropic_to_hy3_messages(req, think_unasked=True):
    # The model's no_think default is explicit, independent of Qwen's opt-in policy.
    options = _options(req, 'anthropic')
    messages = []
    if req.get('system'):
        messages.append({'role': 'system', 'content': _content(req['system'])})
    incoming = _objects(req.get('messages'), 'messages')
    if not incoming:
        raise TemplateRequestError('Hy3 requires at least one message')
    for message in incoming:
        role, content = message.get('role'), message.get('content')
        if role not in ('user', 'assistant'):
            raise TemplateRequestError('Anthropic Hy3 messages require user or assistant role')
        if content is None or isinstance(content, str):
            messages.append({'role': role, 'content': _content(content)})
            continue
        out = {'role': role, 'content': ''}

        def flush():
            nonlocal out
            if out['content'] or 'reasoning_content' in out or out.get('tool_calls'):
                messages.append(out)
            out = {'role': role, 'content': ''}

        for block in _objects(content, 'content'):
            kind = block.get('type')
            if kind == 'text':
                if out.get('tool_calls'):
                    flush()
                out['content'] += _content([block])
            elif kind == 'thinking' and role == 'assistant':
                if not isinstance(block.get('thinking'), str):
                    raise TemplateRequestError('Hy3 thinking content must be text')
                if out['content'] or out.get('tool_calls'):
                    flush()
                out['reasoning_content'] = out.get('reasoning_content', '')+block['thinking']
            elif kind == 'tool_use' and role == 'assistant':
                out.setdefault('tool_calls', []).append({'id': _identifier(block.get('id')), 'type': 'function',
                    'function': {'name': _name(block.get('name')), 'arguments': _arguments(block.get('input', {}))}})
            elif kind == 'tool_result' and role == 'user':
                flush()
                text = _content(block.get('content'))
                if block.get('is_error'):
                    text = 'Error: '+text
                messages.append({'role': 'tool', 'tool_call_id': _identifier(block.get('tool_use_id')), 'content': text})
            else:
                raise TemplateRequestError('unsupported Anthropic Hy3 content block: '+str(kind))
        flush()
    tools = [{'type': 'function', 'function': {'name': _name(t.get('name')), 'description': t.get('description', ''),
              'parameters': deepcopy(t.get('input_schema', {}))}} for t in _objects(req.get('tools'), 'tools')]
    return messages, _tools(tools), options


class Hy3Template(ChatTemplate):
    architecture = 'hy_v3'
    reasoning_capabilities = {'efforts': list(EFFORTS), 'default': 'no_think',
                              'replay_reasoning': True, 'clear_thinking': False}
    supports_reasoning_budget = False
    explicit_thinking_controls_effort = True
    normalize_openai = staticmethod(openai_to_hy3_messages)
    normalize_anthropic = staticmethod(anthropic_to_hy3_messages)

    @staticmethod
    def starts_in_reasoning(prompt):
        return prompt.endswith(THINK_START)

    @staticmethod
    def resolve_stop_ids(tokenizer):
        if tokenizer.pre != 'hunyuan-dense':
            raise ValueError('Hy3 requires hunyuan-dense tokenizer')
        controls = {'bos': '<｜hy_begin_of_sentence:opensource｜>', 'eos': '<｜hy_eos:opensource｜>',
                    'padding': '<｜hy_pad:opensource｜>', 'seperator': '<｜hy_Assistant:opensource｜>'}
        required = {**{v: 3 for v in controls.values()}, '<｜hy_User:opensource｜>': 3,
                    '<｜reasoning_mode:opensource｜>': 3,
                    **{v: 4 for v in (THINK_START, THINK_END, GROUP_START, GROUP_END, CALL_START, CALL_END,
                       SEPARATOR, KEY_START, KEY_END, VALUE_START, VALUE_END,
                       '<tool_responses:opensource>', '</tool_responses:opensource>',
                       '<tool_response:opensource>', '</tool_response:opensource>')}}
        for spelling, kind in required.items():
            token_id = tokenizer.ids.get(spelling)
            if (type(token_id) is not int or not 0 <= token_id < len(tokenizer.tokens)
                    or tokenizer.tokens[token_id] != spelling or tokenizer.token_types is None
                    or len(tokenizer.token_types) != len(tokenizer.tokens) or tokenizer.token_types[token_id] != kind):
                raise ValueError('Hy3 tokenizer missing or altered special token: '+spelling)
        for key, spelling in controls.items():
            value = tokenizer.special_ids.get('tokenizer.ggml.'+key+'_token_id')
            if type(value) is not int or value != tokenizer.ids[spelling]:
                raise ValueError('Hy3 metadata disagrees with '+key)
        eos = tokenizer.ids[controls['eos']]
        for key in ('eot', 'eom', 'eod'):
            value = tokenizer.special_ids.get('tokenizer.ggml.'+key+'_token_id')
            if value is not None and (type(value) is not int or value != eos):
                raise ValueError('unreviewed Hy3 '+key+' stop')
        return {eos}

    def __init__(self, path):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != TEMPLATE_SHA256:
            raise TemplateRequestError('unreviewed Hy3 template')
        super().__init__(path)

    def render(self, messages, tools=None, add_generation_prompt=True, **options):
        allowed = {'reasoning_effort', 'preserved_thinking', 'is_training',
                   'raw_last_assistant', 'fallback_strategy'}
        if set(options)-allowed:
            raise TemplateRequestError('unsupported Hy3 template option')
        if options.get('reasoning_effort', 'no_think') not in EFFORTS:
            raise TemplateRequestError('Hy3 reasoning_effort must be no_think, low or high')
        for key in ('preserved_thinking', 'is_training', 'raw_last_assistant'):
            if key in options and type(options[key]) is not bool:
                raise TemplateRequestError('Hy3 '+key+' must be boolean')
        if 'fallback_strategy' in options and options['fallback_strategy'] != 'reasoning_toolcall_retry':
            raise TemplateRequestError('unsupported Hy3 fallback strategy')
        if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
            raise TemplateRequestError('Hy3 messages must be a list of objects')
        messages = deepcopy(messages)
        for message in messages:
            if message.get('role') not in ('system', 'user', 'assistant', 'tool'):
                raise TemplateRequestError('unsupported Hy3 message role')
            content = message.get('content')
            if isinstance(content, list):
                if any(not isinstance(p, str) and (not isinstance(p, dict) or p.get('type') != 'text'
                       or not isinstance(p.get('text'), str)) for p in content):
                    raise TemplateRequestError('Hy3 accepts text content only')
            elif content is not None and not isinstance(content, str):
                raise TemplateRequestError('Hy3 content must be text or text blocks')
            for key in ('reasoning_content', 'reasoning'):
                if key in message and not isinstance(message[key], str):
                    raise TemplateRequestError('Hy3 reasoning must be text')
            calls = message.get('tool_calls') or []
            if not isinstance(calls, list) or (calls and message['role'] != 'assistant'):
                raise TemplateRequestError('Hy3 tool_calls belong to assistant messages')
            for call in calls:
                if not isinstance(call, dict) or call.get('type', 'function') != 'function':
                    raise TemplateRequestError('invalid Hy3 tool call')
                function = call.get('function')
                if not isinstance(function, dict):
                    raise TemplateRequestError('Hy3 tool call requires a function')
                _name(function.get('name'))
                function['arguments'] = _arguments(function.get('arguments', {}))
        return super().render(messages, _tools(tools), add_generation_prompt, **options)

    @staticmethod
    def create_output_parser(thinking, tools, sampling):
        return Hy3StopParser(thinking=thinking, tools=tools, stops=stop_sequences(sampling or {}))


def _hold(text, markers):
    return max((n for marker in markers for n in range(1, min(len(text), len(marker)-1)+1)
                if text.endswith(marker[:n])), default=0)


def _is_string(schema):
    declared = schema.get('type') if isinstance(schema, dict) else None
    return declared == 'string' or isinstance(declared, list) and 'string' in declared


def _value_end(text, start, literal=False):
    """Find the native delimiter followed by another key or the call closer.

    Tags elsewhere in raw values are literal. JSON object/array quoted strings
    are skipped so a structural-looking sequence inside JSON remains data.
    """
    structured = not literal and text[start:].lstrip().startswith(('{', '['))
    quoted = escaped = False
    i = start
    while i < len(text):
        char = text[i]
        if structured:
            if quoted:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    quoted = False
                i += 1
                continue
            if char == '"':
                quoted = True
        if text.startswith(VALUE_END, i):
            rest = text[i+len(VALUE_END):].lstrip()
            if rest.startswith((KEY_START, CALL_END)):
                return i
        i += 1
    return -1


def _group_end(text, schemas):
    pos, name = 0, None
    while True:
        markers = [(text.find(tag, pos), tag) for tag in (GROUP_END, CALL_START, VALUE_START)
                   if text.find(tag, pos) >= 0]
        if not markers:
            return -1
        at, tag = min(markers)
        if tag == GROUP_END:
            return at
        if tag == CALL_START:
            end = text.find(SEPARATOR, at+len(CALL_START))
            if end < 0:
                return -1
            name, pos = text[at+len(CALL_START):end].strip(), end+len(SEPARATOR)
            continue
        key_start = text.rfind(KEY_START, pos, at)
        key_end = text.find(KEY_END, key_start+len(KEY_START)) if key_start >= 0 else -1
        key = text[key_start+len(KEY_START):key_end] if key_end >= 0 else None
        schema = schemas.get(name, {}).get('parameters', {}).get('properties', {}).get(key)
        end = _value_end(text, at+len(VALUE_START), _is_string(schema))
        if end < 0:
            return -1
        pos = end+len(VALUE_END)


def _value(text, schema):
    declared = schema.get('type') if isinstance(schema, dict) else None
    types = declared if isinstance(declared, list) else [declared] if declared is not None else []
    if 'string' in types:
        return text  # Raw native strings, including whitespace and JSON-looking text.
    try:
        value = _json(text)
    except (ValueError, RecursionError):
        if types:
            raise ValueError('Hy3 typed argument is not valid JSON')
        return text
    matches = {'null': value is None, 'boolean': type(value) is bool,
               'integer': type(value) is int or (type(value) is float and value.is_integer()),
               'number': type(value) in (int, float), 'array': isinstance(value, list), 'object': isinstance(value, dict)}
    if types and not any(matches.get(t, False) for t in types):
        raise ValueError('Hy3 argument type differs from schema')
    return value


def _parse_group(body, schemas):
    pos, calls = 0, []

    def skip():
        nonlocal pos
        while body[pos:pos+1].isspace():
            pos += 1

    def expect(tag):
        nonlocal pos
        if not body.startswith(tag, pos):
            raise ValueError('invalid Hy3 tool framing')
        pos += len(tag)

    def until(tag):
        nonlocal pos
        end = body.find(tag, pos)
        if end < 0:
            raise ValueError('missing Hy3 delimiter')
        value, pos = body[pos:end], end+len(tag)
        return value

    skip()
    while pos < len(body):
        expect(CALL_START)
        name = _name(until(SEPARATOR).strip())
        if name not in schemas:
            raise ValueError('undeclared Hy3 tool')
        props = schemas[name].get('parameters', {}).get('properties', {})
        args = {}
        skip()
        while not body.startswith(CALL_END, pos):
            expect(KEY_START)
            key = _name(until(KEY_END))
            if key in args:
                raise ValueError('duplicate Hy3 argument')
            skip()
            expect(VALUE_START)
            end = _value_end(body, pos, _is_string(props.get(key)))
            if end < 0:
                raise ValueError('missing Hy3 value delimiter')
            args[key], pos = _value(body[pos:end], props.get(key)), end+len(VALUE_END)
            skip()
        expect(CALL_END)
        calls.append(ToolCall(name, args))
        skip()
    if not calls:
        raise ValueError('empty Hy3 tool_calls group')
    return calls


class Hy3OutputParser:
    """Atomic validated tool groups; text/reasoning stream before their closer.

    Feed decoded str deltas (UTF-8 byte joining belongs to the detokenizer).
    stream_tools is accepted for Service compatibility, but no partial tools
    are emitted. A group exceeding max_group_chars becomes literal output and
    disables further control parsing for that response.
    """
    def __init__(self, thinking=False, tools=None, stream_tools=False, max_group_chars=1024*1024):
        if type(max_group_chars) is not int or max_group_chars < len(GROUP_START):
            raise ValueError('invalid Hy3 group buffer limit')
        self.state = 'reasoning' if thinking else 'content'
        self.buf, self.initial, self.closed = '', True, False
        self.max_group_chars = max_group_chars
        self.schemas = {t['function']['name']: t['function'] for t in _tools(tools) or []}

    def feed(self, delta):
        if self.closed:
            raise ValueError('Hy3 parser already finished')
        if not isinstance(delta, str):
            raise TypeError('Hy3 parser expects decoded text')
        # Bound retained group data even when the caller supplies a large delta.
        if len(delta) > 4096:
            events = []
            for i in range(0, len(delta), 4096):
                events.extend(Hy3OutputParser.feed(self, delta[i:i+4096]))
            return events
        self.buf += delta
        out = []
        if self.initial:
            if THINK_START.startswith(self.buf) and self.buf != THINK_START:
                return out
            if self.buf.startswith(THINK_START):
                self.buf = self.buf[len(THINK_START):]
                self.state = 'reasoning'
            self.initial = False
        while True:
            if self.state == 'literal':
                if self.buf:
                    out.append(Event('content', self.buf))
                    self.buf = ''
                return out
            if self.state == 'group':
                at = _group_end(self.buf, self.schemas)
                if len(GROUP_START)+(at+len(GROUP_END) if at >= 0 else len(self.buf)) > self.max_group_chars:
                    self.buf, self.state = GROUP_START+self.buf, 'literal'
                    continue
                if at < 0:
                    return out
                body, self.buf = self.buf[:at], self.buf[at+len(GROUP_END):]
                try:
                    calls = _parse_group(body, self.schemas)
                    out.extend(Event('tool_call', call=c) for c in calls)
                except (ValueError, TypeError, OverflowError, RecursionError):
                    out.append(Event('content', GROUP_START+body+GROUP_END))
                self.state = 'content'
                continue
            markers = (THINK_END,) if self.state == 'reasoning' else (THINK_START, GROUP_START)
            positions = [(self.buf.find(tag), tag) for tag in markers if tag in self.buf]
            if positions:
                at, tag = min(positions)
                if at:
                    out.append(Event(self.state, self.buf[:at]))
                self.buf = self.buf[at+len(tag):]
                self.state = {THINK_END: 'content', THINK_START: 'reasoning', GROUP_START: 'group'}[tag]
                continue
            end = len(self.buf)-_hold(self.buf, markers)
            if end:
                out.append(Event(self.state, self.buf[:end]))
                self.buf = self.buf[end:]
            return out

    def finish(self):
        if self.closed:
            return []
        text = (GROUP_START if self.state == 'group' else '')+self.buf
        kind = 'reasoning' if self.state == 'reasoning' else 'content'
        self.buf, self.closed = '', True
        return [Event(kind, text)] if text else []


def stop_sequences(request):
    if 'stop' in request and 'stop_sequences' in request:
        raise TemplateRequestError('use only one of stop or stop_sequences')
    stops = request.get('stop', request.get('stop_sequences'))
    if stops is None:
        return []
    stops = [stops] if isinstance(stops, str) else stops
    if not isinstance(stops, list) or any(not isinstance(s, str) or not s for s in stops):
        raise TemplateRequestError('Hy3 stop sequences must be nonempty strings')
    return stops


class Hy3StopParser(Hy3OutputParser):
    """Stop on raw generated text before interpreting tool calls or reasoning."""
    def __init__(self, *, stops, **kwargs):
        super().__init__(**kwargs)
        self.stops = stop_sequences({'stop': stops})
        self.pending, self.stop_sequence = '', None

    def feed(self, delta):
        if self.stop_sequence is not None:
            return []
        self.pending += delta
        # Earliest completed sequence, with list order breaking simultaneous
        # completions. This is invariant to input fragmentation and overlaps.
        matches = [(self.pending.find(s)+len(s), i, self.pending.find(s), s)
                   for i, s in enumerate(self.stops) if s in self.pending]
        if matches:
            _, _, at, self.stop_sequence = min(matches)
            text, self.pending = self.pending[:at], ''
        else:
            at = len(self.pending)-_hold(self.pending, self.stops)
            text, self.pending = self.pending[:at], self.pending[at:]
        return super().feed(text)

    def finish(self):
        if self.closed:
            return []
        events = super().feed(self.pending)
        self.pending = ''
        return events+super().finish()


Hy3Template.output_parser = Hy3OutputParser
