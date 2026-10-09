"""MiniMax text history before the pinned Jinja; no API registration or tool execution.

Leading system/developer texts share the template's single system slot. Late
instructions are rejected. Tool results are correlated by ID and rendered in
call order because the embedded template has no result-ID field.
"""
from copy import deepcopy
import hashlib
import json
import math
import re

from serve.frontend import TemplateRequestError
from tools.minimax_m2_template import renderer

TEMPLATE_SHA256 = '893d908f7b5cc65fdde270dcae5ea1a99647c6a7ce572ae874a57b7160069566'
_TOOL_MARKER = re.compile(r'</?(?:minimax:tool_call|invoke|parameter|response)\b')
_MESSAGE_KEYS = {
    'system': {'role', 'content', 'current_date', 'current_location'},
    'developer': {'role', 'content', 'current_date', 'current_location'},
    'user': {'role', 'content'},
    'assistant': {'role', 'content', 'reasoning_content', 'tool_calls'},
    'tool': {'role', 'content', 'tool_call_id', 'name'},
}


def _require(ok, message):
    if not ok:
        raise TemplateRequestError('MiniMax: '+message)


def _string(value, label, *, nonempty=False):
    _require(isinstance(value, str), label+' must be text')
    _require(not nonempty or bool(value.strip()), label+' must be nonempty')
    try:
        value.encode('utf-8')
    except UnicodeError as exc:
        raise TemplateRequestError('MiniMax: '+label+' must be valid UTF-8') from exc
    return value


def _text(value, *, assistant=False):
    if value is None and assistant:
        return ''
    if isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, str):
                parts.append(_string(part, 'content'))
            else:
                _require(isinstance(part, dict) and set(part) == {'type', 'text'} and part['type'] == 'text',
                         'only text content parts are supported')
                parts.append(_string(part['text'], 'content'))
        return ''.join(parts)
    return _string(value, 'content')


def _json_value(value):
    if type(value) is int:
        _require(-(2**63) <= value < 2**63, 'JSON integers must fit signed 64-bit; encode larger values as strings')
        return
    if value is None or type(value) is bool:
        return
    if isinstance(value, str):
        _string(value, 'JSON string')
    elif type(value) is float:
        _require(math.isfinite(value), 'JSON numbers must be finite')
    elif isinstance(value, list):
        for item in value:
            _json_value(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _string(key, 'JSON object key')
            _json_value(item)
    else:
        raise TemplateRequestError('MiniMax: unsupported JSON value')


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON argument key: '+key)
        result[key] = value
    return result


def _safe_tool_text(value):
    # The unchanged template does not XML-escape arguments/results. Refuse
    # ambiguous structure until there is a separately verified escaping format.
    if isinstance(value, str):
        _require(not _TOOL_MARKER.search(value), 'tool payload contains an unescaped structural tag')
    elif isinstance(value, list):
        for item in value:
            _safe_tool_text(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _safe_tool_text(key)
            _safe_tool_text(item)


def _function_name(value):
    value = _string(value, 'function name', nonempty=True)
    _require(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]*', value) is not None,
             'function name must use letters, digits, underscore, dot or hyphen')
    return value


def _call(value):
    _require(isinstance(value, dict), 'tool call must be an object')
    nested = 'function' in value
    allowed = {'id', 'type', 'function'} if nested else {'id', 'type', 'name', 'arguments'}
    _require(not (set(value)-allowed) and value.get('type', 'function') == 'function', 'unsupported tool call fields/type')
    call_id = _string(value.get('id'), 'tool call id', nonempty=True)
    function = value['function'] if nested else value
    _require(isinstance(function, dict), 'function must be an object')
    if nested:
        _require(not (set(function)-{'name', 'arguments'}), 'unsupported function call fields')
    name = _function_name(function.get('name'))
    arguments = function.get('arguments')
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments, object_pairs_hook=_no_duplicates)
        except (ValueError, RecursionError) as exc:
            raise TemplateRequestError('MiniMax: arguments must be a JSON object without duplicate keys') from exc
    _require(isinstance(arguments, dict), 'arguments must be a JSON object')
    _json_value(arguments)
    for key in arguments:
        _require(bool(key) and not any(c in key for c in '<>&"') and all(ord(c) >= 32 for c in key),
                 'argument name cannot be represented in the template attribute')
    _safe_tool_text(arguments)
    return {'id': call_id, 'name': name, 'arguments': deepcopy(arguments)}


def _assistant(message, content):
    explicit = message.get('reasoning_content', '')
    _string(explicit, 'reasoning_content')
    if 'reasoning_content' not in message and content.startswith('<think>'):
        end = content.find('</think>', len('<think>'))
        _require(end >= 0, 'unclosed inline reasoning; provide separated history fields')
        inline, content = content[len('<think>'):end], content[end+len('</think>'):]
        explicit = inline
    # Always provide this field, even when empty: it disables the template's
    # lossy fallback split on the *last* closing tag in literal final text.
    out = {'role': 'assistant', 'content': content, 'reasoning_content': explicit}
    calls = message.get('tool_calls', [])
    _require(isinstance(calls, list), 'tool_calls must be an array')
    if calls:
        out['tool_calls'] = [_call(call) for call in calls]
    return out


def prepare_minimax_context(request):
    """Return a new canonical/template context; accept only template inputs.

    API-specific controls belong to the future adapter. IDs remain in this
    context even though the template omits them from the serialized prompt.
    """
    try:
        return _prepare(request)
    except RecursionError as exc:
        raise TemplateRequestError('MiniMax: cyclic or excessively nested history') from exc


def _prepare(request):
    allowed = {'messages', 'tools', 'add_generation_prompt', 'model_identity', 'current_date',
               'current_location', 'enable_thinking'}
    _require(isinstance(request, dict) and not (set(request)-allowed), 'unsupported template request fields')
    messages = request.get('messages')
    _require(isinstance(messages, list), 'messages must be an array')
    prompt = request.get('add_generation_prompt', True)
    _require(type(prompt) is bool, 'add_generation_prompt must be boolean')
    _require('enable_thinking' not in request or request['enable_thinking'] is True,
             'the embedded template always reasons; no no-thinking profile is implemented')
    context = {'messages': [], 'add_generation_prompt': prompt}
    if 'model_identity' in request:
        context['model_identity'] = _string(request['model_identity'], 'model_identity')
    metadata = {k: _string(request[k], k) for k in ('current_date', 'current_location') if k in request}
    instructions, conversation, seen_ids = [], [], set()
    leading = True
    pending, responses = {}, {}
    for message in messages:
        _require(isinstance(message, dict), 'message must be an object')
        role = message.get('role')
        _require(isinstance(role, str) and role in _MESSAGE_KEYS, 'unsupported message role')
        _require(not (set(message)-_MESSAGE_KEYS[role]), 'unsupported fields on '+role+' message')
        content = _text(message.get('content', ''), assistant=role == 'assistant')
        if role in ('system', 'developer'):
            _require(leading, 'system/developer messages must precede the conversation')
            instructions.append(content)
            for key in ('current_date', 'current_location'):
                if key in message:
                    value = _string(message[key], key)
                    _require(key not in metadata or metadata[key] == value, 'conflicting '+key)
                    metadata[key] = value
            continue
        leading = False
        if role == 'tool':
            call_id = _string(message.get('tool_call_id'), 'tool_call_id', nonempty=True)
            _require(call_id in pending and call_id not in responses, 'unknown, stale or duplicate tool result id')
            if 'name' in message:
                _require(message['name'] == pending[call_id]['name'], 'tool result name disagrees with call id')
            _safe_tool_text(content)
            responses[call_id] = {'role': 'tool', 'tool_call_id': call_id,
                                  'name': pending[call_id]['name'], 'content': content}
            if len(responses) == len(pending):
                conversation.extend(responses[key] for key in pending)
                pending, responses = {}, {}
            continue
        _require(not pending, 'all tool results must precede the next conversation message')
        out = _assistant(message, content) if role == 'assistant' else {'role': role, 'content': content}
        for call in out.get('tool_calls', []):
            _require(call['id'] not in seen_ids, 'duplicate tool call id')
            seen_ids.add(call['id'])
            pending[call['id']] = call
        conversation.append(out)
    _require(not pending or (not prompt and not responses), 'cannot render incomplete tool results or generate before results arrive')
    last_user = max((i for i,m in enumerate(conversation) if m['role'] == 'user'), default=-1)
    for i, message in enumerate(conversation):
        if message['role'] == 'assistant' and i < last_user:
            message['reasoning_content'] = ''
    if instructions or metadata:
        context['messages'].append({'role': 'system', 'content': '\n\n'.join(s for s in instructions if s), **metadata})
    context['messages'].extend(conversation)
    tools = request.get('tools', [])
    _require(isinstance(tools, list), 'tools must be an array')
    tool_names = set()
    for tool in tools:
        _require(isinstance(tool, dict) and set(tool) == {'type', 'function'} and tool['type'] == 'function',
                 'tools must contain function definitions')
        function = tool['function']
        _require(isinstance(function, dict) and not (set(function)-{'name','description','parameters','strict'}),
                 'unsupported function definition fields')
        name = _function_name(function.get('name'))
        _require(name not in tool_names, 'duplicate tool definition')
        tool_names.add(name)
        if 'description' in function:
            _string(function['description'], 'tool description')
        _require(isinstance(function.get('parameters'), dict) and function['parameters'].get('type') == 'object',
                 'tool parameters must be an object schema')
        _require('strict' not in function or type(function['strict']) is bool, 'tool strict must be boolean')
        _json_value(function)
    context['tools'] = deepcopy(tools)
    return context


class MiniMaxChatTemplate:
    """Render only the audited GGUF template after explicit history normalization."""
    def __init__(self, source):
        _require(isinstance(source, str) and hashlib.sha256(source.encode('utf-8')).hexdigest() == TEMPLATE_SHA256,
                 'unreviewed chat template')
        self.source = source
        self.template = renderer(source)

    def render(self, request):
        return self.template.render(**prepare_minimax_context(request))
