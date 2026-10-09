"""MiniMax generated tools, separate from API registration and tool execution.

The audited template emits raw strings and JSON for other argument types. It
does not define XML entity decoding. Only complete, valid groups become calls;
malformed, ambiguous or oversized groups disable tool parsing for this response.
"""
import io
import json
import re

from serve.frontend import Event, TemplateRequestError, ToolCall
from serve.minimax_m2 import MiniMaxReasoningParser
from serve.minimax_m2_history import (
    _json_value, _no_duplicates, _safe_tool_text, prepare_minimax_context,
)

GROUP_START = '<minimax:tool_call>'
GROUP_END = '</minimax:tool_call>'
INVOKE = re.compile(r'<invoke name="([A-Za-z_][A-Za-z0-9_.-]*)">')
PARAMETER = re.compile(r'<parameter name="([^<>&"\x00-\x1f]+)">')
TYPES = {'string', 'integer', 'number', 'boolean', 'null', 'object', 'array'}


def _hold(text, markers):
    return max((n for marker in markers for n in range(1, min(len(text)+1, len(marker)))
                if text.endswith(marker[:n])), default=0)


def _schemas(tools):
    tools = prepare_minimax_context({'messages': [], 'tools': [] if tools is None else tools})['tools']
    if not tools:
        return {}
    # This optional dependency is required for tools, never silently bypassed.
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError
        from referencing import Registry
        from referencing.exceptions import NoSuchResource
    except ImportError as exc:
        raise TemplateRequestError('MiniMax tools require jsonschema>=4.23,<5') from exc

    def no_remote(uri):
        raise NoSuchResource(ref=uri)

    def check_refs(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ('$ref', '$dynamicRef') and isinstance(value, str) and not value.startswith('#'):
                    raise TemplateRequestError('MiniMax tools support only local schema references')
                check_refs(value)
        elif isinstance(node, list):
            for value in node:
                check_refs(value)

    result = {}
    for tool in tools:
        function = tool['function']
        schema = function['parameters']
        if schema.get('$schema', 'https://json-schema.org/draft/2020-12/schema') != 'https://json-schema.org/draft/2020-12/schema':
            raise TemplateRequestError('MiniMax tools use JSON Schema draft 2020-12')
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise TemplateRequestError('MiniMax invalid tool schema: '+exc.message) from exc
        check_refs(schema)
        properties = schema.get('properties', {})
        for prop in properties.values():
            declared = prop.get('type') if isinstance(prop, dict) else None
            types = declared if isinstance(declared, list) else [declared]
            if not types or any(t not in TYPES for t in types):
                raise TemplateRequestError('MiniMax wire parameters need an explicit type in properties')
        result[function['name']] = (properties, Draft202012Validator(schema, registry=Registry(retrieve=no_remote)))
    return result


def _matches(value, types):
    matches = {'null': value is None, 'boolean': type(value) is bool,
               'integer': type(value) is int or (type(value) is float and value.is_integer()),
               'number': type(value) in (int, float), 'array': isinstance(value, list),
               'object': isinstance(value, dict), 'string': isinstance(value, str)}
    return any(matches[t] for t in types)


def _value(text, schema):
    declared = schema['type']
    types = declared if isinstance(declared, list) else [declared]
    if 'string' in types:
        # Native strings are raw, including quotes, entities and whitespace.
        # A union cannot distinguish e.g. the string "null" from JSON null.
        if len(types) > 1:
            try:
                candidate = json.loads(text, object_pairs_hook=_no_duplicates)
                _json_value(candidate)
            except (ValueError, RecursionError):
                candidate = text
            if not isinstance(candidate, str) and _matches(candidate, types):
                raise ValueError('ambiguous raw string / JSON union argument')
        return text
    value = json.loads(text, object_pairs_hook=_no_duplicates)
    _json_value(value)  # Finite, signed int64, valid UTF-8; also checks nested values.
    if not _matches(value, types):
        raise ValueError('argument type differs from schema')
    return value


def _parse_group(text, schemas):
    body = text[len(GROUP_START):-len(GROUP_END)]
    pos, calls = 0, []

    def skip():
        nonlocal pos
        while body[pos:pos+1].isspace():
            pos += 1

    skip()
    while pos < len(body):
        match = INVOKE.match(body, pos)
        if not match or match[1] not in schemas:
            raise ValueError('invalid or undeclared invoke')
        name, pos = match[1], match.end()
        properties, validator = schemas[name]
        arguments = {}
        skip()
        while not body.startswith('</invoke>', pos):
            match = PARAMETER.match(body, pos)
            if not match or match[1] not in properties or match[1] in arguments:
                raise ValueError('invalid, undeclared or duplicate parameter')
            key, pos = match[1], match.end()
            end = body.find('</parameter>', pos)
            if end < 0:
                raise ValueError('unclosed parameter')
            raw = body[pos:end]
            _safe_tool_text(raw)
            value = _value(raw, properties[key])
            _json_value(value)
            _safe_tool_text(value)
            arguments[key], pos = value, end+len('</parameter>')
            skip()
        pos += len('</invoke>')
        # A broken reference or any validation error leaves the whole group text.
        try:
            validator.validate(arguments)
        except Exception as exc:
            raise ValueError('arguments failed schema validation: '+type(exc).__name__) from exc
        calls.append((name, arguments))
        skip()
    if not calls:
        raise ValueError('empty tool group')
    return [ToolCall(name, arguments) for name, arguments in calls]


class MiniMaxOutputParser:
    """Reasoning first, content and atomic tool groups after the first </think>.

    Feed strictly decoded UTF-8 text. EOS is handled by token ID in the caller;
    finish also flushes truncation/cancel tails, without inventing any closer.
    Stop matching happens on raw generated text, before interpreting tags.
    Earliest completed stop wins, with list order breaking equal end positions.
    No partial tool deltas are emitted. Calls only have IDs after validation.
    """
    def __init__(self, *, tools=None, stops=None, max_group_chars=1024*1024):
        if type(max_group_chars) is not int or max_group_chars < len(GROUP_START)+len(GROUP_END):
            raise ValueError('invalid MiniMax group buffer limit')
        self.schemas = _schemas(tools)
        stops = [] if stops is None else [stops] if isinstance(stops, str) else stops
        if not isinstance(stops, list) or any(not isinstance(s, str) or not s for s in stops):
            raise ValueError('MiniMax stops must be nonempty strings')
        self.stops = tuple(stops)
        self.stop_sequence = None
        self.max_group_chars = max_group_chars
        self.reasoning = MiniMaxReasoningParser()
        self.closed = False
        self._state, self._pending, self._stop_pending = 'content', '', ''
        self._group, self._group_chars = None, 0
        self.tool_error = None

    @property
    def reasoning_complete(self):
        return self.reasoning.reasoning_complete

    @property
    def buffered_chars(self):
        return self._group_chars+len(self._pending)+len(self._stop_pending)+len(self.reasoning._pending)

    def _write(self, text):
        self._group.write(text)
        self._group_chars += len(text)

    def _take_group(self):
        text = self._group.getvalue()
        self._group.close()
        self._group, self._group_chars = None, 0
        return text

    def _content(self, delta):
        if not self.schemas:
            return [Event('content', delta)] if delta else []
        text, self._pending = self._pending+delta, ''
        out = []
        while text:
            if self._state == 'literal':
                out.append(Event('content', text))
                break
            if self._state == 'group':
                at = text.find(GROUP_END)
                count = at+len(GROUP_END) if at >= 0 else len(text)
                if self._group_chars+count > self.max_group_chars:
                    out.append(Event('content', self._take_group()+text))
                    self._state, self.tool_error = 'literal', 'tool group exceeds character limit'
                    break
                if at < 0:
                    hold = _hold(text, (GROUP_END,))
                    end = len(text)-hold
                    self._write(text[:end])
                    self._pending = text[end:]
                    break
                self._write(text[:count])
                group, text = self._take_group(), text[count:]
                try:
                    calls = _parse_group(group, self.schemas)
                except (ValueError, TypeError, OverflowError, RecursionError) as exc:
                    # Includes validator/ref errors. Never recover within a bad
                    # group and accidentally interpret nested calls as new ones.
                    self._state, self.tool_error = 'literal', type(exc).__name__+': '+str(exc).split('\n')[0]
                    out.append(Event('content', group))
                else:
                    out.extend(Event('tool_call', call=call) for call in calls)
                    self._state = 'content'
                continue
            at = text.find(GROUP_START)
            if at < 0:
                end = len(text)-_hold(text, (GROUP_START,))
                if end:
                    out.append(Event('content', text[:end]))
                self._pending = text[end:]
                break
            if at:
                out.append(Event('content', text[:at]))
            text = text[at+len(GROUP_START):]
            self._state, self._group = 'group', io.StringIO()
            self._write(GROUP_START)
        return out

    def _decoded(self, text):
        out = []
        for event in self.reasoning.feed(text):
            out.extend(self._content(event.text) if event.kind == 'content' else [event])
        return out

    def feed(self, delta):
        if self.closed:
            raise ValueError('MiniMax stream is already finished')
        if not isinstance(delta, str):
            raise TypeError('feed decoded text; use an incremental UTF-8 decoder for token bytes')
        out = []
        # Bound overshoot even for a caller supplying a very large delta.
        for start in range(0, len(delta), 4096):
            if self.stop_sequence is not None:
                break
            text = self._stop_pending+delta[start:start+4096]
            matches = [(text.find(s)+len(s), i, text.find(s), s)
                       for i, s in enumerate(self.stops) if s in text]
            if matches:
                _, _, end, self.stop_sequence = min(matches)
                text, self._stop_pending = text[:end], ''
            else:
                end = len(text)-_hold(text, self.stops)
                text, self._stop_pending = text[:end], text[end:]
            out.extend(self._decoded(text))
        return out

    def finish(self):
        if self.closed:
            return []
        out = self._decoded(self._stop_pending)
        self._stop_pending = ''
        out.extend(self.reasoning.finish())
        if self._group is not None:
            self.tool_error = 'incomplete tool group'
        tail = (self._take_group() if self._group is not None else '')+self._pending
        if tail:
            out.append(Event('content', tail))
        self._pending, self.closed = '', True
        return out
