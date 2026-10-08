"""Text-only input normalization for the unmodified embedded MiniMax Jinja.

This is an oracle helper, not an HTTP adapter or a tool-output parser.
"""
from copy import deepcopy
import json
from jinja2.sandbox import ImmutableSandboxedEnvironment


def text_context(original):
    if not isinstance(original, dict) or not isinstance(original.get('messages'), list):
        raise ValueError('messages must be an array')
    context = deepcopy(original)
    for key in ('enable_thinking', 'add_generation_prompt'):
        if key in context and type(context[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    pending_tools = False
    for i, message in enumerate(context['messages']):
        if not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('invalid message role')
        role = message['role']
        if role == 'system' and i != 0:
            raise ValueError('MiniMax template accepts a system message only at the start')
        content = message.get('content', '')
        if content is None and role == 'assistant':
            content = ''
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and set(part) == {'type', 'text'} and part.get('type') == 'text' and isinstance(part['text'], str):
                    parts.append(part['text'])
                else:
                    raise ValueError('MiniMax GGUF admits text parts only')
            content = ''.join(parts)
        if not isinstance(content, str):
            raise ValueError('content must be text or text parts')
        message['content'] = content
        if 'reasoning_content' in message and not isinstance(message['reasoning_content'], str):
            raise ValueError('reasoning_content must be text')
        calls = message.get('tool_calls', [])
        if not isinstance(calls, list) or (calls and role != 'assistant'):
            raise ValueError('tool_calls must be an assistant array')
        normalized = []
        for call in calls:
            if not isinstance(call, dict) or call.get('type', 'function') != 'function':
                raise ValueError('only function calls are admitted')
            function = call.get('function', call)
            if not isinstance(function, dict) or not isinstance(function.get('name'), str) or not function['name']:
                raise ValueError('missing tool name')
            args = function.get('arguments')
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict) or any(not isinstance(k, str) for k in args):
                raise ValueError('tool arguments must be a JSON object')
            # The template reads both tool_call.function and the last flat
            # tool_call.name. Canonical flat calls avoid undefined-name behavior.
            normalized.append({'name': function['name'], 'arguments': args})
        if calls:
            message['tool_calls'] = normalized
        if role == 'tool':
            if not pending_tools:
                raise ValueError('tool response without a preceding assistant call')
        else:
            pending_tools = bool(calls)
    return context


def renderer(source):
    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                       extensions=['jinja2.ext.loopcontrols'])
    env.filters['tojson'] = lambda value, **kw: json.dumps(value, ensure_ascii=kw.pop('ensure_ascii', False), **kw)
    env.globals['raise_exception'] = raise_exception
    return env.from_string(source)
