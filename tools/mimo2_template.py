"""Text-only admission and Python oracle for the unmodified embedded MiMo Jinja.

This is not a serving adapter; native oracle comparison happens before HTTP work.
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
    for message in context['messages']:
        if not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('invalid message role')
        content = message.get('content', '')
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError('content must be text or text parts')
        for part in content:
            if isinstance(part, str):
                continue
            if not isinstance(part, dict) or set(part) != {'type', 'text'} or part.get('type') != 'text' or not isinstance(part.get('text'), str):
                raise ValueError('MiMo GGUF supports text parts only; no image/audio/video companions')
    return context


def renderer(source):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                       extensions=['jinja2.ext.loopcontrols'])
    env.filters['tojson'] = lambda value, **kw: json.dumps(value, ensure_ascii=kw.pop('ensure_ascii', False), **kw)
    return env.from_string(source)
