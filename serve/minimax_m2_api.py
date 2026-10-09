"""MiniMax adapters for Service; native process/profile startup is separate work.

No-thinking, hard thinking budgets, media, forced tools and interleaved blocks
that the pinned template cannot represent are rejected before generation.
"""
import codecs
from copy import deepcopy
import math
from pathlib import Path
import struct

from serve.frontend import TemplateRequestError
from serve.minimax_m2_history import MiniMaxChatTemplate, prepare_minimax_context
from serve.minimax_m2_tools import MiniMaxOutputParser, _schemas


def require(ok, message):
    if not ok:
        raise TemplateRequestError('MiniMax: '+message)


def objects(value, name):
    require(isinstance(value, list) and all(isinstance(v, dict) for v in value), name+' must be an array of objects')
    return value


def text(value, *, nullable=False):
    if value is None and nullable:
        return ''
    if isinstance(value, str):
        return value
    parts = objects(value, 'text content')
    require(all(set(p) <= {'type', 'text'} and p.get('type') in ('text', 'input_text')
                and isinstance(p.get('text'), str) for p in parts), 'only text content is supported')
    return ''.join(p['text'] for p in parts)


def stop_sequences(req):
    require(not ('stop' in req and 'stop_sequences' in req), 'use only one stop field')
    value = req.get('stop', req.get('stop_sequences'))
    value = [] if value is None else [value] if isinstance(value, str) else value
    require(isinstance(value, list) and all(isinstance(s, str) and s for s in value), 'stops must be nonempty strings')
    return list(value)


def validate_sampling(req):
    # Match the admitted native SamplingConfig, including F32 representability.
    for key in ('temperature', 'top_p'):
        if key not in req:
            continue
        value = req[key]
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        require(finite, key+' must be finite numeric')
        if key == 'temperature':
            require(value == 0 or .01 <= value <= 2, 'temperature must be 0 or 0.01..2')
        else:
            require(0 < value <= 1 and struct.unpack('f', struct.pack('f', value))[0] > 0, 'top_p must be in (0,1] in F32')
    for key, maximum in [('top_k', 200064), ('seed', 4294967294)]:
        if key in req:
            require(type(req[key]) is int and 0 <= req[key] <= maximum, key+' is outside the native integer range')


def options(req, api):
    require(isinstance(req, dict), 'request must be an object')
    common = {'model', 'messages', 'tools', 'stream', 'max_tokens', 'temperature', 'top_p', 'top_k', 'seed',
              'chat_template_kwargs', 'reasoning_budget_tokens', 'tool_choice', 'strata_mcp'}
    fields = {'openai': {'max_completion_tokens', 'stop', 'stream_options', 'reasoning', 'reasoning_effort',
                          'parallel_tool_calls', 'response_format', 'user'},
              'anthropic': {'system', 'stop_sequences', 'thinking', 'output_config', 'metadata'}}
    require(not set(req)-common-fields[api], 'unsupported request field')
    stop_sequences(req)
    validate_sampling(req)
    for key in ('max_tokens', 'max_completion_tokens'):
        if key in req:
            require(type(req[key]) is int and req[key] >= -1, key+' must be an integer >= -1')
    require(not ('max_tokens' in req and 'max_completion_tokens' in req and req['max_tokens'] != req['max_completion_tokens']),
            'conflicting output token limits')
    require('stream' not in req or type(req['stream']) is bool, 'stream must be boolean')
    require(req.get('strata_mcp', False) is False, 'server-side MCP execution is not connected for this profile')
    require('reasoning_budget_tokens' not in req or type(req['reasoning_budget_tokens']) is int
            and req['reasoning_budget_tokens'] == 0, 'hard reasoning budgets are unsupported')
    overrides = req.get('chat_template_kwargs', {})
    require(isinstance(overrides, dict) and not set(overrides)-{
        'enable_thinking', 'reasoning_effort', 'model_identity', 'current_date', 'current_location'}, 'unsupported template option')
    result = deepcopy(overrides)
    require(result.get('enable_thinking', True) is True, 'the template always reasons')
    efforts = [result.pop('reasoning_effort', 'high')]
    if api == 'openai':
        reasoning = req.get('reasoning', {})
        require(isinstance(reasoning, dict) and not set(reasoning)-{'effort'}, 'unsupported reasoning option')
        efforts += [req.get('reasoning_effort', 'high'), reasoning.get('effort', 'high')]
        require(req.get('parallel_tool_calls', True) is True, 'disabling parallel calls is unsupported')
        fmt = req.get('response_format')
        require(fmt is None or fmt == {'type': 'text'}, 'structured response_format is not connected for this profile')
        stream_options = req.get('stream_options', {})
        require(isinstance(stream_options, dict) and not set(stream_options)-{'include_usage'}
                and type(stream_options.get('include_usage', True)) is bool, 'unsupported stream_options')
    else:
        thinking = req.get('thinking', {'type': 'enabled'})
        require(thinking == {'type': 'enabled'}, 'only thinking.type=enabled without a budget is supported')
        config = req.get('output_config', {})
        require(isinstance(config, dict) and not set(config)-{'effort'}, 'unsupported output_config')
        efforts.append(config.get('effort', 'high'))
    require(all(effort == 'high' for effort in efforts), 'only the fixed high/always-on reasoning profile is supported')
    return result


def finish_request(req, messages, tools, kwargs, api):
    require(bool(messages), 'at least one message is required')
    # Validate even when tool_choice=none; malformed definitions are not ignored.
    _schemas(tools)
    choice = req.get('tool_choice', 'auto' if api == 'openai' else {'type': 'auto'})
    if api == 'anthropic':
        require(isinstance(choice, dict) and set(choice) == {'type'}, 'unsupported tool_choice fields')
        choice = choice['type']
    require(choice in ('auto', 'none'), 'forced/named tool_choice is unsupported')
    if choice == 'none':
        tools = []
    context = prepare_minimax_context({'messages': messages, 'tools': tools, **kwargs})
    # Canonical form retains call IDs, and orders results before rendering.
    return context['messages'], context['tools'], kwargs


def openai_to_minimax_messages(req):
    kwargs = options(req, 'openai')
    messages = deepcopy(objects(req.get('messages'), 'messages'))
    for message in messages:
        if 'content' in message:
            message['content'] = text(message['content'], nullable=message.get('role') == 'assistant')
    return finish_request(req, messages, req.get('tools', []), kwargs, 'openai')


def anthropic_to_minimax_messages(req, think_unasked=True):
    kwargs = options(req, 'anthropic')
    messages = []
    if 'system' in req:
        messages.append({'role': 'system', 'content': text(req['system'])})
    source = objects(req.get('messages'), 'messages')
    require(bool(source), 'at least one message is required')
    for message in source:
        require(set(message) <= {'role', 'content'} and message.get('role') in ('user', 'assistant'), 'unsupported Anthropic message')
        role, content = message['role'], message.get('content')
        if isinstance(content, str):
            messages.append({'role': role, 'content': content})
            continue
        blocks = objects(content, 'content')
        if role == 'assistant':
            out = {'role': 'assistant', 'content': '', 'reasoning_content': ''}
            phase = 0
            for block in blocks:
                kind = block.get('type')
                if kind == 'thinking':
                    require(phase == 0 and set(block) <= {'type', 'thinking', 'signature'}
                            and isinstance(block.get('thinking'), str), 'thinking after text/tools or unsupported thinking block')
                    require('signature' not in block or isinstance(block['signature'], str), 'signature must be text')
                    out['reasoning_content'] += block['thinking']
                elif kind == 'text':
                    require(phase <= 1, 'text after tool_use cannot be represented by the template')
                    phase = 1
                    out['content'] += text([block])
                elif kind == 'tool_use':
                    require(set(block) == {'type', 'id', 'name', 'input'} and isinstance(block['input'], dict), 'invalid tool_use block')
                    phase = 2
                    out.setdefault('tool_calls', []).append({'id': block['id'], 'name': block['name'], 'arguments': deepcopy(block['input'])})
                else:
                    raise TemplateRequestError('MiniMax: unsupported assistant block')
            messages.append(out)
        else:
            # Keep block order. Canonical history refuses user text before all
            # outstanding tool results, rather than silently moving that text.
            pending_text = ''
            for block in blocks:
                if block.get('type') == 'text':
                    pending_text += text([block])
                elif block.get('type') == 'tool_result':
                    require(set(block) <= {'type', 'tool_use_id', 'content', 'is_error'} and 'tool_use_id' in block,
                            'invalid tool_result block')
                    require(type(block.get('is_error', False)) is bool, 'is_error must be boolean')
                    if pending_text:
                        messages.append({'role': 'user', 'content': pending_text})
                        pending_text = ''
                    result = text(block.get('content', ''))
                    if block.get('is_error', False):
                        result = 'Error: '+result
                    messages.append({'role': 'tool', 'tool_call_id': block['tool_use_id'], 'content': result})
                else:
                    raise TemplateRequestError('MiniMax: unsupported user block')
            if pending_text or not blocks:
                messages.append({'role': 'user', 'content': pending_text})
    tools = []
    for tool in objects(req.get('tools', []), 'tools'):
        require(not set(tool)-{'name', 'description', 'input_schema'} and 'name' in tool and 'input_schema' in tool,
                'unsupported Anthropic tool definition')
        function = {'name': tool['name'], 'parameters': deepcopy(tool['input_schema'])}
        if 'description' in tool:
            function['description'] = tool['description']
        tools.append({'type': 'function', 'function': function})
    return finish_request(req, messages, tools, kwargs, 'anthropic')


class StrictMiniMaxDetokenizer:
    def __init__(self, tokenizer):
        require(callable(getattr(tokenizer, 'token_bytes', None)), 'tokenizer must expose token bytes')
        self.tokenizer = tokenizer
        self.decoder = codecs.getincrementaldecoder('utf-8')('strict')

    def push(self, token):
        return self.decoder.decode(self.tokenizer.token_bytes(token))

    def pending(self):
        return bool(self.decoder.getstate()[0])

    def finish(self):
        return self.decoder.decode(b'', final=True)


class MiniMaxAPITemplate(MiniMaxChatTemplate):
    """Service hooks only. Not selected by configured_template/startup yet."""
    architecture = 'minimax-m2'
    reasoning_capabilities = {'efforts': ['high'], 'default': 'high', 'clear_thinking': False}
    supports_reasoning_budget = False
    explicit_thinking_controls_effort = True
    normalize_openai = staticmethod(openai_to_minimax_messages)
    normalize_anthropic = staticmethod(anthropic_to_minimax_messages)
    detokenizer = StrictMiniMaxDetokenizer

    def __init__(self, path):
        super().__init__(Path(path).read_bytes().decode('utf-8'))

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        _schemas([] if tools is None else tools)
        return super().render({'messages': messages, 'tools': [] if tools is None else tools,
                               'add_generation_prompt': add_generation_prompt, **kwargs})

    @staticmethod
    def starts_in_reasoning(prompt):
        return prompt.endswith(']~b]ai\n<think>\n')

    @staticmethod
    def resolve_stop_ids(tokenizer):
        require(getattr(tokenizer, 'pre', None) == 'minimax-m2', 'wrong tokenizer pre')
        controls = {'bos': ']~!b[', 'eos': '[e~[', 'padding': '[e~[', 'unknown': ']!d~['}
        required = {**{value: 3 for value in controls.values()}, ']~b]': 3,
                    '<think>': 4, '</think>': 4, '<minimax:tool_call>': 4, '</minimax:tool_call>': 4}
        tokens, ids, kinds = tokenizer.tokens, tokenizer.ids, tokenizer.token_types
        require(kinds is not None and len(kinds) == len(tokens), 'missing token types')
        for spelling, kind in required.items():
            token = ids.get(spelling)
            require(type(token) is int and 0 <= token < len(tokens) and tokens[token] == spelling and kinds[token] == kind,
                    'missing or altered token '+spelling)
        for key, spelling in controls.items():
            token = tokenizer.special_ids.get('tokenizer.ggml.'+key+'_token_id')
            require(type(token) is int and token == ids[spelling], 'metadata disagrees with '+key)
        eos = ids['[e~[']
        for key in ('eot', 'eom', 'eod'):
            token = tokenizer.special_ids.get('tokenizer.ggml.'+key+'_token_id')
            require(token is None or type(token) is int and token == eos, 'unreviewed stop alias '+key)
        return {eos}

    @staticmethod
    def create_output_parser(thinking, tools, sampling):
        require(thinking, 'the generation prefix must start in reasoning')
        validate_sampling(sampling or {})
        return MiniMaxOutputParser(tools=tools, stops=stop_sequences(sampling or {}))
