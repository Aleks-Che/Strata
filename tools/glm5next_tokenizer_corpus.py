"""Shared deterministic inputs for GLM tokenizer fixtures and oracle comparisons."""
CORPUS = [
    '', 'Hello, world!', 'Привет, мир! Ёжик и съёмка.', '你好，世界！中文123456。',
    "I'M we're can't she'd I've", '1234567890 00012345 ١٢٣٤٥٦٧ １２３４５６７',
    'def f(x):\n\treturn x + 123456  # comment\r\n',
    '😀🚀👩\u200d💻🇷🇺', 'e\u0301clair a\u0301_b', '\x00\x01\x7f control',
    '  leading\r\n\ntrailing  ', '\u00a0non-breaking\u00a0space\u2028\u2029',
    '[gMASK]<sop><|system|>Rules<|user|>Hi<|assistant|><think>',
    '</think><tool_call>echo<arg_key>text</arg_key><arg_value>A</arg_value></tool_call>',
    '<|observation|><tool_response>A</tool_response><|endoftext|>',
    '<tool_call><tool_call> <|assis [gMASK', '中' * 90, 'a\u0301' * 90,
]


def cases(template):
    """Plain inputs and actual rendered prompts; IDs are never generated here."""
    result = [(f'plain/{i:02d}', text) for i, text in enumerate(CORPUS)]
    tools = [{'type': 'function', 'function': {
        'name': 'echo', 'description': 'Echo text', 'parameters': {
            'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']}}}]
    user = {'role': 'user', 'content': 'Привет, compare 123456 with 你好 🌍'}
    multi_turn = [user, {'role': 'assistant', 'reasoning_content': 'Old thought', 'content': 'Earlier answer'},
                  {'role': 'user', 'content': 'Continue'}]
    tool_turn = [user, {'role': 'assistant', 'content': '', 'reasoning_content': 'Need two results',
                       'tool_calls': [{'id': name, 'type': 'function', 'function': {
                           'name': 'echo', 'arguments': {'text': text}}} for name, text in (('a', 'А'), ('b', '中'))]},
                 {'role': 'tool', 'tool_call_id': 'b', 'content': '中'},
                 {'role': 'tool', 'tool_call_id': 'a', 'content': 'А'}]
    next_turn = tool_turn + [{'role': 'assistant', 'reasoning_content': 'Both results', 'content': 'Done'},
                             {'role': 'user', 'content': 'Next question'}]
    for name, history in (('multi-turn', multi_turn), ('tool-turn', tool_turn), ('after-tools', next_turn)):
        for effort in ('low', 'high', 'max'):
            for clear in (False, True):
                result.append((f'prompt/{name}/{effort}/clear={int(clear)}', template.render(
                    history, tools=tools, reasoning_effort=effort, clear_thinking=clear)))
    return result
