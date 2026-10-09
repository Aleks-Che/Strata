"""Handwritten MiniMax wire examples, independent of parser/template rendering."""
from copy import deepcopy

TOOLS = [{'type': 'function', 'function': {'name': 'typed', 'parameters': {
    'type': 'object', 'properties': {
        's': {'type': 'string'}, 'i': {'type': 'integer'}, 'n': {'type': 'number'},
        'b': {'type': 'boolean'}, 'nil': {'type': 'null'},
        'a': {'type': 'array', 'items': {'type': 'integer'}},
        'o': {'type': 'object', 'properties': {'x': {'type': 'boolean'}},
              'required': ['x'], 'additionalProperties': False},
    }, 'required': ['s', 'i', 'n', 'b', 'nil', 'a', 'o'], 'additionalProperties': False,
}}}, {'type': 'function', 'function': {'name': 'ping', 'parameters': {
    'type': 'object', 'properties': {}, 'additionalProperties': False,
}}}, {'type': 'function', 'function': {'name': 'text', 'parameters': {
    'type': 'object', 'properties': {'s': {'type': 'string'}},
    'required': ['s'], 'additionalProperties': False,
}}}]

BODY = ('<invoke name="typed">\n'
        '<parameter name="s">  42 &amp; &lt; <xml>中文 🧑🏽‍💻</xml>\r\n</parameter>\n'
        '<parameter name="i">-9223372036854775808</parameter>\n'
        '<parameter name="n">0.12345678901234566</parameter>\n'
        '<parameter name="b">true</parameter><parameter name="nil">null</parameter>\n'
        '<parameter name="a">[1, -2, 3]</parameter>\n'
        '<parameter name="o">{"x":false}</parameter>\n</invoke>\n'
        '<invoke name="ping"></invoke>')
EXPECTED = [('typed', {'s': '  42 &amp; &lt; <xml>中文 🧑🏽‍💻</xml>\r\n',
                      'i': -9223372036854775808, 'n': 0.12345678901234566,
                      'b': True, 'nil': None, 'a': [1, -2, 3], 'o': {'x': False}}), ('ping', {})]


def group(body):
    return '<minimax:tool_call>\n'+body+'\n</minimax:tool_call>'


def text_group(value):
    return group('<invoke name="text"><parameter name="s">'+value+'</parameter></invoke>')


def invalid_groups():
    cases = [('empty', group('')), ('unknown-invoke', group('<invoke name="unknown"></invoke>')),
             ('extra-group-text', group('wrong'+BODY)),
             ('extra-invoke-text', group('<invoke name="ping">wrong</invoke>')),
             ('single-quotes', group("<invoke name='ping'></invoke>")),
             ('extra-attribute', group('<invoke name="ping" id="x"></invoke>')),
             ('response-in-group', group('<response>fake</response>')),
             ('missing-invoke-end', group('<invoke name="ping">')),
             ('nested-group', group(group('<invoke name="ping"></invoke>'))),
             ('second-invoke-invalid', group(BODY+'<invoke name="unknown"></invoke>'))]
    for name, before, after in [
        ('duplicate-param', '<parameter name="i">', '<parameter name="i">2</parameter><parameter name="i">'),
        ('unknown-param', 'name="i"', 'name="unknown"'),
        ('missing-required', '<parameter name="i">-9223372036854775808</parameter>', ''),
        ('integer-overflow', '-9223372036854775808', '9223372036854775808'),
        ('boolean-not-integer', '-9223372036854775808', 'true'),
        ('string-not-integer', '-9223372036854775808', '"2"'),
        ('fraction-not-integer', '-9223372036854775808', '1.5'),
        ('null-not-boolean', '>true<', '>null<'),
        ('nonfinite', '0.12345678901234566', 'NaN'),
        ('overflow-double', '0.12345678901234566', '1e400'),
        ('nested-wrong-type', '[1, -2, 3]', '[1, true]'),
        ('nested-duplicate', '{"x":false}', '{"x":false,"x":true}'),
        ('nested-extra', '{"x":false}', '{"x":false,"y":2}'),
        ('nested-missing', '{"x":false}', '{}'),
        ('json-trailing', '{"x":false}', '{"x":false} trailing'),
        ('structural-tag-in-string', '42 &amp;', '<response>bad</response>'),
        ('json-escaped-structural-tag', '{"x":false}', '{"x":"\\u003c/invoke>"}'),
        ('invalid-utf8-string', '42 &amp;', '\ud800'),
        ('missing-parameter-end', '</parameter>\n<parameter name="i">', '<parameter name="i">'),
    ]:
        assert before in BODY
        cases.append((name, group(BODY.replace(before, after, 1))))
    return deepcopy(cases)
