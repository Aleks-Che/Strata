"""Hand-written MiniMax prompt expectations, independent of the normalizer."""
from copy import deepcopy
from itertools import permutations
import math
import random
import struct

DEFAULT = ']~!b[]~b]system\nYou are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.[e~[\n'
GENERATION = ']~b]ai\n<think>\n'
USER = {'role': 'user', 'content': 'Q'}
USER_TEXT = ']~b]user\nQ[e~[\n'
CALL = {'role': 'assistant', 'content': None, 'tool_calls': [
    {'id': 'a', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"city":"Уфа"}'}},
    {'id': 'b', 'name': 'lookup', 'arguments': {'city': '北京'}},
    {'id': 'c', 'name': 'lookup', 'arguments': {'city': 'Paris'}},
]}
CALL_TEXT = (']~b]ai\n\n<minimax:tool_call>\n'
    '<invoke name="lookup">\n<parameter name="city">Уфа</parameter>\n</invoke>\n'
    '<invoke name="lookup">\n<parameter name="city">北京</parameter>\n</invoke>\n'
    '<invoke name="lookup">\n<parameter name="city">Paris</parameter>\n</invoke>\n'
    '</minimax:tool_call>[e~[\n')
RESULTS = [{'role': 'tool', 'tool_call_id': k, 'content': value} for k,value in [('a','A'),('b','B'),('c','C')]]
RESULT_TEXT = ']~b]tool\n<response>A</response>\n<response>B</response>\n<response>C</response>[e~[\n'


def history_cases():
    cases = []
    def case(name, messages, expected, **options):
        cases.append({'name': name, 'request': {'messages': messages, **options}, 'expected': expected})
    case('default-user', [USER], DEFAULT+USER_TEXT+GENERATION)
    case('empty-transcript', [], DEFAULT, add_generation_prompt=False)
    case('empty-instructions-default', [{'role':'system','content':''},{'role':'developer','content':[]}], DEFAULT+GENERATION)
    case('merged-leading-instructions', [
        {'role':'system','content':'Первая'}, {'role':'developer','content':[{'type':'text','text':'第二'},'!']},
        {'role':'system','content':'  Third\n','current_date':'2026-10-09','current_location':'Уфа'}, USER],
        ']~!b[]~b]system\nПервая\n\n第二!\n\n  Third\n\nCurrent date: 2026-10-09\nCurrent location: Уфа[e~[\n'+USER_TEXT+GENERATION)
    case('metadata-with-identity', [USER], ']~!b[]~b]system\nLocal\nCurrent date: today\nCurrent location: home[e~[\n'+USER_TEXT+GENERATION,
         model_identity='Local', current_date='today', current_location='home')
    case('system-overrides-identity', [{'role':'system','content':'Rules'}, USER], ']~!b[]~b]system\nRules[e~[\n'+USER_TEXT+GENERATION,model_identity='Unused')
    case('explicit-reasoning-whitespace', [USER,{'role':'assistant','reasoning_content':' R\n','content':'\n A\n'}],
         DEFAULT+USER_TEXT+']~b]ai\n<think>\n R\n\n</think>\n\n\n A\n[e~[\n',add_generation_prompt=False)
    case('old-reasoning-cleared', [USER,{'role':'assistant','reasoning_content':'OLD','content':'A'},USER,
         {'role':'assistant','reasoning_content':'NEW','content':'B'}],
         DEFAULT+USER_TEXT+']~b]ai\nA[e~[\n'+USER_TEXT+']~b]ai\n<think>\nNEW\n</think>\n\nB[e~[\n',add_generation_prompt=False)
    case('inline-first-close-only', [USER,{'role':'assistant','content':'<think>R</think>A </think> B'}],
         DEFAULT+USER_TEXT+']~b]ai\n<think>\nR\n</think>\n\nA </think> B[e~[\n',add_generation_prompt=False)
    case('old-inline-final-preserved', [USER,{'role':'assistant','content':'<think>OLD</think>A </think> B'},USER],
         DEFAULT+USER_TEXT+']~b]ai\nA </think> B[e~[\n'+USER_TEXT+GENERATION)
    case('literal-close-without-open', [USER,{'role':'assistant','content':'A </think> B'}],
         DEFAULT+USER_TEXT+']~b]ai\nA </think> B[e~[\n',add_generation_prompt=False)
    case('explicit-field-keeps-literal-open', [USER,{'role':'assistant','reasoning_content':'R','content':'<think>literal</think>'}],
         DEFAULT+USER_TEXT+']~b]ai\n<think>\nR\n</think>\n\n<think>literal</think>[e~[\n',add_generation_prompt=False)
    case('inline-whitespace-not-stripped', [USER,{'role':'assistant','content':'<think>\nR\n</think>\nA\n'}],
         DEFAULT+USER_TEXT+']~b]ai\n<think>\n\nR\n\n</think>\n\n\nA\n[e~[\n',add_generation_prompt=False)
    case('unicode-text-parts', [{'role':'user','content':['Привет ',{'type':'text','text':'中文 🧑🏽‍💻\r\n'}]}],
         DEFAULT+']~b]user\nПривет 中文 🧑🏽‍💻\r\n[e~[\n'+GENERATION,enable_thinking=True)
    case('null-assistant-content', [USER,{'role':'assistant','content':None}],
         DEFAULT+USER_TEXT+']~b]ai\n[e~[\n',add_generation_prompt=False)
    for index, responses in enumerate(permutations(RESULTS)):
        case(f'tool-result-order-{index}',[USER,CALL,*responses],DEFAULT+USER_TEXT+CALL_TEXT+RESULT_TEXT+GENERATION)
    case('unanswered-calls-transcript-only',[USER,CALL],DEFAULT+USER_TEXT+CALL_TEXT,add_generation_prompt=False)
    case('reasoning-after-tool-results',[USER,{**CALL,'reasoning_content':'Plan'},*RESULTS,{'role':'assistant','reasoning_content':'Done','content':'Answer'}],
         DEFAULT+USER_TEXT+']~b]ai\n<think>\nPlan\n</think>\n\n'+CALL_TEXT[len(']~b]ai\n'):]+RESULT_TEXT+
         ']~b]ai\n<think>\nDone\n</think>\n\nAnswer[e~[\n',add_generation_prompt=False)
    return deepcopy(cases)


def numeric_cases():
    # Fixed edge values plus reproducible IEEE-754 bit patterns. Exercise the
    # native JSON patch in scalars, arrays, objects and tool-schema defaults.
    values = [-(2**63),2**63-1,1e-8,1e-5,1e16,1e20,-0.0,0.12345678901234566,
              1.0,0.0,5e-324,1.7976931348623157e308]
    rng = random.Random(270019)
    for _ in range(256):
        value = struct.unpack('<d',rng.getrandbits(64).to_bytes(8,'little'))[0]
        if math.isfinite(value):values.append(value)
    cases = []
    for i,value in enumerate(values):
        cases.append({'name':f'numeric-{i}','request':{'messages':[USER,{'role':'assistant','tool_calls':[{
            'id':'n','name':'f','arguments':{'scalar':value,'array':[value],'object':{'v':value}}}]}],
            'tools':[{'type':'function','function':{'name':'f','parameters':{
                'type':'object','properties':{'x':{'type':'number','default':value}}}}}],
            'add_generation_prompt':False}})
    return cases
