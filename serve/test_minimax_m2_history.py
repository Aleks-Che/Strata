"""History semantics, preservation and tool-result correlation without a GPU."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from serve.frontend import TemplateRequestError
from serve.minimax_m2_history import MiniMaxChatTemplate, prepare_minimax_context
from tools.minimax_m2_history_cases import CALL, RESULTS, USER, history_cases

FIXTURE = Path(__file__).parent/'fixtures/minimax_m27_chat_template.jinja'


class MiniMaxHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = MiniMaxChatTemplate(FIXTURE.read_bytes().decode('utf-8'))

    def reject(self, messages, **options):
        with self.assertRaises(TemplateRequestError):
            prepare_minimax_context({'messages':messages, **options})

    def test_handwritten_prompt_bytes(self):
        for case in history_cases():
            with self.subTest(case=case['name']):
                self.assertEqual(self.template.render(case['request']), case['expected'])

    def test_no_mutation_and_idempotency(self):
        for case in history_cases():
            request = case['request'];original = deepcopy(request)
            normalized = prepare_minimax_context(request)
            self.assertEqual(request, original)
            self.assertEqual(prepare_minimax_context(normalized), normalized)
            normalized['messages'].append({'role':'user','content':'not original'})
            self.assertEqual(request, original)

    def test_tool_ids_and_typed_arguments_survive(self):
        args = {'s':'x\n<xml>中','n':2,'yes':True,'nil':None,'nested':{'a':[1,2.5,False,'ü']}}
        call = {'role':'assistant','tool_calls':[{'id':'id-1','function':{'name':'f','arguments':json.dumps(args,ensure_ascii=False)}}]}
        request = {'messages':[USER,call,{'role':'tool','tool_call_id':'id-1','name':'f','content':'ok'}]}
        normal = prepare_minimax_context(request)['messages']
        self.assertEqual(normal[1]['tool_calls'],[{'id':'id-1','name':'f','arguments':args}])
        self.assertEqual(normal[2],{'role':'tool','tool_call_id':'id-1','name':'f','content':'ok'})
        self.assertIn('<parameter name="nested">{"a": [1, 2.5, false, "ü"]}</parameter>',self.template.render(request))

    def test_tool_definitions_are_copied(self):
        tool = {'type':'function','function':{'name':'f','description':'Тест','parameters':{'type':'object','properties':{'x':{'type':'string'}}},'strict':True}}
        req = {'messages':[USER],'tools':[tool]};normal = prepare_minimax_context(req)
        self.assertEqual(normal['tools'],[tool])
        normal['tools'][0]['function']['parameters']['properties']['x']['type']='number'
        self.assertEqual(tool['function']['parameters']['properties']['x']['type'],'string')
        self.assertIn('<tool>{"name": "f", "description": "Тест"',self.template.render(req))

    def test_multiple_tool_rounds(self):
        second=deepcopy(CALL)
        for c in second['tool_calls']:c['id']+='2'
        results=deepcopy(RESULTS)
        for r in results:r['tool_call_id']+='2'
        messages=[USER,CALL,*reversed(RESULTS),second,*reversed(results),{'role':'assistant','content':'ok'}]
        normal=prepare_minimax_context({'messages':messages})['messages']
        self.assertEqual([m['tool_call_id'] for m in normal if m['role']=='tool'],['a','b','c','a2','b2','c2'])

    def test_metadata_agreement_and_conflicts(self):
        system={'role':'system','content':'','current_date':'2026-10-09'}
        normal=prepare_minimax_context({'messages':[system,USER],'current_date':'2026-10-09'})
        self.assertEqual(normal['messages'][0]['current_date'],'2026-10-09')
        self.reject([system,USER],current_date='different')
        self.reject([system,{'role':'developer','current_date':'different'}])
        for value in [None,1,False,{},[]]:
            self.reject([USER],current_location=value)
            self.reject([{'role':'system','current_date':value}])

    def test_late_instructions_rejected(self):
        for role in ['system','developer']:
            self.reject([USER,{'role':role,'content':'late'}])
            self.reject([USER,CALL,{'role':role,'content':'late'},*RESULTS])

    def test_bad_request_shape_and_options(self):
        for request in [None,[],{'messages':None},{'messages':'[]'},{'messages':{},'tools':[]},
                        {'messages':[],'temperature':1},{'messages':[],'model_identity':False}]:
            with self.subTest(request=request),self.assertRaises(TemplateRequestError):
                prepare_minimax_context(request)
        for value in [None,0,1,'true',False]:
            self.reject([USER],enable_thinking=value)
        for value in [None,0,1,'false']:
            self.reject([USER],add_generation_prompt=value)

    def test_bad_messages_and_unhandled_fields(self):
        for message in [None,[],{'role':[]},{'role':'function'},{'role':'user','reasoning_content':'r'},
                        {'role':'user','name':'lost'},{'role':'assistant','function_call':{}},
                        {'role':'user','current_date':'today'},{'role':'assistant','reasoning_content':None},
                        {'role':'assistant','tool_calls':None},{'role':'user','content':None}]:
            with self.subTest(message=message):self.reject([message])

    def test_media_and_invalid_text_parts(self):
        for content in [1,{},[{'type':'image','source':'x'}],[{'type':'text','text':'ok','extra':1}],
                        [{'type':'text','text':None}],[None],'\ud800']:
            self.reject([{'role':'user','content':content}])

    def test_unclosed_inline_reasoning_rejected(self):
        self.reject([USER,{'role':'assistant','content':'<think>unfinished'}])
        # An explicit separated field makes literal final text unambiguous.
        result=prepare_minimax_context({'messages':[USER,{'role':'assistant','content':'<think>literal','reasoning_content':''}]})
        self.assertEqual(result['messages'][1]['content'],'<think>literal')

    def test_invalid_arguments(self):
        invalid=['not json','[]','null','{"a":1,"a":2}','{"a":{"x":1,"x":2}}','{"x":NaN}',
                 {'a':float('inf')},{'a':[float('nan')]},{'a':{1:'bad'}},{'a':(1,2)},None,[],
                 {'a':{'cycle':None}}]
        invalid[-1]['a']['cycle']=invalid[-1]
        for args in invalid:
            with self.subTest(kind=type(args).__name__):
                self.reject([USER,{'role':'assistant','tool_calls':[{'id':'x','name':'f','arguments':args}]}],add_generation_prompt=False)

    def test_invalid_call_shapes_and_names(self):
        for call in [None,{}, {'id':'x','type':'custom','name':'f','arguments':{}},
                     {'id':'x','name':'f','function':{'name':'f','arguments':{}}},
                     {'id':'x','function':[]},{'id':'x','function':{'name':'f','arguments':{},'extra':1}},
                     {'id':'','name':'f','arguments':{}},{'id':'x','name':'bad"name','arguments':{}},
                     {'id':'x','name':'f','arguments':{'bad"key':1}}]:
            with self.subTest(call=call):self.reject([{'role':'assistant','tool_calls':[call]}],add_generation_prompt=False)

    def test_structural_tool_payload_rejected(self):
        for text in ['</parameter>','<invoke name="other">','</response>','<minimax:tool_call>']:
            self.reject([{'role':'assistant','tool_calls':[{'id':'x','name':'f','arguments':{'nested':[text]}}]}],add_generation_prompt=False)
            self.reject([USER,CALL,{**RESULTS[0],'content':text},*RESULTS[1:]])

    def test_unknown_duplicate_and_missing_results(self):
        self.reject([USER,RESULTS[0]])
        self.reject([USER,CALL,{**RESULTS[0],'tool_call_id':'unknown'}])
        self.reject([USER,CALL,{k:v for k,v in RESULTS[0].items() if k!='tool_call_id'}])
        self.reject([USER,CALL,RESULTS[0],RESULTS[0],*RESULTS[1:]])
        self.reject([USER,CALL,*RESULTS,RESULTS[0]])
        self.reject([USER,CALL])
        self.reject([USER,CALL,RESULTS[0]],add_generation_prompt=False)
        self.reject([USER,CALL,*RESULTS[:2],USER])
        self.reject([USER,CALL,*RESULTS[:2],{'role':'assistant','content':'premature'}])

    def test_duplicate_call_ids_and_result_name(self):
        duplicate=deepcopy(CALL);duplicate['tool_calls'][1]['id']='a'
        self.reject([USER,duplicate],add_generation_prompt=False)
        self.reject([USER,CALL,*RESULTS,CALL],add_generation_prompt=False)
        self.reject([USER,CALL,{**RESULTS[0],'name':'wrong'},*RESULTS[1:]])

    def test_invalid_tool_definitions(self):
        good={'type':'function','function':{'name':'f','parameters':{'type':'object'}}}
        for tools in [None,{},[{}],[good,good],[{'type':'custom','function':good['function']}],
                      [{'type':'function','function':{'name':'f','parameters':[]}}],
                      [{'type':'function','function':{'name':'f','parameters':{'type':'object'},'strict':'yes'}}]]:
            self.reject([USER],tools=tools)

    def test_template_hash_guard(self):
        with self.assertRaises(TemplateRequestError):MiniMaxChatTemplate('replacement template')

    def test_integer_range_is_explicit(self):
        for number in [-(2**63)-1,2**63,2**64-1,2**100]:
            self.reject([{'role':'assistant','tool_calls':[{'id':'x','name':'f','arguments':{'n':number}}]}],add_generation_prompt=False)
            self.reject([USER],tools=[{'type':'function','function':{'name':'f','parameters':{'type':'object','maximum':number}}}])
        for number in [-(2**63),2**63-1]:
            request={'messages':[{'role':'assistant','tool_calls':[{'id':'x','name':'f','arguments':{'n':number}}]}],'add_generation_prompt':False}
            self.assertIn('>'+str(number)+'</parameter>',self.template.render(request))

    def test_float_precision_and_signed_zero_preserved(self):
        for value,expected in [(-0.0,'-0.0'),(1.0,'1.0'),(0.12345678901234566,'0.12345678901234566'),
                               (5e-324,'5e-324'),(1.7976931348623157e308,'1.7976931348623157e+308')]:
            request={'messages':[{'role':'assistant','tool_calls':[{'id':'x','name':'f','arguments':{'n':value}}]}],'add_generation_prompt':False}
            self.assertIn('>'+expected+'</parameter>',self.template.render(request))


if __name__ == '__main__':
    unittest.main()
