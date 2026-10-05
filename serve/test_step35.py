"""Step template and parser contract; no HTTP server, GPU or large model required."""
from copy import deepcopy
import codecs
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from serve.frontend import ChatTemplate, TemplateRequestError
from serve.step35 import (StepTemplate, StepOutputParser, TEMPLATE_SHA256,
                          openai_to_step_messages, anthropic_to_step_messages)
from tools.check_step35_template import corpus

FIXTURE = Path(__file__).parent / 'fixtures/step37_chat_template.jinja'
BOS, EOS = '<｜begin▁of▁sentence｜>', '<|im_end|>'
GENERATION = '<|im_start|>assistant\n<think>\n'


def template():
    return StepTemplate(FIXTURE, bos_token=BOS, eos_token=EOS)


def signature(events):
    result = []
    for e in events:
        if e.kind == 'tool_call':
            result.append((e.kind, e.call.name, e.call.arguments))
        elif result and result[-1][0] == e.kind:
            result[-1] = (e.kind, result[-1][1]+e.text)
        else:
            result.append((e.kind, e.text))
    return result


class StepTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tpl = template()

    def test_fixture_hash_and_manual_prompt(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), TEMPLATE_SHA256)
        self.assertEqual(self.tpl.render([{'role':'user','content':'Привет, 你好!'}]),
                         BOS+'<|im_start|>user\nПривет, 你好!'+EOS+'\n'+GENERATION)
        for effort in ('low','medium','high'):
            self.assertEqual(self.tpl.render([{'role':'user','content':'x'}],reasoning_effort=effort),
                BOS+'<|im_start|>system\nReasoning: '+effort+'\n\n'+EOS+'\n<|im_start|>user\nx'+EOS+'\n'+GENERATION)

    def test_all_saved_native_template_oracle_hashes(self):
        # Fixed STEP-08 native oracle evidence, not expected strings generated
        # by this adapter. Full native comparison is also available separately.
        report=json.loads((FIXTURE.parents[2]/'docs/Step-3.7-Flash/STEP37_FLASH_ADMISSION_TEMPLATE.json').read_text(encoding='utf8'))
        expected={x['name']:x for x in report['cases']}
        for case in corpus():
            with self.subTest(name=case['name']):
                before=deepcopy(case['context'])
                rendered=self.tpl.render(**case['context'])
                self.assertEqual(hashlib.sha256(rendered.encode()).hexdigest(),expected[case['name']]['render_sha256'])
                self.assertEqual(case['context'],before)

    def test_text_blocks_keep_native_separator_and_reject_media(self):
        prompt=self.tpl.render([{'role':'user','content':[{'type':'text','text':'a'},{'type':'input_text','text':'b'}]}])
        self.assertIn('user\na b'+EOS,prompt)
        for content in ([{'type':'image','image':'x'}],[{'type':'audio','text':'x'}],{'text':3},42):
            with self.subTest(content=content),self.assertRaises((ValueError,TypeError)):
                self.tpl.render([{'role':'user','content':content}])

    def test_thinking_replay_and_tool_results_native_order(self):
        messages=[{'role':'user','content':'old'},
                  {'role':'assistant','content':'answer','reasoning_content':'old thought'},
                  {'role':'user','content':'new'},
                  {'role':'assistant','content':None,'reasoning_content':'current thought','tool_calls':[
                      {'id':'a','function':{'name':'f','arguments':'{"x":true,"y":null}'}},
                      {'id':'b','function':{'name':'g','arguments':{}}}]},
                  {'role':'tool','tool_call_id':'b','content':'B'},
                  {'role':'tool','tool_call_id':'a','content':'A'}]
        before=deepcopy(messages);rendered=self.tpl.render(messages)
        self.assertNotIn('old thought',rendered)
        self.assertIn('<think>\ncurrent thought\n</think>',rendered)
        self.assertIn('<parameter=x>\nTrue\n</parameter>',rendered)
        self.assertIn('<parameter=y>\nNone\n</parameter>',rendered)
        self.assertIn('<tool_response>B</tool_response><tool_response>A</tool_response>',rendered)
        self.assertEqual(messages,before)

    def test_bare_tools_normalize_and_fromjson_is_local(self):
        fn={'name':'f','parameters':{'type':'object'}}
        messages=[{'role':'user','content':'x'}]
        self.assertEqual(self.tpl.render(messages,[fn]),self.tpl.render(messages,[{'type':'function','function':fn}]))
        self.assertNotIn('fromjson',ChatTemplate(FIXTURE).template.environment.filters)
        self.assertIn('fromjson',self.tpl.template.environment.filters)

    def test_bad_templates_options_and_arguments_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'wrong.jinja';path.write_text('wrong')
            with self.assertRaisesRegex(TemplateRequestError,'Unreviewed'):
                StepTemplate(path,bos_token=BOS,eos_token=EOS)
        for options in ({'reasoning_effort':'xhigh'},{'reasoning_effort':'none'},{'enable_thinking':False},
                        {'clear_thinking':True},{'bos_token':'override'}):
            with self.subTest(options=options),self.assertRaises(TemplateRequestError):
                self.tpl.render([{'role':'user','content':'x'}],**options)
        for args in ('{','[]','null','123','"text"','{"x":NaN}','{"x":1e400}','{"x":1,"x":2}',{'x':float('nan')},{'bad>name':1}):
            with self.subTest(args=args),self.assertRaises(TemplateRequestError):
                self.tpl.render([{'role':'assistant','tool_calls':[{'name':'f','arguments':args}]}])


class StepRequestTests(unittest.TestCase):
    def test_openai_preserves_ids_effort_and_history(self):
        req={'messages':[{'role':'developer','content':'late'},
                         {'role':'assistant','content':None,'reasoning_content':'r','tool_calls':[
                             {'id':'a','type':'function','function':{'name':'f','arguments':'{"x":1}'}}]},
                         {'role':'tool','tool_call_id':'a','content':'ok'}],
             'reasoning':{'effort':'low'},'reasoning_effort':'medium',
             'chat_template_kwargs':{'reasoning_effort':'high'},
             'tools':[{'name':'f','parameters':{'type':'object'}}]}
        before=deepcopy(req);messages,tools,options=openai_to_step_messages(req)
        self.assertEqual(options,{'reasoning_effort':'high'})
        self.assertEqual(messages[0]['role'],'system')
        self.assertEqual(messages[1]['tool_calls'][0]['id'],'a')
        self.assertEqual(messages[1]['tool_calls'][0]['function']['arguments'],{'x':1})
        self.assertEqual(messages[2]['tool_call_id'],'a')
        self.assertEqual(tools[0]['type'],'function')
        messages[1]['tool_calls'][0]['function']['arguments']['x']=2
        tools[0]['function']['name']='changed';self.assertEqual(req,before)

    def test_no_effort_is_not_thinking_off(self):
        for normalize in (openai_to_step_messages,anthropic_to_step_messages):
            m,t,o=normalize({'messages':[{'role':'user','content':'x'}]})
            self.assertEqual(o,{})
            self.assertTrue(template().render(m,t,**o).endswith(GENERATION))
        for effort in ('low','medium','high'):
            self.assertEqual(anthropic_to_step_messages({'output_config':{'effort':effort}})[2],{'reasoning_effort':effort})

    def test_anthropic_mixed_blocks_keep_order(self):
        req={'system':[{'type':'text','text':'system'}], 'thinking':{'type':'enabled'},
             'messages':[{'role':'assistant','content':[{'type':'thinking','thinking':'r'},
                 {'type':'tool_use','id':'a','name':'f','input':{'n':2}}, {'type':'text','text':'after'}]},
                 {'role':'user','content':[{'type':'text','text':'before'},
                    {'type':'tool_result','tool_use_id':'a','content':'bad','is_error':True},
                    {'type':'text','text':'next'}]}],
             'tools':[{'name':'f','input_schema':{'type':'object'}}]}
        before=deepcopy(req);m,t,o=anthropic_to_step_messages(req,False)
        self.assertEqual([x['role'] for x in m],['system','assistant','assistant','user','tool','user'])
        self.assertEqual(m[1]['reasoning_content'],'r');self.assertEqual(m[1]['tool_calls'][0]['id'],'a')
        prompt=template().render(m,t,**o)
        self.assertLess(prompt.index('<function=f>'),prompt.index('after'))
        self.assertIn('<tool_response>Error: bad</tool_response>',prompt)
        self.assertEqual(req,before)

    def test_reject_unsupported_effort_budget_and_media(self):
        for normalize in (openai_to_step_messages,anthropic_to_step_messages):
            for req in ({'chat_template_kwargs':{'reasoning_effort':'max'}},{'chat_template_kwargs':{'reasoning_effort':None}},
                        {'chat_template_kwargs':[]},{'enable_thinking':True},{'clear_thinking':False},
                        {'reasoning_budget_tokens':128},{'messages':[{'role':'user','content':[{'type':'image','source':{}}]}]}):
                with self.subTest(req=req,normalize=normalize),self.assertRaises((ValueError,TypeError)):
                    normalize(req)
        for thinking in ({'type':'disabled'},{'type':'enabled','budget_tokens':128}):
            with self.assertRaises(TemplateRequestError):anthropic_to_step_messages({'thinking':thinking})

    def test_reject_invalid_calls_and_ids(self):
        for message in ({'role':'tool','content':'x'},
                        {'role':'user','tool_calls':[{'id':'a','name':'f'}]},
                        {'role':'assistant','tool_calls':[{'name':'f','arguments':{}}]},
                        {'role':'assistant','tool_calls':[{'id':'a','name':'f','arguments':'[]'}]},
                        {'role':'assistant','reasoning_content':42}):
            with self.subTest(message=message),self.assertRaises(TemplateRequestError):
                openai_to_step_messages({'messages':[message]})

    def test_invalid_tool_schema_and_mapping_error_result(self):
        for tool in ({'name':'f','parameters':[]},{'name':'f','parameters':{'properties':[]}},
                     {'type':'web_search','name':'search'}):
            with self.assertRaises(TemplateRequestError):openai_to_step_messages({'tools':[tool]})
        m,t,o=anthropic_to_step_messages({'messages':[{'role':'user','content':[
            {'type':'tool_result','tool_use_id':'a','content':{'text':'bad'},'is_error':True}]}]})
        self.assertIn('<tool_response>Error: bad</tool_response>',template().render(m,t,**o))


class StepParserTests(unittest.TestCase):
    checks = 0

    def parse(self,chunks,**kwargs):
        p=StepOutputParser(**kwargs);events=[]
        for chunk in chunks:events.extend(p.feed(chunk))
        events.extend(p.finish());self.assertEqual(p.finish(),[])
        self.assertTrue(all(e.kind in ('reasoning','content','tool_call') for e in events))
        type(self).checks+=1
        return signature(events)

    def fragmentation(self,text,expected,**kwargs):
        for i in range(len(text)+1):
            with self.subTest(split=i):self.assertEqual(self.parse([text[:i],text[i:]],**kwargs),expected)
        for width in (1,2,3,7,19):
            self.assertEqual(self.parse([text[i:i+width] for i in range(0,len(text),width)],**kwargs),expected)

    def test_reasoning_text_and_multiple_calls(self):
        text='Думаю 🌍</think>\nОтвет\n<tool_call>\n<function=f>\n<parameter=x>\n{"a":[1,true,null]}\n</parameter>\n</function>\n</tool_call>\n<tool_call><function=g></function></tool_call>Done'
        self.fragmentation(text,[('reasoning','Думаю 🌍'),('content','\nОтвет\n'),
            ('tool_call','f',{'x':{'a':[1,True,None]}}),('content','\n'),('tool_call','g',{}),('content','Done')],stream_tools=True)

    def test_think_markers_and_calls_inside_thought(self):
        for thinking in (True,False):
            self.fragmentation('<think>r</think>a',[('reasoning','r'),('content','a')],thinking=thinking)
            self.fragmentation('<think></think>a',[('content','a')],thinking=thinking)
        text='<tool_call><function=f></function></tool_call>'
        self.fragmentation(text+'</think>ok',[('reasoning',text),('content','ok')])
        self.fragmentation('prefix<think>r</think>end',[('content','prefix'),('reasoning','r'),('content','end')],thinking=False)

    def test_schema_types_and_python_scalars(self):
        props={'s':{'type':'string'},'b':{'type':'boolean'},'n':{'type':'null'},'i':{'type':'integer'},
               'a':{'type':'array'},'o':{'type':'object'},'r':{'type':'number'},'u':{'type':['string','null']}}
        tools=[{'name':'f','parameters':{'properties':props}}]
        values={'s':'\n  true\n','b':'True','n':'None','i':'2','a':'[false, null]','o':'{"v":1}','r':'1.25','u':'null'}
        text='<tool_call><function=f>'+''.join('<parameter='+k+'>\n'+v+'\n</parameter>\n' for k,v in values.items())+'</function></tool_call>'
        expected=[('tool_call','f',{'s':'\n  true\n','b':True,'n':None,'i':2,'a':[False,None],'o':{'v':1},'r':1.25,'u':'null'})]
        for schema in (tools,[{'type':'function','function':tools[0]}]):
            self.fragmentation(text,expected,thinking=False,tools=schema)

    def test_literal_tags_in_string(self):
        value='keep </tool_call> and <tool_call>nested</tool_call>; </parameter> is text; <think>中</think> &amp;'
        text='<tool_call><function=write><parameter=text>\n'+value+'\n</parameter></function></tool_call>'
        self.fragmentation(text,[('tool_call','write',{'text':value})],thinking=False)

    def test_malformed_calls_preserve_all_text(self):
        for text in ('<tool_call></tool_call>', '<tool_call>f</tool_call>',
                     '<tool_call><function=bad name></function></tool_call>',
                     '<tool_call><function=f><parameter=x>1</parameter><parameter=x>2</parameter></function></tool_call>',
                     '<tool_call><function=f>garbage</function></tool_call>',
                     '<tool_call><function=f><parameter=x>1</parameter></function>extra</tool_call>',
                     '<tool_call><function=f><parameter=x>missing close</function></tool_call>',
                     '<tool_call><function=f><parameter=>1</parameter></function></tool_call>'):
            self.fragmentation(text,[('content',text)],thinking=False,stream_tools=True)

    def test_bad_typed_values_never_execute(self):
        for kind,value in [('integer','true'),('integer','1.2'),('boolean','1'),('object','[]'),('array','{}'),
                           ('number','NaN'),('number','Infinity'),('number','1e400'),('object','{"x":NaN}'),
                           ('object','{"x":1,"x":2}'),('null','false')]:
            text='<tool_call><function=f><parameter=x>'+value+'</parameter></function></tool_call>'
            self.fragmentation(text,[('content',text)],thinking=False,tools=[{'name':'f','parameters':{'properties':{'x':{'type':kind}}}}])

    def test_each_truncated_call_prefix_is_literal_and_never_announced(self):
        text='<tool_call><function=f><parameter=x>\n{"中":[1,true]}\n</parameter></function></tool_call>'
        for end in range(1,len(text)):
            self.assertEqual(self.parse(list(text[:end]),thinking=False,stream_tools=True),[('content',text[:end])])
        p=StepOutputParser(thinking=False,stream_tools=True)
        self.assertEqual(p.feed(text[:-1]),[])
        events=p.feed(text[-1]);self.assertEqual(len(events),1);self.assertEqual(events[0].kind,'tool_call')

    def test_unknown_schema_infers_scalars_and_preserves_text(self):
        for value,expected in [('True',True),('False',False),('None',None),('true',True),('null',None),('hello','hello')]:
            text='<tool_call><function=f><parameter=x>'+value+'</parameter></function></tool_call>'
            self.fragmentation(text,[('tool_call','f',{'x':expected})],thinking=False)

    def test_utf8_byte_fragments_and_partial_final_tags(self):
        text='考える🌍</think>Привет<tool_call><function=f><parameter=x>中🙂</parameter></function></tool_call>'
        decoder=codecs.getincrementaldecoder('utf8')()
        chunks=[decoder.decode(bytes([b])) for b in text.encode()]
        chunks.append(decoder.decode(b'',final=True))
        self.assertEqual(self.parse(chunks),self.parse([text]))
        for suffix in ('<','<tool_','<think','</thi'):
            self.assertEqual(self.parse(['text',suffix],thinking=False),[('content','text'+suffix)])

    def test_template_parser_continuation_roundtrip(self):
        args={'s':'\n 123\n','b':True,'n':None,'obj':{'a':[1,False]},'num':2}
        tools=[{'name':'f','parameters':{'properties':{'s':{'type':'string'},'b':{'type':'boolean'},'n':{'type':'null'}}}}]
        history=[{'role':'user','content':'do it'},{'role':'assistant','reasoning_content':'r','content':'',
            'tool_calls':[{'id':'a','function':{'name':'f','arguments':args}}]}]
        rendered=template().render(history,tools,add_generation_prompt=False)
        call=rendered[rendered.index('<tool_call>\n<function=f>'):].split('</tool_call>',1)[0]+'</tool_call>'
        self.fragmentation(call,[('tool_call','f',args)],thinking=False,tools=tools)
        history.append({'role':'tool','tool_call_id':'a','content':'done'})
        self.assertTrue(template().render(history,tools).endswith('<tool_response>done</tool_response>'+EOS+'\n'+GENERATION))


if __name__ == '__main__':
    unittest.main()
