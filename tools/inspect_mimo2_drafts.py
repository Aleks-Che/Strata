"""Validate local MiMo MTP/DFlash sidecars; static compatibility is not runtime acceptance."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tools.gguf_reader import GGUFFile
from tools.inspect_hy3_gguf import file_hash, protect_output, validate_ranges
from tools.mimo2_loader_contract import validate_loader_contract


def validate_draft(meta, tensors, target):
    arch=meta.get('general.architecture')
    if arch not in ('mimo2','dflash'):
        raise ValueError('Expected MiMo MTP or DFlash architecture')
    def require(key,value):
        actual=meta.get(key)
        if isinstance(value,float):
            ok=type(actual) is float and abs(actual-value)<1e-7
        else:
            ok=type(actual) is type(value) and actual==value
        if not ok: raise ValueError('Unreviewed sidecar metadata: '+key)
    prefix=arch+'.'
    for key,value in {'embedding_length':4096,'context_length':1048576,'feed_forward_length':16384,
        'attention.head_count':64,'attention.value_length':128,'rope.dimension_count':64,
        'attention.layer_norm_rms_epsilon':1e-6}.items(): require(prefix+key,value)
    for key in ('model','pre','tokens','token_type','merges','eos_token_id','padding_token_id','add_bos_token'):
        k='tokenizer.ggml.'+key
        if meta.get(k)!=target.get(k): raise ValueError('Target/draft tokenizer mismatch: '+key)
    if len(meta['tokenizer.ggml.tokens'])!=152576: raise ValueError('Wrong vocabulary')
    expected={}
    def add(name,shape): expected[name]=shape
    add('token_embd.weight',[4096,152576]);add('output.weight',[4096,152576]);add('output_norm.weight',[4096])
    if arch=='mimo2':
        require(prefix+'block_count',51);require(prefix+'nextn_predict_layers',3)
        for key in ('expert_count','expert_used_count','expert_group_count','expert_group_used_count','expert_gating_func',
                    'expert_feed_forward_length','attention.key_length','attention.sliding_window','rope.freq_base','rope.freq_base_swa','attention.value_scale'):
            require(prefix+key,target[prefix+key])
        require(prefix+'attention.head_count_kv',target[prefix+'attention.head_count_kv']+[8]*3)
        require(prefix+'attention.sliding_window_pattern',target[prefix+'attention.sliding_window_pattern']+[1]*3)
        for i in range(48,51):
            p=f'blk.{i}.'
            add(p+'nextn.eh_proj.weight',[8192,4096])
            for n in ('nextn.enorm','nextn.hnorm','layer_output_norm'): add(p+n+'.weight',[4096])
            add(p+'attn_qkv.weight',[4096,14848]);add(p+'attn_output.weight',[8192,4096])
    else:
        for key,value in {'block_count':5,'attention.head_count_kv':8,'attention.key_length':128,
            'attention.causal':False,'attention.sliding_window':1024,'attention.sliding_window_pattern':[True]*5,
            'attention.value_scale':.612,'rope.freq_base':10000.0,'block_size':8,'target_layers':[1,12,24,36,48]}.items(): require(prefix+key,value)
        require('tokenizer.ggml.mask_token_id',151675)
        if any(k in meta for k in ('dflash.sample_from_anchor','dflash.hyper_connection.count','dflash.selector_rank')):
            raise ValueError('DSpark/DFlash2 is not this MiMo draft')
        add('fc.weight',[20480,4096]);add('enc.output_norm.weight',[4096])
        for i in range(5):
            p=f'blk.{i}.'
            for n in ('q','k'): add(p+'attn_'+n+'_norm.weight',[128])
            for n,w in [('q',8192),('k',1024),('v',1024)]: add(p+'attn_'+n+'.weight',[4096,w])
            add(p+'attn_output.weight',[8192,4096])
    for i in (range(48,51) if arch=='mimo2' else range(5)):
        p=f'blk.{i}.'
        for n in ('attn_norm','ffn_norm'): add(p+n+'.weight',[4096])
        add(p+'attn_sinks.weight',[64])
        for n in ('gate','up'): add(p+'ffn_'+n+'.weight',[4096,16384])
        add(p+'ffn_down.weight',[16384,4096])
    if len(tensors)!=len(expected) or {t.name for t in tensors}!=set(expected):
        raise ValueError('Incomplete or unexpected draft tensors')
    matrix_types=set()
    for t in tensors:
        if t.shape!=expected[t.name]: raise ValueError('Wrong shape: '+t.name)
        if len(t.shape)==1:
            if t.type_name!='F32': raise ValueError('Expected F32 norm/sink: '+t.name)
        else:
            allowed = ('BF16','Q8_0','Q4_0') if arch=='mimo2' else ('BF16','Q8_0')
            if t.type_name not in allowed: raise ValueError('Unreviewed draft quant: '+t.name)
            matrix_types.add(t.type_name)
    if len(matrix_types)!=1: raise ValueError('Mixed sidecar matrix encoding not reviewed')
    return dict(kind='mtp' if arch=='mimo2' else 'dflash',matrix_type=next(iter(matrix_types)),
        tensor_count=len(tensors),tokenizer_equal=True,template_equal=meta.get('tokenizer.chat_template')==target.get('tokenizer.chat_template'),
        draft_blocks=3 if arch=='mimo2' else 5,max_new_draft_tokens=3 if arch=='mimo2' else 7,
        runtime_compatible=None,acceptance_measured=False)


def inspect(target_path,draft_paths,full_hash=False):
    target=GGUFFile(target_path)
    validate_loader_contract(target.metadata,target.tensors)
    result=dict(status='pass',scope='Static tensor/range/tokenizer checks only; no inference or speed claim',
        target=str(Path(target_path).resolve()),target_header_sha256=file_hash(target_path,target.header_end),drafts=[])
    for path in draft_paths:
        path=Path(path).resolve();g=GGUFFile(path)
        contract=validate_draft(g.metadata,g.tensors,target.metadata)
        rows=validate_ranges(g.tensors,g.data_start,path.stat().st_size,g.alignment)
        shared=sum(t.expected_bytes() for t in g.tensors if t.name in ('output.weight','token_embd.weight','output_norm.weight'))
        meta={k:v for k,v in g.metadata.items() if not k.startswith('tokenizer.')}
        result['drafts'].append(dict(path=str(path),file_bytes=path.stat().st_size,header_sha256=file_hash(path,g.header_end),
            full_sha256=file_hash(path,path.stat().st_size) if full_hash else None,contract=contract,metadata=meta,
            payload_bytes=sum(r['bytes'] for r in rows),embedding_and_head_bytes=shared,
            block_payload_bytes=sum(r['bytes'] for r in rows)-shared,
            types=dict(Counter(t.type_name for t in g.tensors)),tensors=rows))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--draft',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--full-hash',action='store_true')
    a=p.parse_args();out=protect_output(a.output,a.model,*a.draft)
    if out.suffix!='.json' or out.exists(): raise ValueError('Use a fresh .json report')
    result=inspect(a.model,a.draft,a.full_hash)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('PASS',len(result['drafts']),'sidecars;',out)


if __name__=='__main__': main()
