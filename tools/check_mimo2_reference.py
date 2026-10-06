"""Compare a pinned upstream config/loader with the admitted local MiMo GGUF.

Uses already downloaded public files; never executes upstream Python or reads weights.
Only mimo2.cpp equality is checked, not equivalence of complete dependency trees.
"""
import argparse
import json
import math
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.gguf_reader import GGUFFile
from tools.inspect_mimo2_gguf import inspect_model, file_hash, protect_output
from tools.mimo2_loader_contract import SOURCE_SHA, REFERENCE_SHA, LOADER_SHA, LOADER_FILE_SHA256

CONFIG_SHA256 = '61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621'


def compare(config, meta):
    rows = []

    def check(name, source, local):
        equal = (math.isclose(source, local, rel_tol=1e-6)
                 if type(source) is float and type(local) in (int, float) else source == local)
        rows.append({'name': name, 'source': source, 'local': local, 'pass': equal})

    mapping = {
        'num_hidden_layers': 'block_count', 'hidden_size': 'embedding_length',
        'intermediate_size': 'feed_forward_length', 'moe_intermediate_size': 'expert_feed_forward_length',
        'n_routed_experts': 'expert_count', 'num_experts_per_tok': 'expert_used_count',
        'num_attention_heads': 'attention.head_count', 'head_dim': 'attention.key_length',
        'v_head_dim': 'attention.value_length', 'sliding_window': 'attention.sliding_window',
        'hybrid_layer_pattern': 'attention.sliding_window_pattern', 'rope_theta': 'rope.freq_base',
        'swa_rope_theta': 'rope.freq_base_swa', 'layernorm_epsilon': 'attention.layer_norm_rms_epsilon',
        'attention_value_scale': 'attention.value_scale', 'max_position_embeddings': 'context_length',
        'n_group': 'expert_group_count', 'topk_group': 'expert_group_used_count',
    }
    for source, local in mapping.items():
        check(source, config[source], meta['mimo2.'+local])
    for source, local in [('eos_token_id', 'eos_token_id'), ('pad_token_id', 'padding_token_id')]:
        check(source, config[source], meta['tokenizer.ggml.'+local])
    check('vocab_size', config['vocab_size'], len(meta['tokenizer.ggml.tokens']))
    check('rope_dimension', int(config['head_dim']*config['partial_rotary_factor']), meta['mimo2.rope.dimension_count'])
    pattern = config['hybrid_layer_pattern']
    check('layerwise_kv_heads', [config['swa_num_key_value_heads'] if s else config['num_key_value_heads'] for s in pattern],
          meta['mimo2.attention.head_count_kv'])
    for source, local in [('swa_head_dim', 'attention.key_length'), ('swa_v_head_dim', 'attention.value_length'),
                          ('swa_num_attention_heads', 'attention.head_count')]:
        check(source, config[source], meta['mimo2.'+local])
    # Semantics established by the pinned loader and strict tensor contract,
    # not all separately represented in GGUF metadata.
    for key, expected in {
        'architectures': ['MiMoV2ForCausalLM'], 'model_type': 'mimo_v2',
        'attention_projection_layout': 'fused_qkv', 'attention_bias': False,
        'add_full_attention_sink_bias': False, 'add_swa_attention_sink_bias': True,
        'moe_layer_freq': [0]+[1]*47, 'n_shared_experts': None, 'norm_topk_prob': True,
        'scoring_func': 'sigmoid', 'topk_method': 'noaux_tc', 'routed_scaling_factor': None,
        'hidden_act': 'silu', 'tie_word_embeddings': False, 'bos_token_id': None,
        'dtype': 'bfloat16', 'moe_router_dtype': 'bfloat16', 'num_nextn_predict_layers': 3,
    }.items():
        check('contract/'+key, config[key], expected)
    check('source_quant_method', config['quantization_config']['quant_method'], 'fp8')
    check('source_expert_storage', config['quantization_config']['store_dtype'], 'mxfp4')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gguf', 'config', 'reference-loader', 'candidate-loader', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = protect_output(args.output, args.gguf, args.config, args.reference_loader, args.candidate_loader)
        inventory = inspect_model(args.gguf)
        hashes = {name: file_hash(path, path.stat().st_size) for name, path in
                  [('config', args.config), ('reference_loader', args.reference_loader), ('candidate_loader', args.candidate_loader)]}
        if hashes != dict(config=CONFIG_SHA256, reference_loader=LOADER_FILE_SHA256, candidate_loader=LOADER_FILE_SHA256):
            raise ValueError('Pinned reference/candidate hash differs')
        config = json.loads(args.config.read_text(encoding='utf-8'))
        rows = compare(config, GGUFFile(args.gguf).metadata)
        failures = sum(not row['pass'] for row in rows)
        report = {
            'status': 'fail' if failures else 'pass', 'checks': rows, 'check_count': len(rows), 'mismatch_count': failures,
            'scope': 'source config and single loader file; no conversion execution, whole-tree equality or inference',
            'gguf_header_sha256': inventory['header_sha256'], 'hashes': hashes,
            'source_config_url': f'https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/resolve/{SOURCE_SHA}/config.json',
            'reference_loader_url': f'https://raw.githubusercontent.com/ggml-org/llama.cpp/{REFERENCE_SHA}/src/models/mimo2.cpp',
            'candidate_revision': LOADER_SHA, 'complete_dependency_equivalence_verified': False,
            'expected_export_differences': {
                'nextn_layers': {'source': 3, 'gguf': 0}, 'vision_audio_companions': 'excluded from GGUF',
                'router_dtype': {'source': 'bfloat16', 'gguf': 'F32'},
                'weights': 'source FP8/MXFP4 description; GGUF types independently admitted as BF16/F32/Q2_K/Q3_K/MXFP4',
            },
        }
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(report['status'], 'reference', len(rows), 'checks;', failures, 'mismatches;', output)
        return int(bool(failures))
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print('MiMo reference check failed:', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
