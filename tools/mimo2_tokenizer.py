"""MiMo-specific vocabulary normalization matching the pinned native oracle.

Raw GGUF metadata stays untouched. Do not apply MiMo's special-token override
to unrelated qwen2 vocabularies, or infer a serving stop policy from this module.
"""
from .gguf_reader import GGUFFile
from .mimo2_loader_contract import model_metadata
from .strata_tokenizer import Tokenizer


def from_gguf(path):
    meta = GGUFFile(path).metadata
    model_metadata(meta)
    tokens, kinds = list(meta['tokenizer.ggml.tokens']), list(meta['tokenizer.ggml.token_type'])
    # llama-vocab.cpp 86ebfef2 promotes this normal token by spelling during
    # EOG discovery. With parse_special=true it becomes a direct literal match.
    if tokens[128247] != '</s>' or kinds[128247] != 1:
        raise ValueError('Unreviewed MiMo </s> vocabulary metadata')
    kinds[128247] = 3
    special = {k: v for k, v in meta.items() if k.startswith('tokenizer.ggml.') and k.endswith('_token_id')}
    return Tokenizer(tokens, meta['tokenizer.ggml.merges'], kinds, pre='qwen2', special_ids=special)
