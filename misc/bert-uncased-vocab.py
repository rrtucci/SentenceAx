from transformers import AutoTokenizer
from pprint import pprint

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

min_vocab = min(tokenizer.vocab.values())
max_vocab = max(tokenizer.vocab.values())

name_to_tok = tokenizer.special_tokens_map
name_to_ilabel = {name: tokenizer.convert_tokens_to_ids(tok)
                  for name, tok in name_to_tok.items()}
print("min icode, max icode=", min_vocab, max_vocab)
print("name_to_tok=")
pprint(name_to_tok)
print("name_to_ilabel=")
pprint(name_to_ilabel)