from transformers import GPT2Config

config = GPT2Config(
    vocab_size=50257,
    n_positions=512,
    n_ctx=512,
    n_embd=256,
    n_layer=4,
    n_head=4
)