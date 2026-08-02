# Vendored from MusicTransformer-tensorflow2.0's params.py, reduced to the constants the
# vendored slice reads, with a tiny event dimension so the fixture runs quickly.
event_dim = 16
pad_token = event_dim
token_sos = event_dim + 1
token_eos = event_dim + 2
vocab_size = event_dim + 3
