import tensorflow as tf
import numpy as np

from datasets.cornell_corpus import data
import data_utils
import seq2seq_model

metadata, idx_q, idx_a = data.load_data(PATH='datasets/cornell_corpus/')

xseq_len = 25
yseq_len = 25
batch_size = 32
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024

model = seq2seq_model.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )


sess = model.restore_last_session()
print("\n")
for x in range(100):
	text = input(">")
	try:
		print(model.get_response(text, metadata, sess))
	except ValueError:
		print("Hmm")
