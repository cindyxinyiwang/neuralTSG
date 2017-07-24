import argparse
import os
import numpy as np
import types

root_dir = '/Users/cindywang/Documents/research/TSG/code/thumt'
code_dir = root_dir + '/thumt'
debug = 1
training_criterion = 0
save_all = 0
unkreplace = 0
device = 'cpu'

def config(config_file, output_file):
	d = {}

	d['model'] = 'RNNtsg'
	# training criteria
	if training_criterion == 0:
		d['MRT'] = False
		d['semi_learning'] = False
	elif training_criterion == 1:
		d['MRT'] = True
		d['semi_learning'] = False
	else:
		d['MRT'] = False
		d['semi_learning'] = True
		d['model'] = 'BiRNNsearch'
	# data
	d['src_text'] = 'data/source'
	d['trg_text'] = 'data/tsg'
	d['valid_src'] = 'data/source'
	d['valid_ref'] = 'data/tsg'
	d['src'] = 'corpus/source.json'
	d['trg'] = 'corpus/tsg.json'
	d['tsg_obj'] = 'corpus/tsg.tsg.pkl'
	d['src_shuf'] = 'corpus/source.json.shuf'
	d['trg_shuf'] = 'corpus/tsg.json.shuf'
	d['vocab_src'] = 'corpus/vocab.source.pkl'
	d['vocab_trg'] = 'corpus/vocab.tsg.pkl'
	d['ivocab_src'] = 'corpus/ivocab.source.pkl'
	d['ivocab_trg'] = 'corpus/ivocab.tsg.pkl'
	d['data_corpus'] = 'json'	

	if debug == 0:
		d['verbose_level'] = 'info'
	else:
		d['verbose_level'] = 'debug'

	d['src_mono_text'] = ''
	d['src_mono'] = ''
	d['src_mono_shuf'] = ''


	f1 = open(config_file, 'r')
	while True:
		line = f1.readline()
		if line == '':
			break
		if '[source vocabulary size]' in line:
			d['index_eos_src'] = int(line.split(']')[-1].strip())
			d['num_vocab_src'] = d['index_eos_src'] + 1
		elif '[target vocabulary size]' in line:
			d['index_eos_trg'] = int(line.split(']')[-1].strip())
			d['num_vocab_trg'] = d['index_eos_trg'] + 1
		elif '[source word embedding dimension]' in line:
			d['dim_emb_src'] = int(line.split(']')[-1].strip())
		elif '[target word embedding dimension]' in line:
			d['dim_emb_trg'] = int(line.split(']')[-1].strip())
		elif '[encoder hidden layer dimension]' in line:
			d['dim_rec_enc'] = int(line.split(']')[-1].strip())
		elif '[decoder hidden layer dimension]' in line:
			d['dim_rec_dec'] = int(line.split(']')[-1].strip())
		elif '[MRT sample size]' in line:
			d['sampleN'] = int(line.split(']')[-1].strip()) 
		elif '[MRT length ratio limit]' in line:
			d['LenRatio'] = float(line.split(']')[-1].strip())
		elif '[maximum sentence length]' in line:
			d['maxlength'] = int(line.split(']')[-1].strip())
		elif '[mini-batch size]' in line:
			d['batchsize'] = int(line.split(']')[-1].strip())
			if training_criterion == 1:  # batch size must be set to 1 in MRT
				d['batchsize'] = 1
		elif '[mini-batch sorting size]' in line:
			d['sort_batches'] = int(line.split(']')[-1].strip()) 
		elif '[iteration limit]' in line:
			d['max_iter'] = int(line.split(']')[-1].strip()) 
		elif '[convergence limit]' in line:
			d['try_iter'] = int(line.split(']')[-1].strip())
		elif '[optimizer]' in line:
			x = int(line.split(']')[-1].strip())
			if x == 0: 
				d['optimizer'] = 'SGD'
			elif x == 1:
				d['optimizer'] = 'adadelta'
			elif x == 2:
				d['optimizer'] = 'adam_slowstart'
			else:
				d['optimizer'] = 'adam'
		elif '[clip]' in line:
			d['clip'] = float(line.split(']')[-1].strip())
		elif '[SGD learning rate]' in line:
			d['lr'] = float(line.split(']')[-1].strip())
		elif '[AdaDelta rho]' in line:
			d['rho'] = float(line.split(']')[-1].strip())
		elif '[AdaDelta epsilon]' in line:
			d['epsilon'] = float(line.split(']')[-1].strip())
		elif '[Adam alpha]' in line:
			d['alpha_adam'] = float(line.split(']')[-1].strip()) 
		elif '[Adam alpha decay]' in line:
			d['alphadecay_adam'] = float(line.split(']')[-1].strip()) 
		elif '[Adam beta1]' in line:
			d['beta1_adam'] = float(line.split(']')[-1].strip())
		elif '[Adam beta2]' in line:
			d['beta2_adam'] = float(line.split(']')[-1].strip()) 
		elif '[Adam eps]' in line:
			d['eps_adam'] = float(line.split(']')[-1].strip()) 
		elif '[beam size]' in line:
			d['beam_size'] = int(line.split(']')[-1].strip()) 
		elif '[model dumping iteration]' in line:
			d['save_freq'] = int(line.split(']')[-1].strip()) 
		elif '[checkpoint iteration]' in line:
			d['checkpoint_freq'] = int(line.split(']')[-1].strip()) 
	# generate configuation file in the internal format
	f2 = open(output_file, 'w')
	for key in d:
		if type(d[key]) is types.StringType:
			f2.write('"' + key + '": "' + str(d[key]) + '"\n')
		else:
			f2.write('"' + key + '": ' + str(d[key]) + '\n')

if __name__=="__main__":
	config('config/Thumt.config', '_config')

	if not os.path.exists('model'):
		os.makedirs('model')
	if not os.path.exists('valid'):
		os.makedirs('valid')

	# training
	optional = ''
	if debug == 1:
		optional += ' --debug'
	if save_all == 1:
		optional += ' --save-all'
	if unkreplace == 1:
		optional += ' --map mapping.pkl'
	os.system('THEANO_FLAGS=floatX=float32,optimizer=None,device=' + device + \
			  	',lib.cnmem=0.5 python ' + code_dir + \
			    '/train.py -c _config' + optional)

