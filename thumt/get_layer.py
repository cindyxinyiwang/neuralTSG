# -*- encoding: utf-8 -*-
import numpy
import theano
import theano.tensor as tensor
from nmt import RNNsearch
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import tools
from layer import LayerFactory
from config import * 
from optimizer import adadelta, SGD
from data import *

import cPickle
import json
import argparse
import signal
import time

parser = argparse.ArgumentParser("train the translation model")
parser.add_argument('-m', '--model', required=True, help='path to NMT model')
parser.add_argument('-s', '--sentence', required = True, help = 'the sentence need to be visualized')
if __name__ == "__main__":
	args = parser.parse_args()

	#load vocab and model
	print 'load vocab and model'
	values = numpy.load(args.model)
	config = values['config']
	config = json.loads(str(config))
	
	print config
	
	model = eval(config['model'])(config)
	model.build(verbose=True)
	values = model.load(args.model, decode=True)
	data = DataCollection(config, train=False)
	data.vocab_src = json.loads(str(values['vocab_src']))
	data.ivocab_src = json.loads(str(values['ivocab_src']))
	data.vocab_trg = json.loads(str(values['vocab_trg']))
	data.ivocab_trg = json.loads(str(values['ivocab_trg']))
	data.encode_vocab()
	src = ''
	src = args.sentence
	src_index = data.toindex_source(src.split())
	print'src_index', src_index,src_index[::-1]
	result = model.translate(src_index)
	result = numpy.asarray(result)
	trg_index = numpy.transpose(numpy.asarray([result]))
	print 'trg_index',trg_index
	trg = data.print_target(trg_index)
	print trg
	layers = model.get_layer(src_index, numpy.ones(src_index.shape, dtype=numpy.float32), trg_index, numpy.ones(trg_index.shape, dtype=numpy.float32))
	
	layers['enc_for_x'] = src_index
	layers['enc_back_x'] = src_index[::-1]
	layers['dec_y'] = trg_index
	numpy.savez('./val.npz',**layers)
	f = open('sen.txt','w')
	f.write(src + ' eof\n')
	f.write(trg + ' eof\n')
	print len(layers)
	for i in layers:
		print i, layers[i].shape

