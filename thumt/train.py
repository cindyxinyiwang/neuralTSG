import numpy
import theano
import theano.tensor as tensor
from nmt import RNNsearch, RNNtsg
from binmt import BiRNNsearch
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import tools
from layer import LayerFactory
from config import * 
from optimizer import adadelta, SGD, adam, adam_slowstart
from data import DataCollection, getbatch
from mrt_utils import getMRTBatch

import cPickle
import json
import argparse
import signal
import time
import datetime
import logging
import types

parser = argparse.ArgumentParser("script to train the NMT model")
parser.add_argument('-c', '--config', help = 'path to config file', required = True)
parser.add_argument('--debug', action = 'store_true', help = 'set verbose level to debug')
parser.add_argument('--map', help = 'path to mapping file')
parser.add_argument('--save-all', action = 'store_true', help = 'save all models on validation')
args = parser.parse_args()

if args.debug:
	logging.basicConfig(level = logging.DEBUG,
		format = '[%(asctime)s %(levelname)s] %(message)s',
		datefmt = '%d %b %H:%M:%S')
	logging.debug('training with debug info')
else:
	logging.basicConfig(level = logging.INFO,
		format = '[%(asctime)s %(levelname)s] %(message)s',
		datefmt = '%d %b %H:%M:%S')

if __name__ == '__main__':

	# initialize config
	config = config()
	if args.config:
		config = update_config(config, load_config(open(args.config, 'r').read()))
	print_config(config)

	if config['MRT']:
		config['batchsize'] = 1

	mapping = None
	if args.map:
		mapping = cPickle.load(open(args.map, 'r'))

	logging.info('STEP 2: Training')
	# prepare data
	logging.info('STEP 2.1: loading training data ...')
	data = DataCollection(config)
	logging.info('Done\n')

	# build model
	logging.info('STEP 2.2: building model ...')
	model = eval(config['model'])(config)
	model.build()
	logging.info('Done\n')

	logging.info('STEP 2.3: build optimizer ...')
	trainer = eval(config['optimizer'])(config, model.creater.params)
	update_grads, update_params = trainer.build(model.cost, model.inputs)
	logging.info('Done\n')

	# load checkpoint
	logging.info('STEP 2.4: loading checkpoint ...')
	data.load_status(config['checkpoint_status'])
	model.load(config['checkpoint_model'])
	logging.info('Done\n')

	# train
	logging.info('STEP 2.5: start training ...')
	f = open('log', 'w')
	while data.num_iter < config['max_iter']:
		try:
			st = time.time()
			data.num_iter += 1
			trainx, trainy = data.next()
			x, xmask, y, ymask = getbatch(trainx, trainy, config)
			
			if 'MRT' in config and config['MRT'] is True:
				x, xmask, y, ymask, MRTLoss = getMRTBatch(x, xmask, y, ymask, config, model, data)
			if config['semi_learning']:
				xm, ym = data.next_mono()
				xm, xmask, ym, ymask = getbatch(xm, ym, config)
				x, xmask, y, ymask, valid = model.get_inputs_batch(x, y, xm, ym)
			
			y_rule_idx, y_parent_idx, y_parent_t = y[:,:,0], y[:,:,1], y[:,:,2]
			
			# sample
			if data.num_iter % config['sample_freq'] == 0:
				logging.info('%d iterations passed, %d sentences trained' % (data.num_iter, data.num_iter*config['batchsize']))
				logging.info('sampling')
				if config['sample_sentence']:
					xs = data.toindex_source(config['sample_sentence'].split(' '))
					logging.info('source: %s' % data.print_source(xs))
					sample, probs = model.sample(tools.cut_sentence(xs,config['index_eos_src']), config['sample_length'])
					logging.info('output: %s\n' % data.print_target(sample[0]))
				else:
					for i in range(min(x.shape[1], config['sample_times'])):
						logging.info('source: %s' % data.print_source(x[:,i]))
						logging.info('target: %s' % data.print_target(y[:,i]))
						sample, probs = model.sample(tools.cut_sentence(x[:,i], config['index_eos_src']), config['sample_length'])
						logging.info('output: %s\n' % data.print_target(sample[0]))
			
			# save checkpoint
			if data.num_iter % config['checkpoint_freq'] == 0:
				model.save(config['checkpoint_model'], data = data, mapping = mapping)
				data.save_status(config['checkpoint_status'])

			# save model and validate
			if config['save']:
				if data.num_iter % config['save_freq'] == 0:
					if args.save_all:
						logging.info('saving model...')
						model.save(config['save_path'] + '/model_iter' + str(data.num_iter) + '.npz', data = data, mapping = mapping)
					logging.info('Done\n')
					logging.info('validating the model...')
					output_path = config['valid_dir'] + '/iter_' + str(data.num_iter) + '.trans'
					valid_input = open(config['valid_src'], 'r')
					valid_output = open(output_path, 'w')
					line = valid_input.readline()
					valid_num = 0
					while line != '':
						line = line.strip()
						result = model.translate(data.toindex_source(line.split(' ')))
						print >> valid_output, data.print_target(numpy.asarray(result))
						valid_num += 1
						if valid_num % 100 == 0:
							logging.info('%d sentences translated' % valid_num)
						line = valid_input.readline()
					valid_output.close()
					valid_refs = tools.get_ref_files(config['valid_ref'])
					data.valid_result[data.num_iter] = 100 * tools.bleu_file(output_path, valid_refs)
					tolog = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\t' + \
					        'iterations:' + str(data.num_iter) + '\t' + \
							'BLEU:' + str(data.valid_result[data.num_iter]) + '\t' + \
							'cost:' + str(sum(data.train_cost[(data.num_iter - config['save_freq']): data.num_iter]) / config['save_freq'])
					print >> f, tolog
					data.print_valid()
					#update the best model
					if data.last_improved(last=True) == 0:
						model.save(config['save_path'] + '/model_best.npz', data = data, mapping = mapping)
						#testcfg = {}
						#testcfg["config"] = args.config
						#testcfg["model"] = '"' + config['save_path'] + '/model_iter' + str(data.num_iter) + '.npz"'
						#testcfg["num_iterations"] = data.num_iter
						#json.dump(testcfg, open('test.config', 'w'))
					if data.last_improved() >= config['try_iter']:
						logging.info('no improvement in %d iterations, stop training' % data.last_improved())
						break
					logging.info('Done\n')
					

			# train
			upst = time.time()
			if 'MRT' in config and config['MRT'] is True:
				cost, grad_norm = update_grads(x, xmask, y_rule_idx, y_parent_t, ymask, MRTLoss)
			elif config['semi_learning']:
				cost, grad_norm = update_grads(x, xmask, y, ymask, y, ymask, x, xmask, valid)
			else:
				#print y_parent_t
				cost, grad_norm = update_grads(x, xmask, y_rule_idx, y_parent_t, ymask)
			# NaN process
			if numpy.isinf(cost.mean()) or numpy.isnan(cost.mean()):
				logging.warning('nan while training')
			update_params()
			ed = time.time()
			data.time += ed-st
			data.updatetime += ed-upst

			data.train_cost.append(cost.mean())
			logging.debug('iter: %d; cost: %.4f; grad_norm: %.3e;' % (data.num_iter, cost.mean(), grad_norm)+
			'iter time: %.3f sec; total time: %s' % (ed - st, tools.print_time(data.time)))
		except KeyboardInterrupt:
			logging.info('stop training on keyboard interrupt')
			break

	#save checkpoint
	s = signal.signal(signal.SIGINT, signal.SIG_IGN)
	logging.info('saving model and status...')
	model.save(config['checkpoint_model'], data = data, mapping = mapping)
	data.save_status(config['checkpoint_status'])
	logging.info('Done')
	signal.signal(signal.SIGINT, s)

