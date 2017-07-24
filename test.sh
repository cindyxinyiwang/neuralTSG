#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=48:00:00
#PBS -N session1_default
#PBS -A course
#PBS -q GpuQ

THEANO_FLAGS='floatX=float32' python ./scripts/test.py \
	--model-file  checkpoint_model.npz \
	--test-src-file data/source \
	--test-trg-file tsg.trg \
	--device cpu
	
