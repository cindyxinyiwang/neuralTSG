import re
import cPickle, json
import os
from preprocess import preprocess, shuffle


class TreeNode:
	def __init__(self, label, children=None):
		if ":" in label:
			toks = label.split(":")
			self.nonterm = toks[0]	# x0, x1 etc,
			self.tag = toks[1]	# NP, VP etc, 
		else:
			self.nonterm = None
			self.tag = label 
		self.label = label # The whole label string, including nonterm info
		self.children = children

	def __str__(self):
		if not self.children:
			return "( " + self.label + " )"
		child_str = " ".join([str(c) for c in self.children])
		return "( " + self.label + " " + child_str + " )"

class ParseError(Exception):
	pass

class TSGrule:
	def __init__(self, lhs, rhs, nonterm_node_dict={}):
		self.lhs = lhs 	# A TreeNode object
		self.rhs = rhs	# a list of target tokens and nonterminals
		self.nonterm_node_dict = nonterm_node_dict	# dict key: nonterminals, x0, x1 etc; values: lhs tree node
		self.parent = None # parent TSGrule
		self.children = [] # list of TSGrule as current rule's children
		self.expand_tags = []
		for tok in self.rhs.split():
			if tok in self.nonterm_node_dict:
				self.expand_tags.append(self.nonterm_node_dict[tok].tag)

	def __str__(self):
		return str(self.lhs) + "->" + str(self.rhs)

	def get_expand_tags(self):
		'''
		return a list of nonterminal labels in the order of rhs derivation
		'''
		return self.expand_tags


class TSG:
	def __init__(self, tsg_vocab, vocab, ivocab):
		self.tsg_vocab = tsg_vocab	# list, key: rule_idx; value: TSG_rule
		self.vocab = vocab # list, key: rule_idx, value: TSG string
		self.ivocab = ivocab # dict, key: TSG string, valueL rule_idxs
		self.rules_with_root_tag = {}
		for idx, tsg in enumerate(self.tsg_vocab):
			if not tsg:
				continue
			if tsg.lhs.tag in self.rules_with_root_tag:
				self.rules_with_root_tag[tsg.lhs.tag].append(idx)
			else:
				self.rules_with_root_tag[tsg.lhs.tag] = [idx]

	def get_rule_from_idx(self, idx):
		'''
		return the TSGrule object associated with the given idx
		'''
		return self.tsg_vocab[idx]

	def get_idx_from_rule(self, tsg_str):
		return self.ivocab[tsg_str]

	def rule_idx_with_root(self, root_tag):
		'''
		Return all rule indices with root_tag as root node of LHS
		'''
		if root_tag in self.rules_with_root_tag:
			return self.rules_with_root_tag[root_tag]
		else:
			print 'No rules given root tag, should not happen: ', root_tag
			return []


def parse_rule(line):
	""""
	Parse a line into a TSG rule
	"""
	def tokenize(s):
		toks = re.compile(' +|[A-Za-z0-9:\-\.]+|[()]')
		for match in toks.finditer(s):
			s = match.group(0)
			if s[0] == ' ':
				continue
			if s[0] in '()':
				yield (s, s)
			else:
				yield ('WORD', s)
	def parse_inner(toks):
		ty, name = next(toks)
		if ty != 'WORD': raise ParseError
		children_tok = []
		while True:
			ty, s = next(toks)
			if ty == '(':
				children_tok.append(parse_inner(toks))
			elif ty == ')':
				# check for nonterm
				node = TreeNode(name, children_tok)
				if ":" in name:
					t = name.split(":")
					nonterm_dict[t[0]] = node
				return node
	def parse_root(toks):
		ty, _ = next(toks)
		if ty != '(': raise ParseError
		return parse_inner(toks)
	# Get a TSGrule object from a line
	line = line.split("->")
	lhs, rhs = line[0], line[1]
	# parse lhs into a TreeNode
	nonterm_dict = {}
	lhs_tree = parse_root(tokenize(lhs))
	return TSGrule(lhs=lhs_tree, rhs=rhs, nonterm_node_dict=nonterm_dict)


def parse_tsg(tsg_file):
	'''
	Get a list of tsg sequence from a tsg file
	'''
	rule_seqs = []
	rules = []
	parents = []
	with open(tsg_file) as file:
		for line in file:
			if not line.strip():
				rule_seqs.append(rules)
				rules = []
				assert len(parents) == 0
			else:
				tsg = parse_rule(line.strip())
				#print len(tsg.nonterm_node_dict), len(tsg.children)
				#print "children ", str(tsg)
				#for c in tsg.children:
				#	print str(c)
				rules.append(tsg)
				if parents:
					tsg.parent = parents[-1]
					parents[-1].children.append(tsg)
					#print str(tsg), str(tsg.parent), len(tsg.nonterm_node_dict)
				if len(tsg.nonterm_node_dict) > 0:
					parents.append(tsg)
				#print "parents"
				#for p in parents:
				#	print str(p), len(p.nonterm_node_dict), len(p.children)
				# pop out filled up parents
				while parents:
					if len(parents[-1].children) == len(parents[-1].nonterm_node_dict):
						parents.pop()
					elif len(parents[-1].children) < len(parents[-1].nonterm_node_dict):
						break
	if rules:
		rule_seqs.append(rules)
	return rule_seqs

def preprocess_tsg(num, vocab_f, ivocab_f, tsg_f, idx_rule_f, tsg_obj_f, data_vocab='cPickle', data_corpus='json'):
	'''
	Get vocab and inverse vocab based on word count
	Index the rule seqs

	vocab_f: output vocab file name
	ivocab_f: output inverse vocab file name
	tsg_f: tsg file
	idx_rule_f: indexed tsg file
	tsg_obj_f: TSG object pickled file

	'''
	print "Building dictionary tsg"
	rule_seqs = parse_tsg(tsg_file)
	count = {}
	tsg_dict = {}
	for rule_seq in rule_seqs:
		for tsg in rule_seq:
			r = str(tsg)
			tsg_dict[r] = tsg
			if not count.has_key(r):
				count[r] = 1
			else:
				count[r] += 1

	count_sort = sorted(count.items(), key=lambda x:x[1], reverse=True)
	vocab = ['<s>', 'UNK']
	ivocab = {'UNK': 1, '<s>': 0}
	tsg_vocab = [None, None]
	vnum = 2
	while vnum < num:
		if vnum-2 >= len(count_sort):
			break
		r = count_sort[vnum-2][0]
		vocab.append(r)
		ivocab[count_sort[vnum-2][0]] = vnum 
		tsg_vocab.append(tsg_dict[r])
		vnum += 1
	with open(vocab_f, 'wb') as f:
		cPickle.dump(vocab, f)
	with open(ivocab_f, 'wb') as f:
		cPickle.dump(ivocab, f)

	# Get TSG object
	T = TSG(tsg_vocab, vocab, ivocab)
	with open(tsg_obj_f, 'wb') as f:
		cPickle.dump(T, f)

	# index the corpus
	print "Indexing the rules"
	idx_rule_seqs = []
	num_trees = 0
	UNKs = 0
	num_rules = 0

	for rule_seq in rule_seqs:
		idx_rule_seqs.append([])
		rule_time_dict = {}
		for t, tsg in enumerate(rule_seq):
			r = str(tsg)
			rule_time_dict[tsg] = t+1
			if not ivocab.has_key(r):
				idx_rule_seqs[num_trees].append([1, 1, 0]) #[rule_idx, parent_rule_idx, parent_time]
				UNKs += 1
			else:
				# rule index
				cy = [ivocab[r]]
				# parent rule index
				if not ivocab.has_key(str(tsg.parent)):
					cy += [1]
				else:
					cy += [ivocab[str(tsg.parent)]]
				# parent time step
				if not rule_time_dict.has_key(tsg.parent):
					cy += [0]
				else:
					cy += [rule_time_dict[tsg.parent]]

				idx_rule_seqs[num_trees].append(cy)
			num_rules += 1
		num_trees += 1

	if data_corpus == 'cPickle':
		with open(idx_rule_f, 'wb') as f:
			cPickle.dump(idx_rule_seqs, f)
	elif data_corpus == 'json':
		with open(idx_rule_f, 'wb') as f:
			json.dump(idx_rule_seqs, f)




if __name__ == "__main__":
	source_file = '../data/source'
	tsg_file = '../data/tsg'
	corpus_dir = '../corpus/'
	if not os.path.exists(corpus_dir):
		os.makedirs(corpus_dir)
	# number is eos index
	preprocess_tsg(15, 
					corpus_dir+'vocab.'+os.path.basename(tsg_file) + '.pkl',
					corpus_dir +'ivocab.'+os.path.basename(tsg_file) + '.pkl',
					tsg_file, 
					corpus_dir+os.path.basename(tsg_file)+'.json', 
					corpus_dir+'tsg.'+os.path.basename(tsg_file)+'.pkl')
	preprocess(15, corpus_dir+'vocab.'+os.path.basename(source_file) + '.pkl', \
					corpus_dir +'ivocab.'+os.path.basename(source_file) + '.pkl', \
					source_file, corpus_dir+os.path.basename(source_file)+'.json')
	shuffle(corpus_dir+os.path.basename(source_file)+'.json', corpus_dir+os.path.basename(tsg_file)+'.json',\
			corpus_dir+os.path.basename(source_file)+'.json.shuf', corpus_dir+os.path.basename(tsg_file)+'.json.shuf')
