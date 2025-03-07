# Modified from https://github.com/kaize0409/HyperGAT_TextClassification

from r8_utils import show_statisctic, clean_document, clean_str
from collections import Counter
import numpy as np
import pandas
import pickle
from nltk import tokenize
# import time

load = True

# np.savetxt('cnnCVInd.csv', [1]*287113+[2]*13368, delimiter=',')


def timeRMG(t, head=''):
	dys = np.floor(t / 86400)
	t = t - dys * 86400
	hrs = np.floor(t / 3600)
	t = t - hrs * 3600
	mins = np.floor(t / 60)
	secs = t - mins * 60
	print(head + 'Estimated time remaining: %2i dys, %2i hrs, %2i min, %4.2f sec' % (dys, hrs, mins, secs))


def list_dict(content):
	word_freq = Counter()
	word_set = set()
	doc_freq = Counter()
	for doc_words in content:
		for words in doc_words:
			for word in words:
				word_set.add(word)
				word_freq[word] += 1
			for word in set(words):
				doc_freq[word] += 1

	vocab_dic = {}
	for i in word_set:
		if word_freq[i] > 1 and doc_freq[i] > 1:
			vocab_dic[i] = len(vocab_dic) + 1

	return vocab_dic


def read_file(dataset):

	doc_sentence_list = []

	if dataset == 'CNN':
		f = pandas.read_csv('./data/cnn_dailymail/train.csv', encoding='latin1')
		for line in f['article']:
			doc_sentence_list.append(tokenize.sent_tokenize(clean_str(line.strip())))
		train_docs = len(doc_sentence_list)
		print('Number of training docs: ' + str(train_docs))

		f = pandas.read_csv('./data/cnn_dailymail/validation.csv', encoding='latin1')
		for line in f['article']:
			doc_sentence_list.append(tokenize.sent_tokenize(clean_str(line.strip())))
		valid_docs = len(doc_sentence_list) - train_docs
		print('Number of validation docs: ' + str(valid_docs))
	else:
		raise Exception('Data set not implemented.')

	doc_content_list, doc_phrase_list = clean_document(doc_sentence_list, dataset)

	max_num_sentence = show_statisctic(doc_content_list)

	doc_train_list_original = []

	for i in range(len(doc_content_list)):
		doc_train_list_original.append((doc_content_list[i], doc_phrase_list[i]))

	vocab_dic = list_dict(doc_content_list)
	phrase_dic = list_dict(doc_phrase_list)

	print('Total_number_of_words: ' + str(len(vocab_dic)))
	print('Total_number_of_phrases: ' + str(len(phrase_dic)))

	doc_train_list = []

	for doc, phrase in doc_train_list_original:
		temp_doc, temp_phrase = [], []
		for i in range(len(doc)):
			sentence, phrases = doc[i], phrase[i]
			temp, temp_p = [], []
			for j in range(len(sentence)):
				word = sentence[j]
				p = phrases[j]
				if word in vocab_dic.keys():
					temp.append(vocab_dic[word])
				if p in phrase_dic.keys():
					temp_p.append(phrase_dic[p])
				elif None in phrase_dic.keys():
					temp_p.append(phrase_dic[None])
			temp_doc.append(temp)
			temp_phrase.append(temp_p)
		doc_train_list.append((temp_doc, temp_phrase))

	return doc_content_list, doc_train_list, vocab_dic, phrase_dic, max_num_sentence


if load:
	with open('cnn_doc_train.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
		doc_train_list = pickle.load(f)
else:
	_, doc_train_list, vocab_dic, phrase_dic, _ = read_file('CNN')
	print('Finished compiling.')

	np.savetxt("cnn_vocab.csv", list(vocab_dic.items()), delimiter=',', fmt='%s')
	np.savetxt("cnn_phrases.csv", list(phrase_dic.items()), delimiter=',', fmt='%s')
	with open('cnn_doc_train.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
		pickle.dump(doc_train_list, f)
	print('Saved dictionaries.')

sparse = []
# start = time.time()
for i in range(len(doc_train_list)):
	for j in range(len(doc_train_list[i][0])):
		new0 = [[doc_train_list[i][1][j][k], doc_train_list[i][0][j][k]] for k in range(len(doc_train_list[i][0][j]))]
		u, c = np.unique(new0, return_counts=True, axis=0)
		l = len(u)
		new = np.vstack(([i + 1] * l, u[:, 0], u[:, 1], c)).transpose()
		# sparse = np.append(sparse, new, axis=0) if len(sparse) > 0 else new
		sparse.append(new)
	"""
	if (i + 1) % 100 == 0:
		stop = time.time() - start
		tr = stop * (len(doc_train_list) - (i+1)) / (i+1)
		timeRMG(tr, head='{0} of {1} processed. '.format(i+1, len(doc_train_list)))
	"""
sparse = np.vstack(tuple(sparse))

np.savetxt('cnn_sparse.csv', sparse, delimiter=',')
