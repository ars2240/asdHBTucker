# Modified from https://github.com/kaize0409/HyperGAT_TextClassification

from r8_utils import show_statisctic, clean_document, clean_str
from collections import Counter
import numpy as np
import pickle
from nltk import tokenize
from sklearn.utils import class_weight


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


def read_file(dataset, LDA=True):
	
	doc_content_list = []
	doc_sentence_list = []
	f = open('data/' + dataset + '_corpus.txt', 'rb')

	for line in f.readlines():
		doc_content_list.append(line.strip().decode('latin1'))
		doc_sentence_list.append(tokenize.sent_tokenize(clean_str(doc_content_list[-1])))
	f.close()

	doc_content_list, doc_phrase_list = clean_document(doc_sentence_list, dataset)

	max_num_sentence = show_statisctic(doc_content_list)

	doc_train_list_original = []
	doc_test_list_original = []
	labels_dic = {}
	label_count = Counter()

	i = 0
	f = open('data/' + dataset + '_labels.txt', 'r')
	lines = f.readlines()
	for line in lines:
		temp = line.strip().split("\t")
		if temp[1].find('test') != -1:
			doc_test_list_original.append((doc_content_list[i], doc_phrase_list[i], temp[2]))
		elif temp[1].find('train') != -1:
			doc_train_list_original.append((doc_content_list[i], doc_phrase_list[i], temp[2]))
		if not temp[2] in labels_dic:
			labels_dic[temp[2]] = len(labels_dic)
		label_count[temp[2]] += 1
		i += 1

	f.close()
	print(label_count)

	vocab_dic = list_dict(doc_content_list)
	phrase_dic = list_dict(doc_phrase_list)

	print('Total_number_of_words: ' + str(len(vocab_dic)))
	print('Total_number_of_phrases: ' + str(len(phrase_dic)))
	print('Total_number_of_categories: ' + str(len(labels_dic)))

	doc_train_list = []
	doc_test_list = []

	for doc, phrase, label in doc_train_list_original:
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
		doc_train_list.append((temp_doc, temp_phrase, labels_dic[label]))

	for doc, phrase, label in doc_test_list_original:
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
		doc_test_list.append((temp_doc, temp_phrase, labels_dic[label]))

	keywords_dic = {}
	if LDA:
		keywords_dic_original = pickle.load(open('data/' + dataset + '_LDA.p', "rb"))
	
		for i in keywords_dic_original:
			if i in vocab_dic:
				keywords_dic[vocab_dic[i]] = keywords_dic_original[i]

	train_set_y = [k for _, _, k in doc_train_list]
	
	class_weights = class_weight.compute_class_weight('balanced', np.unique(train_set_y), train_set_y)

	return doc_content_list, doc_train_list, doc_test_list, vocab_dic, phrase_dic, labels_dic, max_num_sentence, keywords_dic, class_weights


_, doc_train_list, _, vocab_dic, phrase_dic, _, _, _, _ = read_file('R8')

np.savetxt("r8_vocab.csv", list(vocab_dic.items()), delimiter=',', fmt='%s')
np.savetxt("r8_phrases.csv", list(phrase_dic.items()), delimiter=',', fmt='%s')

sparse = []
for i in range(len(doc_train_list)):
	for j in range(len(doc_train_list[i][0])):
		"""
		new0 = doc_train_list[i][0][j]
		u, c = np.unique(new0, return_counts=True)
		l = len(u)
		new = np.vstack(([i+1] * l, [j+1] * l, u, c)).transpose()
		"""
		new0 = [[doc_train_list[i][1][j][k], doc_train_list[i][0][j][k]] for k in range(len(doc_train_list[i][0][j]))]
		u, c = np.unique(new0, return_counts=True, axis=0)
		l = len(u)
		new = np.vstack(([i + 1] * l, u[:, 0], u[:, 1], c)).transpose()
		sparse = np.append(sparse, new, axis=0) if len(sparse) > 0 else new

np.savetxt('r8p_sparse.csv', sparse, delimiter=',')
