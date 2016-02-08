__author__ = 'thomas'
import os
try:
    import cPickle as Pickle
except:
    import Pickle


class Text:
    def __init__(self, filename, vocabulary, max_doc_num=None):
        self.filename = filename
        self.text_vectors = {}  # going to be {id:vector}, vector = [word1, word2, ...]
        self.max_doc_num = max_doc_num
        self.bow_vectors = {}  # going to be {id:bow vector}, vector = [(id1, num)]
        self.vocabulary = vocabulary

    def get_bow_from_text(self, text_list):
        doc_count_dict = {}
        doc_bow = []
        for token in text_list:
            try:
                doc_count_dict[token] += 1
            except KeyError:
                doc_count_dict[token] = 1
        for token in doc_count_dict:
            try:
                word_id = self.vocabulary.word2id[token]
                count = doc_count_dict[token]
                doc_bow.append((word_id, count))
            except KeyError:
                continue
        return doc_bow

    def load_from_text(self):
        doc_counter = 0
        text_file = open(self.filename)
        for line in text_file:
            if line[:2] == ".I":
                line_in = line.split(" ", 1)

                doc_id = line_in[1].strip()
                line = text_file.next()
                if line[:2] != ".W":
                    continue

                text = text_file.next().strip()
                textlist = []
                while text != "":
                    textlist.extend(text.split())
                    text = text_file.next().strip()

                if self.max_doc_num is None or self.max_doc_num > doc_counter:
                    self.text_vectors[doc_id] = textlist
                    bow_vec = self.get_bow_from_text(textlist)
                    self.bow_vectors[doc_id] = bow_vec
                    doc_counter += 1
                else:
                    return

    def load_from_cache(self, pickle_path):
        text_vectors_file = os.path.join(pickle_path, "text_vectors")
        bow_vectors_file = os.path.join(pickle_path, "bow_vectors")
        self.text_vectors = Pickle.load(open(text_vectors_file, "rb"))
        self.bow_vectors = Pickle.load(open(bow_vectors_file, "rb"))

    def save_to_cache(self, pickle_path):
        text_vectors_file = os.path.join(pickle_path, "text_vectors")
        bow_vectors_file = os.path.join(pickle_path, "bow_vectors")
        Pickle.dump(self.text_vectors, open(text_vectors_file, "wb"))
        Pickle.dump(self.bow_vectors, open(bow_vectors_file, "wb"))



