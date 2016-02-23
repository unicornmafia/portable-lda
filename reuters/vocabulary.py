__author__ = 'thomas'
import os
import pickle


class Vocabulary:
    def __init__(self, filename):
        self.filename = filename
        self.id2word = {}
        self.word2id = {}
        self.id2idf = {}

    def load_from_text(self):
        for line in open(self.filename):
            linein = line.split()
            word = linein[0]
            term_id = int(linein[1])
            idf = float(linein[2])
            self.id2word[term_id] = word
            self.word2id[word] = term_id
            self.id2idf[term_id] = idf

    def load_from_cache(self, picklepath):
        id2wordfile = os.path.join(picklepath, "id2word")
        word2idfile = os.path.join(picklepath, "word2id")
        id2idffile = os.path.join(picklepath, "id2idf")
        self.id2word = pickle.load(open(id2wordfile, "rb"))
        self.word2id = pickle.load(open(word2idfile, "rb"))
        self.id2idf = pickle.load(open(id2idffile, "rb"))

    def save_to_cache(self, picklepath):
        id2wordfile = os.path.join(picklepath, "id2word")
        word2idfile = os.path.join(picklepath, "word2id")
        id2idffile = os.path.join(picklepath, "id2idf")
        pickle.dump(self.id2word, open(id2wordfile, "wb"))
        pickle.dump(self.word2id, open(word2idfile, "wb"))
        pickle.dump(self.id2idf, open(id2idffile, "wb"))


