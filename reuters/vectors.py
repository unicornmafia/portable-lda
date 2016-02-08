__author__ = 'thomas'
import os
try:
   import cPickle as pickle
except:
   import pickle


class Vectors:
    def __init__(self, filename, max_doc_num=None):
        self.filename = filename
        self.vectors = {}  # going to be {id:vector}, vector = [(wid,tf(log, cos normalized))...]
        self.max_doc_num = max_doc_num

    def load_from_text(self):
        doc_counter = 0
        for line in open(self.filename):
            linein = line.split(" ", 1)
            doc_id = linein[0]
            raw_vector = linein[1].split()
            vector = []
            for pair in raw_vector:
                split_pair = pair.split(":")
                vector.append((int(split_pair[0]), float(split_pair[1])))
            if self.max_doc_num is None or self.max_doc_num > doc_counter:
                self.vectors[doc_id] = vector
                doc_counter += 1
            else:
                return

    def load_from_cache(self, picklepath):
        vectorsfile = os.path.join(picklepath, "vectors")
        self.vectors = pickle.load(open(vectorsfile, "rb"))

    def save_to_cache(self, picklepath):
        vectorsfile = os.path.join(picklepath, "vectors")
        pickle.dump(self.vectors, open(vectorsfile, "wb"))


