import os
import glob
import sys
import re

import pickle

p = re.compile('itemid=\"([0-9]+)\"')
processed_corpus_path = "/home/thomas/projects/clms/internship/corpora/reuters/rcv1v2-ids.dat"
corpus_path = "/corpora/reuters"
corpus_index_cache = '/home/thomas/projects/clms/internship/lda/cache/reuters/reuters_index.pickle'


class ReutersIndex:
    def __init__(self):
        self.index = {}

    def get_id(self, file_name):
        for line in open(file_name, encoding="ISO-8859-1"):
            search_results = re.search(p, line)
            if search_results is not None:
                id = search_results.group(1)
                if id is not None:
                    print(id)
                    try:
                        self.index[int(id)] = file_name
                    except:
                        pass

    def load_id_list(self):
        for line in open(processed_corpus_path):
            id = int(line)
            self.index[id] = ""

    def walk_corpus(self):
        i = 0
        for root, dirs, files in os.walk(corpus_path):
            for file in files:
                if file.endswith(".xml"):
                    file_name = os.path.join(root, file)
                    print(file_name)
                    self.get_id(file_name)
                    i += 1
                # if i > 500:
                #	return

    def print_dictionary(self):
        num_found = 0
        num_not_found = 0
        print("Dictionary: ")
        for id in self.index:
            if self.index[id] != "":
                num_found += 1
                print(str(id) + ",  " + self.index[id])
            else:
                num_not_found += 1
        print("found: " + str(num_found) + ", not_found:" + str(num_not_found))

    def save_dict(self):
        with open(corpus_index_cache, 'wb') as picklefile:
            pickle.dump(self.index, picklefile, protocol=pickle.HIGHEST_PROTOCOL)

    def load_dict(self):
        with open(corpus_index_cache, 'rb') as picklefile:
            self.index = pickle.load(picklefile)


if __name__ == '__main__':
    # generate index
    print("generating index for reuters data")
    indexer = ReutersIndex()
    indexer.load_id_list()
    indexer.walk_corpus()
    indexer.print_dictionary()

    print("testing..")
    indexer.save_dict()
    indexer.index = {}
    indexer.load_dict()
    indexer.print_dictionary()
