__author__ = 'thomas'

from nltk.corpus import stopwords
import os
try:
   import cPickle as pickle
except:
   import pickle

cache_subdir = "bow"
pickle_file_name = "bow_vectors"

def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False


class BowBuilder:

    def __init__(self, docs=None, max_docs=None, cache_dir=""):
        self.bow_cache_dir = os.path.join(cache_dir, cache_subdir)
        self.pickle_file_name = os.path.join(self.bow_cache_dir, pickle_file_name)
        self.stoplist = stopwords.words('english')
        self.nDocs = 0
        self.wordCorpusCounts = dict()  # {word:count}  (over all documents)
        self.wordIndices = dict()  # {word:index}
        self.id2word = dict()  # {index:word}
        self.lowerLimit = 2
        self.upperLimitPercent = 0.8
        self.docDicts = {}  # {fileid:wordcountdict}  wordcountdict={word:count} (over 1 document)
        self.docs = docs
        self.max_docs = max_docs
        self.bowVectorCorpus = {}  # {fileid:bowvector}

    def add_word_to_global_word_count_dict(self, word):
        try:
            self.wordCorpusCounts[word] += 1
        except KeyError:
            self.wordCorpusCounts[word] = 1

    def add_word_to_word_count_dict(self, word, wcdict):
        if word not in self.stoplist:
            self.add_word_to_global_word_count_dict(word)
            try:
                wcdict[word] += 1
            except KeyError:
                wcdict[word] = 1

    def compress_word_dictionary(self):
        compressedWordCorpusCounts = dict()
        for word in self.wordCorpusCounts.keys():
            count = self.wordCorpusCounts[word]
            percentDocsWordOccursIn = float(count)/self.nDocs
            if count >= self.lowerLimit and \
                    percentDocsWordOccursIn <= self.upperLimitPercent:
                compressedWordCorpusCounts[word] = count
        self.wordCorpusCounts = compressedWordCorpusCounts

    def build_word_indices(self):
        index = 0
        for word in self.wordCorpusCounts.keys():
            self.wordIndices[word] = index
            self.id2word[index] = word
            index += 1


    def make_bow_vectors(self):
        # now build the bow vectors
        for fileid in self.docDicts.keys():
            docDict = self.docDicts[fileid]
            bowVector = list()
            for word in docDict.keys():
                if word in self.wordIndices:
                    index = self.wordIndices[word]
                    count = docDict[word]
                    bowVector.append((index, count))
            self.bowVectorCorpus[fileid] = bowVector


    def build_doc_dicts(self):
        i = 0
        for fileid in self.docs.keys():
            self.nDocs += 1
            wordCountDict = dict()
            for word in self.docs[fileid]:
                if word not in self.stoplist and \
                        not is_number(word) and \
                        word.isalpha():
                    self.add_word_to_word_count_dict(word.lower(), wordCountDict)
            self.docDicts[fileid] = wordCountDict
            if i == 500:
                break
            else:
                i += 1

    def generate_bows(self):
        # get word counts for documents
        self.build_doc_dicts()

        # compress word dictionary
        self.compress_word_dictionary()

        # build word indices
        self.build_word_indices()

        #now make bow vectors from our data
        self.make_bow_vectors()

    def save(self):
        pickle.dump(self.bowVectorCorpus, open(self.pickle_file_name, "wb"))

    def load(self):
        self.bowVectorCorpus = pickle.load(open(self.pickle_file_name, "rb"))
