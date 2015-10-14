__author__ = 'thomas'

from nltk.corpus import brown
import os


class TextExtractor:
    def __init__(self, cache_dir, max_docs=None):
        self.cache_dir = os.path.join(cache_dir, "plaintext")
        self.texts = {}
        self.max_docs = max_docs

    def get_texts(self):
        i = 0
        for fileid in brown.fileids():
            self.texts[fileid] = brown.words(fileid)
            if self.max_docs is not None:
                i += 1
                if i >= self.max_docs:
                    break

    def save(self):
        for fileid in self.texts.keys():
            filename = os.path.join(self.cache_dir, fileid)
            outfile = open(filename, "w")
            outfile.write(" ".join(self.texts[fileid]))
            outfile.close()

    def load(self):
        files = os.listdir(self.cache_dir)
        for filename in files:
            fullfilename = os.path.join(self.cache_dir, filename)
            with open(fullfilename, 'r') as content_file:
                content = content_file.read()
            words = content.split()
            self.texts[filename] = words



