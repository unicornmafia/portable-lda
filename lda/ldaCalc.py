__author__ = 'thomas'

import gensim
import os
import numpy as np
import operator
import itertools
import scipy.misc


try:
   import cPickle as pickle
except:
   import pickle

cache_subdir = "model"
model_file_name = "gensim_lda.model"
sims_cache_subdir = "sims"
cos_sims_file_name = "sims.cos"
hellinger_sims_file_name = "sims.hell"




class LdaCalc:
    def __init__(self, bows=None, id2word=None, cache_dir="", hell_threshold=0.98, cos_threshold=0.8):
        self.bows = bows  # {fileid:bowvector}
        self.bow_vector_list = bows.values()
        self.id2word = id2word
        self.cache_dir = os.path.join(cache_dir, cache_subdir)
        self.model_file_name = os.path.join(self.cache_dir, model_file_name)
        self.sims_cache_dir = os.path.join(cache_dir, sims_cache_subdir)
        self.cos_sims_file_name = os.path.join(self.sims_cache_dir, cos_sims_file_name)
        self.hell_sims_file_name = os.path.join(self.sims_cache_dir, hellinger_sims_file_name)
        self.lda_model = None
        self.num_topics = 20
        self.cos_sims = list()
        self.hell_sims = list()
        self.cos_sim_threshold = cos_threshold
        self.hell_sim_threshold = hell_threshold

    def run_lda(self):
        # get an lda-compatible dictionary
        dictionary = gensim.corpora.Dictionary.from_corpus(self.bow_vector_list, self.id2word)

        # run the lda
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=self.bow_vector_list, id2word=dictionary,
                                                         num_topics=self.num_topics, update_every=1,
                                                         chunksize=10000, passes=1)
        print "done running the lda"

    def print_topics(self):
        # print that shit!
        topics = self.lda_model.print_topics(num_topics=self.num_topics, num_words=20)
        for topic in topics:
            print topic

    def save(self):
        self.lda_model .save(self.model_file_name)

    def load(self):
        self.lda_model = gensim.models.LdaModel.load(self.model_file_name)
        print "loaded"

    def get_sim_cos(self, vec1, vec2):
        sim = gensim.matutils.cossim(vec1, vec2)
        return sim

    def get_sim_hellinger(self, vec1, vec2):
        #Hellinger distance is useful for similarity between probability distributions (such as LDA topics):
        dense1 = gensim.matutils.sparse2full(vec1, self.num_topics)
        dense2 = gensim.matutils.sparse2full(vec2, self.num_topics)
        sim = 1.0 - np.sqrt(0.5 * ((np.sqrt(dense1) - np.sqrt(dense2))**2).sum())
        return sim

    def get_sim(self, vec1, vec2):
        return self.get_sim_hellinger(vec1, vec2)

    def get_sims_sorted(self, sims):
        sorted_sims = sorted(sims, key=lambda tup: tup[2], reverse=True)
        return sorted_sims

    def calc_sim(self, topicid1, topicid2):
        if topicid1 == topicid2:
            return

        topics1 = self.lda_model.get_document_topics(self.bows[topicid1])
        topics2 = self.lda_model.get_document_topics(self.bows[topicid2])

        hellsim = self.get_sim_hellinger(topics1, topics2)
        cossim = self.get_sim_cos(topics1, topics2)

        #if self.hell_sim_threshold <= hellsim:
        self.hell_sims.append((topicid1, topicid2, hellsim))
        #if self.cos_sim_threshold <= cossim:
        self.cos_sims.append((topicid1, topicid2, cossim))
        return hellsim, cossim

    def calc_sims_for_topic_distribution(self, topics):
        sims = []
        for topicid in self.bows.keys():

            topics2 = self.lda_model.get_document_topics(self.bows[topicid])

            cossim = self.get_sim_cos(topics, topics2)
            hellsim = self.get_sim_hellinger(topics, topics2)

            sims.append((topicid, cossim, hellsim))
        sorted_sims = sorted(sims, key=lambda x: x[1], reverse=True)
        return sorted_sims

    def calc_sims(self):
        i = 0

        num_combos = scipy.misc.comb(len(self.bows.keys()), 2)
        print "########################################################################################\n"
        for fileid1, fileid2 in itertools.combinations(self.bows.keys(), 2):
            i += 1
            sims = self.calc_sim(fileid1, fileid2)
            print "(%d of %d) comparing %s and %s: hell=%f, cos=%f" \
                  % (i, num_combos, fileid1, fileid2, sims[0], sims[1])
            #if i == 100:
            #    break
        self.hell_sims = self.get_sims_sorted(self.hell_sims)
        self.cos_sims = self.get_sims_sorted(self.cos_sims)

    def save_sims(self):
        pickle.dump(self.cos_sims, open(self.cos_sims_file_name, "wb"))
        pickle.dump(self.hell_sims, open(self.hell_sims_file_name, "wb"))

    def load_sims(self):
        self.cos_sims = pickle.load(open(self.cos_sims_file_name, "rb"))
        self.hell_sims = pickle.load(open(self.hell_sims_file_name, "rb"))

