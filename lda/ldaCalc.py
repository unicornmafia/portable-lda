__author__ = 'thomas'

import gensim
import os
import numpy as np
import operator
import itertools

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
    def __init__(self, bows=None, id2word=None, cache_dir=""):
        self.bows = bows  # {fileid:bowvector}
        self.bow_vector_list = bows.values()
        self.id2word = id2word
        self.cache_dir = os.path.join(cache_dir, cache_subdir)
        self.model_file_name = os.path.join(self.cache_dir, model_file_name)
        self.sims_cache_dir = os.path.join(cache_dir, sims_cache_subdir)
        self.cos_sims_file_name = os.path.join(self.sims_cache_dir, cos_sims_file_name)
        self.hellinger_sims_file_name = os.path.join(self.sims_cache_dir, hellinger_sims_file_name)
        self.lda_model = None
        self.num_topics = 20
        self.cos_sims = dict()
        self.hellinger_sims = dict()
        self.top_N = 20
        self.cos_sim_threshold = 0.98
        self.hell_sim_threshold = 0.8
        self.threshold = True

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
        if self.threshold:
            sorted_sims = sorted(sims.items(), key=operator.itemgetter(1), reverse=True)
        else: #  if we're not using threshold, then we're using topN
            sorted_sims = sorted(sims.items(), key=operator.itemgetter(1), reverse=True)[:self.top_N]
        return sorted_sims

    def calc_sim(self, starting_topic_id):
        #sims = list()
        starting_id = starting_topic_id
        starting_topics = self.lda_model.get_document_topics(self.bows[starting_id])

        cos_sim = {}
        hell_sim = {}
        sorted_cos_sim = []
        sorted_hell_sim = []
        for comp_id in self.bows.keys():
            if comp_id == starting_id:
                continue
            #if comp_id != starting_id:
            comp_topics = self.lda_model.get_document_topics(self.bows[comp_id])
            hellsim = self.get_sim_hellinger(starting_topics, comp_topics)
            cossim = self.get_sim_cos(starting_topics, comp_topics)
            #sims.append(sim)
            #print "Comparing " + str(starting_id) + " to " + str(comp_id) + " = " + str(hellsim) + ", " + str(cossim)
            if (not self.threshold) or self.hell_sim_threshold <= hellsim:
                hell_sim[comp_id] = hellsim
            if (not self.threshold) or self.cos_sim_threshold <= cossim:
                cos_sim[comp_id] = cossim
            sorted_hell_sim = self.get_sims_sorted(hell_sim)
            sorted_cos_sim = sorted(cos_sim.items(), key=operator.itemgetter(1), reverse=True)[:self.top_N]
        print "\n TOP " + str(self.top_N) + " SIMILAR DOCUMENTS TO " + starting_id
        for sim in sorted_hell_sim:
            print "hell:" + sim[0] + " = " + str(sim[1])
        for sim in sorted_cos_sim:
            print "cos: " + sim[0] + " = " + str(sim[1])
        self.cos_sims[starting_topic_id] = sorted_cos_sim
        self.hellinger_sims[starting_topic_id] = sorted_hell_sim

    def calc_sims(self):
        i = 0
        for fileid in self.bows.keys():
            i += 1
            print "\n########################################################################################"
            print "#  calcluating similarities for fileid, (" + str(i) + " of " + str(len(self.bows)) + ") "
            print "########################################################################################\n"
            self.calc_sim(fileid)

    def save_sims(self):
        pickle.dump(self.cos_sims, open(self.cos_sims_file_name, "wb"))
        pickle.dump(self.hellinger_sims, open(self.hellinger_sims_file_name, "wb"))

    def load_sims(self):
        self.cos_sims = pickle.load(open(self.cos_sims_file_name, "rb"))
        self.hellinger_sims = pickle.load(open(self.hellinger_sims_file_name, "rb"))

