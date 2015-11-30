__author__ = 'thomas'

import numpy as np
import gensim


class DynamicLda:
    def __init__(self, lda, bows):
        self.lda = lda
        self.topics = lda.lda_model.state.get_lambda()
        self.lda_model = lda.lda_model
        self.bows = bows
        self.topic_threshold = 1.0/self.lda_model.num_topics

    def get_topic_distributions(self, ids):
        distributions = np.ndarray(shape=(len(ids), len(self.topics)), dtype=float)
        i = 0
        for id in ids:
            bow = self.bows[id]
            topics = self.lda_model.get_document_topics(bow)
            full_topics = gensim.matutils.sparse2full(topics, self.lda_model.num_topics)
            full_topics[full_topics <= self.topic_threshold] = 0.0
            full_topics[full_topics > self.topic_threshold] = 1.0
            distributions[i] = full_topics
            i += 1
        if i == 0:
            distributions = np.array([np.zeros(len(self.topics))])
        return distributions

    def get_topic_bias(self, good_docs, bad_docs):
        good_dist = self.get_topic_distributions(good_docs)
        bad_dist = self.get_topic_distributions(bad_docs)*(-1)
        total_dist = good_dist + bad_dist
        summed_dist = total_dist.sum(axis=0)  # sum over the columns
        summed_dist[summed_dist > 0] = self.topic_threshold
        summed_dist[summed_dist < 0] = -self.topic_threshold
        return summed_dist

    def get_biased_topic_distribution(self, query_terms, good_docs, bad_docs, good_terms, bad_terms):

        bias = self.get_topic_bias(good_docs, bad_docs)
        vec_bow = self.lda_model.id2word.doc2bow(query_terms)
        vec_topic_distribution = self.lda_model[vec_bow]
        biased_topic_distribution = np.zeros(len(vec_topic_distribution))
        for topicTuple in vec_topic_distribution:
            index = topicTuple[0]
            probability = topicTuple[1]
            biased_probability = probability + bias[index]
            biased_topic_distribution[index] = biased_probability
        biased_topic_distribution[biased_topic_distribution < 0] = 0
        # renormalize:
        biased_topic_distribution /= sum(biased_topic_distribution)
        return vec_topic_distribution



