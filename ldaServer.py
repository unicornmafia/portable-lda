__author__ = 'thomas'

from flask import Flask

from flask import request
import os
from corpus.bowBuilder import BowBuilder
from lda.ldaCalc import LdaCalc
import json
from flask.ext.cors import CORS
import gensim

from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper



app = Flask(__name__)
CORS(app)
counter = 0
bow_vectors = None
lda_model = None
bow_builder = None


@app.route('/')
def health():
    return 'Lda Server is totally working, you guys!'


@app.route('/get-term-sim', methods=['POST'])
def get_term_sim():
    request_data = request.data
    json_data = json.loads(request_data)
    terms = json_data["terms"]
    sim_method = json_data["simmethod"]
    print "Terms: " + str(terms)
    if request.method == 'POST':
        pass
    else:
        pass
    vec_bow = lda_model.lda_model.id2word.doc2bow(terms)
    vec_lda = lda_model.lda_model[vec_bow]
    sims = lda_model.calc_sims_for_topic_distribution(vec_lda, sim_method)
    topn = sims[:20]
    topn_json = json.dumps(topn)
    print str(topn_json)
    return topn_json

if __name__ == '__main__':

    print "Loading BoW Vectors from cache...\n"
    cache_root = os.path.join(os.getcwd(), "cache")
    bow_builder = BowBuilder(cache_dir=cache_root)
    bow_builder.load()
    bow_vectors = bow_builder.bowVectorCorpus

    print "Loading LDA model...\n"
    lda_model = LdaCalc(bows=bow_vectors, cache_dir=cache_root)
    lda_model.load()
    lda_model.print_topics()

    print "\nStarting Server..."
    app.debug = True
    app.run(host="127.0.0.1", port=1338)
