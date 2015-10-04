__author__ = 'thomas'

import gensim
from gensim import corpora
from nltk.corpus import brown
from nltk.corpus import stopwords

stoplist = stopwords.words('english')

# staticdocs = ["Apple is releasing a new product",
#              "Amazon sells many things",
#              "Microsoft announces Nokia acquisition"]

wordCorpusCounts = dict()
wordIndices = dict()
index = 0
id2Word = dict()


def addWordToGlobalWordCountDict(word):
    global index
    try:
        wordCorpusCounts[word] += 1
    except KeyError:
        wordCorpusCounts[word] = 1
        index += 1
        wordIndices[word] = index
        id2Word[index] = word

def addWordToWordCountDict(word, dict):
    if word not in stoplist:
        addWordToGlobalWordCountDict(word)
        try:
            dict[word] += 1
        except KeyError:
            dict[word] = 1

def get_docs():
    docDictList = []
    i = 0
    for fileid in brown.fileids():
        wordCountDict = dict()
        for word in brown.words(fileid):
            addWordToWordCountDict(word, wordCountDict)
        docDictList.append(wordCountDict)
        if i == 100:
            break
        else:
            i += 1

    # now build the bow vectors
    bowVectors = list()
    for docDict in docDictList:
        bowVector = list()
        for word in docDict.keys():
            index = wordIndices[word]
            count = docDict[word]
            bowVector.append((index, count))
        bowVectors.append(bowVector)
    return bowVectors

#corpus = get_docs()

def get_texts():
    texts = []
    i = 0
    for fileid in brown.fileids():
        texts.append(brown.words(fileid))
        if i == 100:
            break
        else:
            i += 1
    return texts

texts = get_texts()
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes()
corpus = [dictionary.doc2bow(text) for text in texts]

#texts2 = [[word for word in document.lower().split() if word not in stoplist] for document in staticdocs]
#dictionary2 = corpora.Dictionary(texts2)
#corpus2 = [dictionary2.doc2bow(text) for text in texts2]

#dictionary = gensim.corpora.Dictionary.from_corpus(corpus, id2Word)
#dictionary.filter_extremes()

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, update_every=1, chunksize=10000, passes=1)
topics = lda.print_topics()
for topic in topics:
    print topic
