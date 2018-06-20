import sys

from nltk import tokenize

sys.path.append('../')
from sklearn.externals import joblib
import string
import numpy
import logging

numpy.random.seed(1337)
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

stoplist = stopwords.words('english')

logging.basicConfig(filename='../log.txt', format='%(levelname)s : %(message)s', level=logging.DEBUG)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)


class LDA_trainer():

    def __init__(self):
        self.documents = list()
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = set(string.punctuation)
        self.load_stopwords()

    def load_stopwords(self):
        self.stoplist = set(stopwords.words('english'))
        self.stoplist.update(
            "i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their" \
            " theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the" \
            " and but if or because as until while of at by for with about against between into through during before after above below to from up down in" \
            " out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only" \
            " own same so than too very s t can will just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn" \
            " needn shan shouldn wasn weren won wouldn".strip().split())

    def filter_data(self, text):

        text = text.split(' ')

        # lemmatize the corpus
        lemmas = [self.lemmatizer.lemmatize(word) for word in text]
        # print(lemmas)

        # cleaning stopwords
        lemmas = [token for token in lemmas if token not in self.stoplist]
        # cleaning punctuation
        lemmas = [token for token in lemmas if token not in self.punctuation]
        # print('line::', ' '.join(lemmas))

        return ' '.join(lemmas)

    def prepare_LDA_corpus(self, filepath='./ruwiki-latest-pages-articles.xml.bz2', no_features=50000):
        documents = []

        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # if tab separated
                line = line.replace('\t', ' ')

                documents.append(self.filter_data(line.strip()))

        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=1, max_features=no_features, stop_words=self.stoplist)
        self.tf = tf_vectorizer.fit_transform(documents)
        self.tf_feature_names = tf_vectorizer.get_feature_names()

    def train_LDA(self, no_topics=20, model_file='./persona_lda.model'):
        # Run LDA
        lda = LatentDirichletAllocation(n_components=no_topics, max_iter=10, learning_method='online',
                                        learning_offset=50.,
                                        random_state=0).fit(self.tf)

        joblib.dump(lda, model_file)

    def load_lda_model(self, filename='./persona_lda.model'):
        self.lda_model = joblib.load(filename)

    def display_topics(self, no_top_words):
        for topic_idx, topic in enumerate(self.lda_model.components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([self.tf_feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


train_file = 'persona_with_role.txt'
test_file = ''

lt = LDA_trainer()
lt.prepare_LDA_corpus(train_file)
lt.train_LDA(model_file='personas_lda_wiki.model')
lt.load_lda_model(filename='personas_lda_wiki.model')
lt.display_topics(10)

# lt.prepare_wiki_corpus(filepath='../resource/enwiki-latest-pages-articles.xml.bz2',output_file='../resource/LDA/wiki_en_bow.mm')
# lt.train_LDA(filepath='../resource/LDA/wiki_en_bow.mm', model_file='../resource/LDA/model/en_lda_wiki.model')

# lt.read_documents(test_file)
# lt.prepare_documents(path='../resource/extracted/**/wiki*')
# lt.load_lda_model(filename='../resource/LDA/model/ru_wiki_lda.model')
