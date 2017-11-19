import json
import fightin_words.fightin_words as fw
import sklearn.feature_extraction.text as sk_text
import logging
from nltk.stem.porter import PorterStemmer
import nltk
from wordfreq import word_frequency
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pprint import pprint

class Debate:

    def __init__(self, jsonfile):
        with open(jsonfile,'r') as f:
            self.debates = json.load(f);


    def talking_points(self,debate_id, k):

        debate = self.debates[debate_id]
        #logging.debug(debate['transcript'])
        for_text = []
        against_text =[]
        for t in debate['transcript']:
            #logging.debug(t)
            if t['segment'] == 0 and t['speakertype'] == 'for':
                for_text.extend(t['paragraphs'])
            elif t['segment'] == 0 and t['speakertype'] == 'against':
                against_text.extend(t['paragraphs'])
            else:
                pass

        for_text = ' '.join(for_text)
        against_text = ' '.join(against_text)
        logging.debug(for_text)
        stemmer = PorterStemmer()
        analyzer = sk_text.CountVectorizer().build_analyzer()
        lemmatizer = WordNetLemmatizer()
        def prep_words(doc):
            #is_noun = lambda pos: pos[:2] == 'NN'
            #tokenized = nltk.word_tokenize(doc)
            tokenized = analyzer(doc)
            #nouns = [word for (word, pos) in nltk.pos_tag(analyzer(doc)) if is_noun(pos)]
            #logging.debug(nouns)
            stopWords = set(stopwords.words('english'))
            return (stemmer.stem(w) for w in tokenized if w not in stopWords)

        prior = 0.01;
        cv = sk_text.CountVectorizer(stop_words='english', tokenizer = prep_words)
        #counter = sk_text.CountVectorizer(stop_words='english')
        #counter.fit_transform([' '.join(for_text) + ' ' + ' '.join(against_text)]).toarray()
        #prior = ( word_frequency(w,'en',minimum=1/len(counter.vocabulary_)) for w in counter.vocabulary_.keys())


        words = fw.FWExtractor(prior,cv).fit_transform([against_text, for_text])
        #words = sorted(words,key=lambda x: x[1])

        pos_filter = lambda pos: pos[:2] in ['NN','JJ','VB']
        tokenized = analyzer(for_text + against_text)
        logging.debug(nltk.pos_tag(tokenized))
        nouns = [stemmer.stem(word) for (word, pos) in nltk.pos_tag(tokenized) if pos_filter(pos)]

        noun_words = [w for (w,f) in words if w in nouns]
        words = [w for (w,f) in words]
        print(words)
        logging.debug(noun_words[-20:])
        logging.debug(words[-20:])

        indexNoFault = lambda x,list:list.index(x) if x in list else -1
        negIndexNoFault = lambda x,list:list.index(x) - len(list) if x in list else -1

        logging.debug('debt:{} {}'.format(indexNoFault('debt',noun_words),indexNoFault('debt',words)))
        logging.debug('boomer:{} {}'.format(indexNoFault('boomer',noun_words),indexNoFault('boomer',words)))
        logging.debug('colleg:{} {}'.format(indexNoFault('colleg',noun_words),indexNoFault('colleg',words)))
        logging.debug('realiti:{} {}'.format(indexNoFault('realiti',noun_words),indexNoFault('realiti',words)))

        logging.debug(noun_words[:20])
        logging.debug(words[:20])

        logging.debug('econom:{} {}'.format(negIndexNoFault('econom',noun_words),negIndexNoFault('econom',words)))
        logging.debug('volunt:{} {}'.format(negIndexNoFault('volunt',noun_words),negIndexNoFault('volunt',words)))
        logging.debug('home:{} {}'.format(negIndexNoFault('home',noun_words),negIndexNoFault('home',words)))
        logging.debug('engag:{} {}'.format(negIndexNoFault('engag',noun_words),negIndexNoFault('engag',words)))


        return noun_words[int(-k):],noun_words[:int(k)]



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    debate = Debate('iq2_data_release/iq2_data_release.json')
    [For,against] = debate.talking_points('040914%20Millennials',20)

    print(For)
    print(against)
    #print(debate["title"])
