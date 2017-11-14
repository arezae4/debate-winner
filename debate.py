import json
import fightin_words.fightin_words as fw
import sklearn.feature_extraction.text as sk_text
import logging
from nltk.stem.snowball import EnglishStemmer
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
            if t['segment'] == 0 and t['speakertype'] == 'for':
                for_text.extend(t['paragraphs'])
            elif t['segment'] == 0 and t['speakertype'] == 'against':
                against_text.extend(t['paragraphs'])
            else:
                pass
        #logging.debug(side_text)
        stemmer = EnglishStemmer()
        analyzer = sk_text.CountVectorizer().build_analyzer()

        def stemmed_words(doc):
            return (stemmer.stem(w) for w in analyzer(doc) if w not in sk_text.ENGLISH_STOP_WORDS)

        prior = .01;
        cv = sk_text.CountVectorizer(max_features=15000,stop_words='english')
        words = fw.FWExtractor(prior,cv).fit_transform([''.join(for_text), ''.join(against_text)])
        words = sorted(words,key=lambda x: abs(x[1]))
        results = words[int(-k):]
        #results.extend(words[int(-k/2):])
        return results



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    debate = Debate('iq2_data_release/iq2_data_release.json')
    tp = debate.talking_points('040914%20Millennials',20)

    print(tp)
    #print(debate["title"])
