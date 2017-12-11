import json
import fightin_words as fw
import sklearn.feature_extraction.text as sk_text
import logging
from nltk.stem.porter import PorterStemmer
import nltk
from wordfreq import word_frequency
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import linear_model
import numpy as np


def custom_tokenizer(doc):
    analyzer = sk_text.CountVectorizer().build_analyzer()
    tokenized = analyzer(doc)
    stemmer = PorterStemmer()
    #nouns = [word for (word, pos) in nltk.pos_tag(analyzer(doc)) if is_noun(pos)]
    #logging.debug(nouns)
    stopWords = set(stopwords.words('english'))
    return (stemmer.stem(w) for w in tokenized if w not in stopWords)

#countVectorizer = sk_text.CountVectorizer(stop_words='english',tokenizer=custom_tokenizer)

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
        #logging.debug(for_text)
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
        cv = sk_text.CountVectorizer(stop_words='english', tokenizer = custom_tokenizer)
        #counter = sk_text.CountVectorizer(stop_words='english')
        #cv.fit_transform([for_text,against_text])
        #prior = list( word_frequency(w,'en',minimum=1e-8) for w in cv.vocabulary_.keys())


        words = fw.FWExtractor(prior,cv).fit_transform([for_text, against_text])
        #words = sorted(words,key=lambda x: x[1])

        pos_filter = lambda pos: pos[:2] in ['NN','JJ','VB'] and pos not in ['NNP','NNPS']
        tokenized = nltk.word_tokenize(for_text + against_text)
        #logging.debug(tokenized)
        #logging.debug(nltk.pos_tag(tokenized))
        nouns = [stemmer.stem(word.lower()) for (word, pos) in nltk.pos_tag(tokenized) if pos_filter(pos)]
        #logging.debug(nouns)

        noun_words = [w for (w,f) in words if w in nouns]
        words = [w for (w,f) in words]
        #print(words)
        #logging.debug(noun_words[-20:])
        #logging.debug(words[-20:])

        indexNoFault = lambda x,list:list.index(x) if x in list else -1
        negIndexNoFault = lambda x,list:list.index(x) - len(list) if x in list else -1

        #logging.debug('debt:{} {}'.format(negIndexNoFault('debt',noun_words),negIndexNoFault('debt',words)))
        #logging.debug('boomer:{} {}'.format(negIndexNoFault('boomer',noun_words),negIndexNoFault('boomer',words)))
        #logging.debug('colleg:{} {}'.format(negIndexNoFault('colleg',noun_words),negIndexNoFault('colleg',words)))
        #logging.debug('realiti:{} {}'.format(negIndexNoFault('realiti',noun_words),negIndexNoFault('realiti',words)))

        #logging.debug(noun_words[:20])
        #logging.debug(words[:20])

        #logging.debug('econom:{} {}'.format(indexNoFault('econom',noun_words),indexNoFault('econom',words)))
        #logging.debug('volunt:{} {}'.format(indexNoFault('volunt',noun_words),indexNoFault('volunt',words)))
        #logging.debug('home:{} {}'.format(indexNoFault('home',noun_words),indexNoFault('home',words)))
        #logging.debug('engag:{} {}'.format(indexNoFault('engag',noun_words),indexNoFault('engag',words)))


        return noun_words[int(-k):],noun_words[:int(k)]

    def winner(self,debate_id):
        results = self.debates[debate_id]['results']
        pre_results = results['pre']
        post_results = results['post']
        vote_diff_for = post_results['for'] - pre_results['against']
        vote_diff_against = post_results['against'] - pre_results['against']
        if(vote_diff_for > vote_diff_against):
            return 'for'
        else:
            return 'against'



class FeatureExporter:

    def __init__(self,debate):
        self.debate = debate
        self.debate_ids = list(debate.debates.keys())


    # see Conversational Flow in Oxford-style Debates section 3
    def coverage_for_id(self,debate_id, round,sideX,sideY):
        textY = []
        for t in debate.debates[debate_id]['transcript']:
            if t['segment'] == round and t['speakertype'] == sideY:
                textY.extend(t['paragraphs'])
        textY = ' '.join(textY)

        cv = sk_text.CountVectorizer(stop_words='english',tokenizer=custom_tokenizer)
        cv.fit_transform([textY])
        textY_words = cv.vocabulary_.items()
        [forTP,againstTP] = debate.talking_points(debate_id,30)

        if(sideX == 'for'):
            tp = forTP
        elif(sideX == 'against'):
            tp = againstTP
        else:
            pass
        #print(tp)
        #print(textY_words)
        common_words = list(w for w,c in textY_words if w in tp)
        #print(common_words)

        # tokenize
        # get talking points for X
        # get fraction
        return [len(common_words)/len(tp),common_words]

    def coverage(self,round,sideX,sideY):
        result = []
        for debate_id in self.debate.debates.keys():
            [c,l] = self.coverage_for_id(debate_id,round,sideX,sideY)
            result.append(c)
        return np.asarray(result,dtype=np.double)


    def discussionpoints_for_id(self,debate_id,side):
        opponent = 'against'
        if(side == 'against'):
            opponent = 'for'
        discuss_words = []
        discuss_words_opoonent = []
        introduc_words = []
        for t in debate.debates[debate_id]['transcript']:
            if t['segment'] == 0 and t['speakertype'] == side:
                introduc_words.extend(t['paragraphs'])
            if t['segment'] == 1 and t['speakertype'] == side:
                discuss_words.extend(t['paragraphs'])
            if t['segment'] == 1 and t['speakertype'] == opponent:
                discuss_words_opoonent.extend(t['paragraphs'])
        introduc_words = ' '.join(introduc_words)
        discuss_words = ' '.join(discuss_words)
        discuss_words_opoonent = ' '.join(discuss_words_opoonent)
        cv1 = sk_text.CountVectorizer(stop_words='english',tokenizer=custom_tokenizer)
        cv2 = sk_text.CountVectorizer(stop_words='english',tokenizer=custom_tokenizer)
        cv3 = sk_text.CountVectorizer(stop_words='english',tokenizer=custom_tokenizer)
        cv1.fit_transform([introduc_words])
        introduc_words = cv1.vocabulary_.keys()
        cv2.fit_transform([discuss_words])
        discuss_words = cv2.vocabulary_.keys()
        cv3.fit_transform([discuss_words_opoonent])
        discuss_words_opoonent = cv3.vocabulary_.items()

        #print(discuss_words_opoonent)

        new_words = list(w for w in discuss_words if w not in introduc_words )
        #print(new_words)
        discuss_points = list(w for w, c in discuss_words_opoonent if w in new_words and c > 1)
        #print(len(discuss_points))
        #print(discuss_words_opoonent)

        return discuss_points

    def discussion_points(self,side):
        result = []
        for debate_id in self.debate.debates.keys():
            d = self.discussionpoints_for_id(debate_id,side)
            result.append(len(d))

        return np.asarray(result, dtype=np.double)

    def winner_labels(self):
        result = []
        for debate_id in self.debate.debates.keys():
            winner = self.debate.winner(debate_id)
            result.append(winner == 'for')

        return np.asarray(result, dtype=np.int)

class winner_predictor:
    def __init__(self,debate):
        self.debate = debate
        self.featureExporter = FeatureExporter(debate)
        self.nsamples = len(debate.debates.keys())
        logging.debug(self.nsamples)
        self.nfeatures = 12
        self.X = np.zeros([self.nsamples,self.nfeatures])
        self.Y = np.zeros(self.nsamples)


    def produce_features(self,feauture_outfile, label_outfile):
        selfCoverages_for = self.featureExporter.coverage(1,'for','for')
        logging.debug('#')
        selfCoverages_against = self.featureExporter.coverage(1,'against','against')
        logging.debug('#')

        opponentCoverages_for = self.featureExporter.coverage(1,'for','against')
        logging.debug('#')

        opponentCoverages_against = self.featureExporter.coverage(1,'against','for')
        logging.debug('#')

        intro_selfCoverages_for = self.featureExporter.coverage(0,'for','for')
        intro_selfCoverages_against = self.featureExporter.coverage(0,'against','against')
        intro_opponentCoverages_for = self.featureExporter.coverage(0,'for','against')
        intro_opponentCoverages_against = self.featureExporter.coverage(0,'against','for')
        logging.debug('#')

        discusspoints_for = self.featureExporter.discussion_points('for')
        discusspoints_against = self.featureExporter.discussion_points('against')
        logging.debug('#')


        self.X[:,0] = selfCoverages_for
        self.X[:,1] = selfCoverages_against

        self.X[:,2] = opponentCoverages_for
        self.X[:,3] = opponentCoverages_against

        self.X[:,4] = opponentCoverages_for + selfCoverages_for
        self.X[:,5] = opponentCoverages_against + selfCoverages_against

        self.X[:,6] = intro_selfCoverages_for - selfCoverages_for
        self.X[:,7] = intro_selfCoverages_against - selfCoverages_against

        self.X[:,8] = intro_opponentCoverages_for - opponentCoverages_for
        self.X[:,9] = intro_opponentCoverages_against - opponentCoverages_against

        self.X[:,10] = discusspoints_for
        self.X[:,11] = discusspoints_against
        self.Y = self.featureExporter.winner_labels()

        self.X.tofile(feauture_outfile, ',', ' %.4f')
        self.Y.tofile(label_outfile, ',', ' %d')

    def load_features(self,feature_file,label_file):
        self.X = np.fromfile(feature_file,dtype=float,sep=',')
        self.X.resize(self.nsamples,self.nfeatures)
        self.Y = np.fromfile(label_file,dtype=int,sep=',')
        self.Y.resize(self.nsamples)

    def logistic_regression(self):
        model = linear_model.LogisticRegression(penalty='l2')
        return model

    def loo_trainAndPredict(self,debate_id):

        X = self.X[[range(0,i),range(i+1,self.nsamples)],:]
        Y = self.Y[[range(0,i),range(i+1,self.nsamples)]]

        #classifier.fit(X,Y)
        #predictions[i] = classifier.predict(self.X[i,:])
    def loocv(self,classifier):

        predictions = np.zeros(self.nsamples,dtype=np.int)
        for i in range(0,self.nsamples-1):
            logging.debug(i)
            X = np.zeros([self.nsamples-1,self.nfeatures],dtype=np.double)
            Y = np.zeros(self.nsamples-1,dtype=np.int)

            X[0:i] = self.X[0:i]
            X[i:] = self.X[i+1:]
            Y[0:i] = self.Y[0:i]
            Y[i:] = self.Y[i+1:]

            classifier.fit(X,Y)
            logging.debug(classifier.score(X,Y))
            sample = self.X[i,:].reshape(1,-1)
            predictions[i] = classifier.predict(sample)
            logging.debug(predictions[i])

        accuracy = np.sum(predictions == self.Y)/self.nsamples
        return accuracy

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    debate = Debate('iq2_data_release/iq2_data_release.json')
    [For,against] = debate.talking_points('afghanistan-lost-cause',30)

    print('for talking points:')
    print(For)
    print('against talking points:')
    print(against)

    fexporter = FeatureExporter(debate)

    #print(fexporter.coverage_for_id('040914%20Millennials',0,'for','for'))
    #print(fexporter.coverage_for_id('040914%20Millennials',1,'for','against'))

    #print(fexporter.coverage(1,'for','for'))
    #print(fexporter.discussionpoints_for_id('040914%20Millennials','for'))
    #print(fexporter.discussion_points('for'))
    #print(debate["title"])
    predictor = winner_predictor(debate)
    #predictor.produce_features('features.txt','labels.txt')
    predictor.load_features('features.txt','labels.txt')
    print(predictor.loocv(predictor.logistic_regression()))
