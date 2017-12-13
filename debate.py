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
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import mutual_info_classif,f_classif,chi2
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC

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
        vote_diff_for = post_results['for'] - pre_results['for']
        vote_diff_against = post_results['against'] - pre_results['against']
        if(vote_diff_for > vote_diff_against):
            return 'for'
        else:
            return 'against'



class FeatureExporter:

    def __init__(self,debate,k_talkingpoint):
        self.debate = debate
        self.debate_ids = list(debate.debates.keys())
        self.k_talkingpoint = k_talkingpoint

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
        [forTP,againstTP] = debate.talking_points(debate_id,self.k_talkingpoint)

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
        cv1 = sk_text.CountVectorizer(stop_words='english', tokenizer=custom_tokenizer)
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
    def __init__(self,debate,k_talkingpoint):
        self.debate = debate
        self.featureExporter = FeatureExporter(debate,k_talkingpoint)
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
        #self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0) # z-score normalization
        newFeature = np.array(np.fromfile('entropy_f.txt',dtype=float,sep=','))
        newFeature.resize(self.nsamples,1)
        # newFeature = newFeature
        print(newFeature)
        self.X = np.concatenate((self.X, newFeature), axis=1)
        self.X = (self.X - self.X.min(axis=0)) / (self.X.max(axis=0) - self.X.min(axis=0))  # min-max normalization

        logging.debug(self.X.mean(axis=0))
        logging.debug(self.X.std(axis=0))
        self.Y = np.fromfile(label_file,dtype=int,sep=',')
        self.Y.resize(self.nsamples)
        logging.debug('pos: {}'.format(np.sum(self.Y == 1)))

    def logistic_regression(self):
        Cs = list(10 ** n for n in range(-7,8))
        tuned_params = [{'C' : Cs, 'penalty':['l1'],'solver':['liblinear'],'tol':[1e-4]},
                    {'C' : Cs, 'penalty':['l2'],'solver':['lbfgs'],'tol':[1e-4]}]

        model = GridSearchCV(linear_model.LogisticRegression(),tuned_params,cv=3,scoring='accuracy',n_jobs=-1)
        #model = linear_model.LogisticRegressionCV(penalty=penalty,Cs= Cs,cv=3,n_jobs=-1,tol=1e-4,solver='saga')
        return model

    def svm(self):
        Cs = list(10 ** n for n in range(-5,5))
        tuned_params = [{'kernel':('linear', 'rbf', 'poly'), 'C':Cs}]

        model = GridSearchCV(SVC(),tuned_params,cv=3,scoring='accuracy',n_jobs=-1)
        return model


    def loocv(self,classifier):

        predictions = np.zeros(self.nsamples,dtype=np.int)
        loo = LeaveOneOut()
        loo.get_n_splits(self.X)

        for train_idx, test_idx in loo.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            Y_train, Y_test = self.Y[train_idx], self.Y[test_idx]


        #for i in range(0,self.nsamples-1):
        #    logging.debug(i)
        #    X_train = np.zeros([self.nsamples-1,self.nfeatures],dtype=np.double)
        #    Y_train = np.zeros(self.nsamples-1,dtype=np.int)

        #    X_train[0:i] = self.X[0:i]
        #    X_train[i:] = self.X[i+1:]
        #    Y_train[0:i] = self.Y[0:i]
        #    Y_train[i:] = self.Y[i+1:]

            # feature selection
            feature_selector = GenericUnivariateSelect(score_func=chi2,mode='percentile', param=80)

            logging.debug(X_train.shape)
            X_train_new = feature_selector.fit_transform(X_train,Y_train)
            logging.debug(X_train_new.shape)
            classifier.fit(X_train_new,Y_train)
            logging.debug(classifier.score(X_train_new,Y_train))
            logging.debug(classifier.best_params_)
            logging.debug(feature_selector.get_support())
            #X_test = self.X[i,feature_selector.get_support()].reshape(1,-1)
            X_test_new = X_test[0,feature_selector.get_support()].reshape(1,-1)
            predictions[test_idx] = classifier.predict(X_test_new)
            logging.debug(predictions[test_idx])

        accuracy = np.sum(predictions == self.Y)/self.nsamples
        return accuracy

class Entropy:
    def __init__(self):
        self.trigrams_count = {}
        self.bigrams_count = {}
        self.unigrams_count = {}
        self.total_unigram = 0
        self.normalizer ={}

    def prep_training(self, jsonfile, debate):
        with open(jsonfile, 'r') as f:
            self.debates = json.load(f);

        debate_ids = list(debate.debates.keys())
        all_text = ''
        for id in debate_ids:
            debate = self.debates[id]
            for_text = []
            against_text = []
            for t in debate['transcript']:
                # logging.debug(t)
                if t['segment'] == 0 and t['speakertype'] == 'for':
                    for_text.extend(t['paragraphs'])
                elif t['segment'] == 0 and t['speakertype'] == 'against':
                    against_text.extend(t['paragraphs'])
                else:
                    pass

            for_text = ' '.join(for_text)
            against_text = ' '.join(against_text)
            for_text.replace('-', ' ')
            against_text.replace('-', ' ')
            # all_text = ''.join(all_text, for_text,against_text)
            all_text += for_text
            all_text += against_text
        pat = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.M)
        all_sents = pat.findall(all_text)
        no_punc = []

        translator = str.maketrans('', '', string.punctuation)
        for sent in all_sents:
            # sent = sent.translate(translator)
            no_punc.append(sent.translate(translator))
        print(no_punc[0])
        print(all_sents[0].translate(translator))
        return no_punc


    def train(self, training):

        trigrams_count = dict()
        bigrams_count = dict()
        unigrams_count = dict()

        for sent in training:
            trigrams = nltk.trigrams(sent.split(" "))
            for trigram in trigrams:
                key = '-'.join(list(trigram))
                if key in trigrams_count:
                    trigrams_count[key] += 1
                else:
                    trigrams_count[key] = 1
            bigrams = nltk.bigrams(sent.split(" "))
            for bigram in bigrams:
                key = '-'.join(list(bigram))
                if key in bigrams_count:
                    bigrams_count[key] += 1
                else:
                    bigrams_count[key] = 1

            unigrams = sent.split(" ")
            for unigram in unigrams:
                key = unigram
                if key in unigrams_count:
                    unigrams_count[key] += 1
                else:
                    unigrams_count[key] = 1
                self.total_unigram += 1
        self.unigrams_count = unigrams_count
        self.bigrams_count = bigrams_count
        self.trigrams_count = trigrams_count

        #Normalzing the entropy of a sentence with respect to its length

        normalizer = dict()

        for sent in training:

            current_Ent =  self.sent_entropy(sent)
            key = len(sent.split(" "))
            if key in normalizer:
                normalizer[key] = (normalizer[key][0]+current_Ent, normalizer[key][1] + 1)
            else:
                normalizer[key] = (current_Ent, 1)

        self.normalizer = normalizer

    def feature_extractor(self,jsonfile, debate):

        feature = []
        debate_ids = list(debate.debates.keys())
        pat = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.M)
        translator = str.maketrans('', '', string.punctuation)
        for id in debate_ids:
            debate = self.debates[id]
            for_text = []
            against_text = []
            for t in debate['transcript']:
                # logging.debug(t)
                if t['segment'] == 0 and t['speakertype'] == 'for':
                    for_text.extend(t['paragraphs'])
                elif t['segment'] == 0 and t['speakertype'] == 'against':
                    against_text.extend(t['paragraphs'])
                else:
                    pass

            for_text = ' '.join(for_text)
            against_text = ' '.join(against_text)

            for_text.replace('-', ' ')
            against_text.replace('-', ' ')
            for_sents = pat.findall(for_text)
            against_sents = pat.findall(against_text)
            no_punc_for = []
            no_punc_against = []


            for sent in for_sents:
                no_punc_for.append(sent.translate(translator))
            for sent in against_sents:
                no_punc_against.append(sent.translate(translator))

            ent_par_for = self.para_entropy(no_punc_for)
            ent_par_against = self.para_entropy(no_punc_against)

            var_for = np.var(ent_par_for)
            var_against = np.var(ent_par_against)
            # print(ent_par_against)
            # print("var_for : ", var_for)
            # print("var_against : " , var_against)
            feature.append(var_for-var_against)

        return feature

    def para_entropy(self, para):
        ents = []
        for sent in para:
            key = len(sent.split(" "))
            ents.append(self.normalizer[key][0] / self.normalizer[key][1] * self.sent_entropy(sent))
        return ents

    def sent_entropy(self, sent):
        trigrams = list(nltk.trigrams(sent.split(" ")))
        bigrams = list(nltk.bigrams(sent.split(" ")))
        words = sent.split(" ")
        prob = math.log2(self.unigrams_count[words[0]] / self.total_unigram)
        if (len(words) > 1):
            prob += math.log2(self.bigrams_count[words[0] + "-" + words[1]] / self.unigrams_count[words[0]])
        if len(words) > 2:
            for i in range(0, len(words) - 2):
                try:
                    prob += math.log2(
                        self.trigrams_count['-'.join(trigrams[i])] / self.bigrams_count['-'.join(bigrams[i + 1])])
                except:
                    print(words)

        return (-1/len(words)) * prob



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    debate = Debate('iq2_data_release/iq2_data_release.json')
    [For,against] = debate.talking_points('afghanistan-lost-cause',20)

    print('for talking points:')
    print(For)
    print('against talking points:')
    print(against)

    #fexporter = FeatureExporter(debate,20)

    #print(fexporter.coverage_for_id('040914%20Millennials',0,'for','for'))
    #print(fexporter.coverage_for_id('040914%20Millennials',1,'for','against'))

    #print(fexporter.coverage(1,'for','for'))
    #print(fexporter.discussionpoints_for_id('040914%20Millennials','for'))
    #print(fexporter.discussion_points('for'))
    #print(debate["title"])
    predictor = winner_predictor(debate,20)
    # entropy = Entropy()
    # training = entropy.prep_training('iq2_data_release/iq2_data_release.json', debate)
    # mf = open("Temp.txt", "w")
    # for sent in training:
    #     mf.write(sent+"\n")
    # entropy.train(training)

    # print(entropy.feature_extractor('iq2_data_release/iq2_data_release.json', debate))
    # newFeature = entropy.feature_extractor('iq2_data_release/iq2_data_release.json', debate)
    #predictor.produce_features('features3.txt','labels.txt')
    predictor.load_features('features3.txt','labels.txt')
    print(predictor.loocv(predictor.logistic_regression()))
