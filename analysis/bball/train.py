import sys
import re
import collections
import random
import math

import simplejson as json
import cPickle as pickle
from sklearn import cross_validation
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, Imputer
import matplotlib.pyplot as plt
import GPy
import numpy as np

from analysis.bball.playerAnalysis import *
from statsETL.db.mongolib import *

class DataPreprocessor(object):

    """
    take samples, with samples split into cont vars and cat vars
    expects cat vars to be one hot encoded to -1 or 1
    normalizes cont vars by minmax to (-1,1), then scaling to 0 mean and 1 stdvar
    concatenates cont and cat vars to product single sample list
    """

    def __init__(self, cont_labels, cont_samples, cat_labels, cat_samples, Y):
        self.cont_labels = cont_labels
        self.cat_labels = cat_labels
        self.labels = None
        self.cont_samples = cont_samples
        self.cat_samples = cat_samples
        self.samples = None
        self.scaler = StandardScaler()
        self.minmax = MinMaxScaler(feature_range=(-1,1))
        self.imputer = Imputer()
        self.Y = Y
        self.normalizeContSamples()

    def normalizeContSamples(self):
        samples = self.cont_samples
        samples = self.imputer.fit_transform(samples)
        samples = self.minmax.fit_transform(samples)
        samples = self.scaler.fit_transform(samples)
        self.cont_samples = samples

    def getAllSamples(self):
        all_samples = []
        for cat_sample, cont_sample in zip(self.cat_samples, self.cont_samples):
            whole_sample = np.concatenate((cont_sample, cat_sample), axis=0)
            all_samples.append(whole_sample)
        return all_samples

    def getAllY(self):
        return self.Y

    def getFeatureLabels(self):
        return self.cont_labels + self.cat_labels

    def transform(self, cont_sample, cat_sample):
        cont_sample = self.imputer.transform(cont_sample)
        cont_sample = self.minmax.transform(cont_sample)
        cont_sample = self.scaler.transform(cont_sample)
        whole_sample = np.concatenate((cont_sample[0], cat_sample), axis=0)
        return whole_sample


class CrossValidationFramework(object):

    def __init__(self, X, Y, test_size=0.25):
        self.X = X
        self.Y = Y
        self.test_size = test_size
        self.model = None
        self.trained = False

    def create_split(self):
        X_train, X_test, Y_train, Y_test = \
            cross_validation.train_test_split(
                self.X, self.Y, test_size=self.test_size)
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def train(self):
        raise Exception("Train Function Not Implemented")

    def test(self, graph=True):
        raise Exception("Validate Function not Implemented")

    def score(self, test_results, graph=True):
        raise Exception("Score function not Implemented")

    def predict(self, x_in):
        if self.trained:
            return self.model.predict(x_in)
        else:
            raise Exception("No model trained")

    def runNTimes(self, n, graph=False):
        scores = []
        for i in range(n):
            print "Training %s/%s" % (i + 1, n)
            self.train()
            print "Testing %s/%s" % (i + 1, n)
            score = self.test(graph=graph)
            print "Score: %s" % score
            scores.append(score)
        return scores

class RandomForestValidationFramework(CrossValidationFramework):

    def __init__(self, X, Y, test_size=0.25, feature_labels=None):
        super(RandomForestValidationFramework, self).__init__(X, Y, test_size)
        self.trained = False
        self.feature_labels = feature_labels

    def save_model(self, model_file):
        pass

    def load_model(self, model_file):
        pass

    def train(self):
        self.trained = False
        self.create_split()

        model = RandomForestRegressor(n_estimators=400, n_jobs=-1)
        model.fit(self.X_train, self.Y_train)
        train_error = mean_squared_error(self.Y_train, model.predict(self.X_train))
        test_error = mean_squared_error(self.Y_test, model.predict(self.X_test))
        featimp = sorted(zip(self.feature_labels, model.feature_importances_), reverse=True, key=lambda x: x[1])
        #for l, imp in featimp:
        #    print "%s : %s" % (l,imp)

        # for gradient boosting
        '''
        model = GradientBoostingRegressor(n_estimators=200)
        model.fit(self.X_train, self.Y_train)
        train_error = mean_squared_error(self.Y_train, model.predict(self.X_train))
        test_error = mean_squared_error(self.Y_test, model.predict(self.X_test))
        print train_error
        print test_error
        featimp = sorted(zip(self.feature_labels, model.feature_importances_), reverse=True, key=lambda x: x[1])
        for l, imp in featimp:
            print "%s : %s" % (l,imp)
        '''

        self.model = model
        self.trained = True

    def test(self, graph=False):
        actual = self.Y_test
        predicted = self.model.predict(self.X_test)
        
        #for x,y in zip(actual,predicted):
        #    print "%s -> %s" % (x,y)
        
        test_error = mean_squared_error(actual, predicted)
        return test_error


class LinearRegressionValidationFramework(CrossValidationFramework):

    def __init__(self, X, Y, test_size=0.25, restarts=10):
        super(LinearRegressionValidationFramework, self).__init__(X, Y, test_size)
        self.restarts = 10
        self.model = None
        self.trained = False

    def save_model(self, model_file):
        pass

    def load_model(self, model_file):
        pass

    def train(self):
        self.trained = False
        self.create_split()

        # load the train data, randomize, constrain, then optimize
        #print self.model.coef_
        train_error = np.empty(3)
        test_error = np.empty(3)
        for degree in [1]: # degree for polynomial features

            # For l1 SGD
            '''
            model_one = make_pipeline(PolynomialFeatures(degree), SGDRegressor(penalty='l1', n_iter=50, shuffle=True))
            model_one.fit(self.X_train, self.Y_train)
            train_error = mean_squared_error(self.Y_train, model_one.predict(self.X_train))
            test_error = mean_squared_error(self.Y_test, model_one.predict(self.X_test))
            print train_error
            print test_error
            p1 = plt.plot(model_one.steps[1][1].coef_, label="l1")
            plt.setp(p1, color='r', linewidth=2.0)
            '''

            # For l2 SGD
            '''
            model_two = make_pipeline(PolynomialFeatures(degree), SGDRegressor(penalty='l2', n_iter=50, shuffle=True))
            model_two.fit(self.X_train, self.Y_train)
            train_error = mean_squared_error(self.Y_train, model_two.predict(self.X_train))
            test_error = mean_squared_error(self.Y_test, model_two.predict(self.X_test))
            print train_error
            print test_error
            p2 = plt.plot(model_two.steps[1][1].coef_, label="l2")
            plt.setp(p2, color='g', linewidth=2.0)
            '''

            model_three = make_pipeline(PolynomialFeatures(degree), SGDRegressor(penalty='elasticnet', n_iter=50, shuffle=True))
            model_three.fit(self.X_train, self.Y_train)
            train_error = mean_squared_error(self.Y_train, model_three.predict(self.X_train))
            test_error = mean_squared_error(self.Y_test, model_three.predict(self.X_test))
            print train_error
            print test_error
            #p3 = plt.plot(model_three.steps[1][1].coef_, label="elasticnet")
            #plt.setp(p3, color='b', linewidth=2.0)

        #plt.show(block=True)
        self.model = model_three
        self.trained = True

    def test(self, graph=False):
        train_error = mean_squared_error(self.Y_train, self.model.predict(self.X_train))
        test_error = mean_squared_error(self.Y_test, self.model.predict(self.X_test))
        return (train_error, test_error)

    def score(self, test_results):
        pass


class GPCrossValidationFramework(CrossValidationFramework):

    def __init__(self, kernel, X, Y, test_size=0.2, restarts=10):
        super(GPCrossValidationFramework, self).__init__(X, Y, test_size)
        self.restarts = restarts
        self.kernel = kernel
        self.model = None
        self.trained = False

    def save_model(self, model_file):
        if self.model:
            self.model.pickle(model_file)
        else:
            raise Exception("No model to save")

    def load_model(self, model_file):
        v = pickle.load(open(model_file))
        self.model = GPy.models.GPRegression.setstate(v)

    def train(self):
        self.trained = False
        self.create_split()

        # normalize self.Y_train to (-1,1) to try to improve precision-recall
        '''
        temp = self.Y_train[:, 0]
        temp -= temp.min()
        norm = np.linalg.norm(temp, np.inf)
        temp = ((temp / norm) * 2.0) - 1.0
        self.Y_train = temp[:, np.newaxis]
        '''

        # load the train data, randomize, constrain, then optimize
        self.model = GPy.models.GPRegression(
            self.X_train, self.Y_train, self.kernel)
        self.model.randomize()
        self.model.ensure_default_constraints()
        self.model.optimize_restarts(
            num_restarts=self.restarts, robust=True, parallel=True, num_processes=None)
        self.trained = True

    def test(self, graph=True):
        if not self.trained:
            raise Exception("Train a regression model first")
        # predict for each X in test set, then score against truth
        means = []
        for x_in in self.X_test:
            mean, var, lower, upper = self.predict(x_in)
            means.append(mean[0][0])
        score = self.score(means, graph=graph)
        return score

    def normalized_dcg(self, test_truth, test_results, p=None):
        zipped = zip(test_truth, test_results)
        sorted_zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
        # create idealized ranking and model result ranking
        sorted_results = [x for x,y in sorted_zipped]
        sorted_truth = sorted(test_truth, reverse=True)
        # p used if we want to compare scores across models/queries
        if p:
            sorted_results = sorted_results[:p]
            sorted_truth = sorted_truth[:p]
        # calculated IDCG
        dcg = self._dcg(sorted_results)
        idcg = self._dcg(sorted_truth)
        ndcg = dcg/idcg
        return ndcg

    def _dcg(self, values):
        dcg = float(values[0])
        for i,v in enumerate(values[1:]):
            rel_i = i + 2
            dcg += float(v)/math.log(rel_i,2)
        return dcg

    def score(self, test_results, graph=False):
        test_truth = self.Y_test[:, 0]
        r2 = r2_score(test_truth, test_results)
        #ndcg = self.normalized_dcg(test_truth, test_results)
        mse = mean_squared_error(test_truth, test_results)
        scores = {'r2': float(r2), 'mse': float(mse),
                  'truth': list(test_truth),
                  'model': list(test_results)}
        return scores

if __name__ == "__main__":
    '''
    kernel = datafetcher.kernel
    X = datafetcher.vectors
    Y = datafetcher.all_cvrs
    '''
    pid = "curryst01"
    print pid

    playergames = findAllTrainingGames(pid)
    cat_X = []
    cont_X = []
    Y = []
    cat_labels = None
    cont_labels = None
    for g in playergames:
        gid = str(g['game_id'])
        fext = featureExtractor(pid, target_game=gid)
        cat_labels, cat_features, cont_labels, cont_features = fext.run()
        y = fext.getY('PTS')
        cat_X.append(cat_features)
        cont_X.append(cont_features)
        Y.append(y)

    # normalize the data
    proc = DataPreprocessor(cont_labels, cont_X, cat_labels, cat_X, Y)
    proc.normalizeContSamples()
    X = proc.getAllSamples()
    Y = proc.getAllY()
    labels = proc.getFeatureLabels()

    rf_framework = RandomForestValidationFramework(X,Y, test_size=0.25, feature_labels=labels)
    rf_framework.train()
    rf_framework.test()

    sys.exit(1)

    lr_framework = LinearRegressionValidationFramework(X, Y, test_size=0.25)
    lr_framework.train()

    '''
    gp_framework = GPCrossValidationFramework(
        kernel, X, np.array(Y), test_size=0.25, restarts=2)
    results = val_framework.runNTimes(1)

    # save model
    val_framework.save_model(model_file)

    # save results
    with open(result_file, 'w') as out_file:
        out_file.write(json.dumps(results))
    '''
