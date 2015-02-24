"""
TODO: build distributional projections
"""

import sys
import re
import collections
import random
import math
import copy

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

    def __init__(self, timestamps, cont_labels, cont_samples, cat_labels, cat_samples, Y, cat_kernel_splits=None):
        self.cont_labels = cont_labels
        self.cat_labels = cat_labels
        self.cont_samples = cont_samples
        self.cat_samples = cat_samples
        self.Y = Y
        self.timestamps = timestamps
        self.cat_kernel_splits = cat_kernel_splits
        if self.cat_kernel_splits is None:
            self.cat_kernel_splits = []

        if not (len(self.cont_samples) == len(self.cat_samples) == len(self.Y)):
            raise Exception("Inconsistent samples")

        self.labels = None
        self.samples = None
        self.scaler = StandardScaler()
        self.minmax = MinMaxScaler(feature_range=(-1,1))
        self.imputer = Imputer()
        
        self.sortSamples()
        self.normalizeContSamples()

    def sortSamples(self):
        indices = [i for i,ts in sorted(enumerate(self.timestamps), key=lambda x: x[1])]
        self.cont_samples = [self.cont_samples[indices[i]] for i in range(len(self.timestamps))]
        self.cat_samples = [self.cat_samples[indices[i]] for i in range(len(self.timestamps))]
        self.Y = [self.Y[indices[i]] for i in range(len(self.timestamps))]

    def normalizeContSamples(self):
        samples = self.cont_samples
        samples = self.imputer.fit_transform(samples)
        samples = self.minmax.fit_transform(samples)
        samples = self.scaler.fit_transform(samples)
        self.cont_samples = samples

    def getKernel(self):
        '''
        added rbf kernels for cont features
        mult rbf for cat features, split at splits 
        '''
        kernel = None
        kernel_pieces = []
        # for conts
        if len(self.cont_labels) > 0:
            cont_kernel = GPy.kern.Matern52(len(self.cont_labels), ARD=True)
            kernel_pieces.append(cont_kernel)
            # GPy.kern.rbf(len(self.cont_labels))

        # for cats
        cat_size = len(self.cat_labels)
        kernel_sizes = []
        if len(self.cat_kernel_splits) == 0:
            kernel_sizes.append(cat_size)
        else:
            kernel_sizes = self.cat_kernel_splits
        for size in kernel_sizes:
            if size > 0:
                cat_kernel = GPy.kern._src.rbf.RBF(size, ARD=True)
                kernel_pieces.append(cat_kernel)

        # make into one kernel
        for p in kernel_pieces:
            if kernel is None:
                kernel = p
            else:
                kernel = kernel.add(p)

        return kernel

    def getAllSamples(self):
        """
        DON'T CHANGE THIS ORDER
        """
        all_samples = []
        for cat_sample, cont_sample in zip(self.cat_samples, self.cont_samples):
            whole_sample = np.concatenate((cont_sample, cat_sample), axis=0)
            all_samples.append(whole_sample)
        return all_samples

    def getY(self):
        return self.Y

    def getTrendY(self, window_size=7):
        '''
        returns Y as a percentage above or below the windowed avg
        '''
        in_vals = copy.deepcopy(self.Y)
        trend_y = []
        for i,v in enumerate(in_vals):
            trend_y.append(np.mean(in_vals[max(0,i+1-window_size):i+1]))
        ratios = []
        for i in range(len(trend_y)):
            denum = float(trend_y[i])
            if denum == 0.0:
                ratios.append(1.0)
            else:
                ratios.append(float(in_vals[i])/denum)
        return ratios

    def runningAverage(self, window_size=7):
        in_vals = copy.deepcopy(self.Y)
        trend_y = []
        for i,v in enumerate(in_vals):
            trend_y.append(np.mean(in_vals[max(0,i+1-window_size):i+1]))
        trend_y = [trend_y[0]] + trend_y[:-1]
        return trend_y

    def runningPolyFit(self, window_size=7):
        in_vals = copy.deepcopy(self.Y)
        x = np.array(range(window_size))
        fit_y = []
        # padd
        padded_vals = [in_vals[0]]*(window_size) + in_vals
        for i in range(window_size,len(padded_vals)):
            window_vals = padded_vals[i-window_size:i]
            z = np.polyfit(x, window_vals, 2)
            p = np.poly1d(z)
            y = p(window_size)
            fit_y.append(y)
            # print "%s -> %s (actual %s)" % (window_vals, y, in_vals[len(fit_y)-1])
        return fit_y

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

    def transform(self, x):
        if self.trained and self.model:
            return self.model.transform(x)
        return

    def train(self, zip_data=None, weigh_recent=True):
        self.trained = False

        if zip_data:
            self.X = zip(zip_data,self.X)
            self.create_split()
            # testing
            self.data_train = [a for a,b in self.X_train]
            self.data_test = [a for a,b in self.X_test]
            self.X_train = [b for a,b in self.X_train]
            self.X_test = [b for a,b in self.X_test]
        else:
            self.data_train = None
            self.data_test = None
            self.create_split()

        model = RandomForestRegressor(n_estimators=400, n_jobs=-1)
        # TODO: weigh more recent samples higher
        model.fit(self.X_train, self.Y_train, sample_weight=None)
        train_error = mean_squared_error(self.Y_train, model.predict(self.X_train))
        test_error = mean_squared_error(self.Y_test, model.predict(self.X_test))
        featimp = sorted(zip(self.feature_labels, model.feature_importances_), reverse=True, key=lambda x: x[1])
        for l, imp in featimp:
            print "%s : %s" % (l,imp)
        
        # for gradient boosting
        '''
        model = GradientBoostingRegressor(n_estimators=200)
        model.fit(self.X_train, self.Y_train)
        train_error = mean_squared_error(self.Y_train, model.predict(self.X_train))
        test_error = mean_squared_error(self.Y_test, model.predict(self.X_test))
        featimp = sorted(zip(self.feature_labels, model.feature_importances_), reverse=True, key=lambda x: x[1])
        for l, imp in featimp:
            print "%s : %s" % (l,imp)
        '''

        self.model = model
        self.trained = True

    def test(self, graph=False):
        actual = self.Y_test
        predicted = self.model.predict(self.X_test)
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

    def train(self, zip_data=None, weigh_recent=True):
        self.trained = False

        if zip_data:
            self.X = zip(zip_data,self.X)
            self.create_split()
            # testing
            self.data_train = [a for a,b in self.X_train]
            self.data_test = [a for a,b in self.X_test]
            self.X_train = np.array([b for a,b in self.X_train])
            self.X_test = np.array([b for a,b in self.X_test])
        else:
            self.data_train = None
            self.data_test = None
            self.create_split()
            self.X_train = np.array(self.X_train)
            self.X_test = np.array(self.X_test)

        # TODO: weight more recent samples heavily

        # add new axis for Y, as needed by GPy
        self.Y_train = np.array(self.Y_train)[:, np.newaxis]
        self.Y_test = np.array(self.Y_test)[:, np.newaxis]

        # load the train data, randomize, constrain, then optimize
        self.model = GPy.models.GPRegression(
            self.X_train, self.Y_train, self.kernel)
        self.model.randomize()
        #self.model.ensure_default_constraints()
        self.model.optimize_restarts(
            num_restarts=self.restarts, robust=True, parallel=True, num_processes=None)
        print self.model
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


def getUpcomingGameFeatures(pid, matched_game):
    fn_names = [('plf','runPlayerFeatures'), ('phf','runPhysicalFeatures'), ('oef','runOppositionEarnedFeatures'), ('oaf','runOppositionAllowedFeatures'), ('opf','runOppositionPlayerFeatures')]
    pfe = featureExtractor(pid, target_game=matched_game)
    features = {}
    for name, fn_name in fn_names:
        new_features = getattr(pfe, fn_name)()
        features[name] = new_features
    return features

def getProcessor(pid, games, fn_name, y_key):
    cat_X = []
    cont_X = []
    Y = []
    ts_X = []
    cat_labels = None
    cont_labels = None
    cat_splits = None

    for g in games:
        gid = str(g['game_id'])
        fext = featureExtractor(pid, target_game=gid) 
        cat_labels, cat_features, cont_labels, cont_features, cat_splits = getattr(fext, fn_name)()
        ts = fext.timestamp()
        y = fext.getY(y_key)
        ts_X.append(ts)
        cat_X.append(cat_features)
        cont_X.append(cont_features)
        Y.append(y)

    # normalize the data
    proc = DataPreprocessor(ts_X, cont_labels, cont_X, cat_labels, cat_X, Y, cat_kernel_splits=cat_splits)
    proc.normalizeContSamples()
    return proc

def trainTrendModels(pid, training_games, y_key, weigh_recent=False, test=False, plot=False):
    processors = {}
    models = {}

    if test:
        test_size = 0.2
    else:
        test_size = 0.0

    for name, f in [('phf','runPhysicalFeatures'), ('oef','runOppositionEarnedFeatures'), ('oaf','runOppositionAllowedFeatures'), ('opf','runOppositionPlayerFeatures')]:
        processors[name] = getProcessor(pid, training_games, f, y_key)
    for name,proc in processors.iteritems():
        print name
        X = proc.getAllSamples()
        trendY = proc.getTrendY()
        realY = proc.getY()
        labels = proc.getFeatureLabels()
        ra = proc.runningAverage()
        kernel = proc.getKernel()
        zipped_ra = zip(ra,realY)

        # for random forest
        '''
        framework = RandomForestValidationFramework(X,trendY, test_size=0.20, feature_labels=labels)
        framework.train(zip_data=zipped_ra)
        predicted_trend = framework.model.predict(framework.X_test)
        '''

        # for gp
        kernel = proc.getKernel()
        framework = GPCrossValidationFramework(kernel, X, trendY, test_size=test_size, restarts=10)
        framework.train(zip_data=zipped_ra, weigh_recent=weigh_recent)

        # save the model
        models[name] = framework

        if test:
            means, variances = framework.model.predict(framework.X_test)
            predicted_trend = [_[0] for _ in means]
            predicted_vars = [_[0] for _ in variances]

            actual_trend = [_[0] for _ in framework.Y_test]
            ra = framework.data_test
            avgs = [_[0] for _ in ra]
            actual = [_[1] for _ in ra]
            predicted = [avgs[i]*_ for i,_ in enumerate(predicted_trend)]
            corrs = np.corrcoef(actual_trend, predicted_trend)
            
            print actual_trend
            print predicted_trend
            print predicted_vars
            print "%s: %s" % (name, corrs)

            if plot:
                indices = [i for i,a in sorted(enumerate(predicted_trend), key=lambda x:x[1])]
                sorted_actual = [actual[indices[i]] for i in range(len(indices))]
                sorted_actual_trends = [actual_trend[indices[i]] for i in range(len(indices))]
                sorted_predicted = [predicted[indices[i]] for i in range(len(indices))]
                sorted_predicted_trends = [predicted_trend[indices[i]] for i in range(len(indices))]
                sorted_avgs = [avgs[indices[i]] for i in range(len(indices))]
                sorted_vars = [predicted_vars[indices[i]] for i in range(len(indices))]

                fig = plt.figure()
                p1 = plt.plot(sorted_actual_trends)
                p2 = plt.plot(sorted_predicted_trends)
                p3 = plt.plot(sorted_vars)
                fig.suptitle(name, fontsize=20)
    if test and plot:
        plt.show(block=True)
    return (processors, models)

def trainCoreModels(pid, training_games, y_key, weigh_recent=False, test=False, plot=False):
    processors = {}
    models = {}

    if test:
        test_size = 0.2
    else:
        test_size = 0.0

    for name, f in [('plf','runPlayerFeatures')]:
        processors[name] = getProcessor(pid, training_games, f, y_key)
    for name,proc in processors.iteritems():
        print name
        X = proc.getAllSamples()
        trendY = proc.getTrendY()
        realY = proc.getY()
        labels = proc.getFeatureLabels()
        ra = proc.runningAverage()
        kernel = proc.getKernel()

        framework = GPCrossValidationFramework(kernel, X, realY, test_size=0.20, restarts=10)
        framework.train(zip_data=None, weigh_recent=weigh_recent)
        models[name] = framework

        if test:
            actual = [_[0] for _ in framework.Y_test]
            means, variances = framework.model.predict(framework.X_test)
            predicted = [_[0] for _ in means]
            predicted_vars = [_[0] for _ in variances]
            corrs = np.corrcoef(actual, predicted)

            print actual
            print predicted
            print predicted_vars
            print "%s: %s" % (name, corrs)

            if plot:
                indices = [i for i,a in sorted(enumerate(predicted), key=lambda x:x[1])]
                sorted_actual = [actual[indices[i]] for i in range(len(indices))]
                sorted_predicted = [predicted[indices[i]] for i in range(len(indices))]
                sorted_vars = [predicted_vars[indices[i]] for i in range(len(indices))]

                fig = plt.figure()
                p1 = plt.plot(sorted_actual)
                p2 = plt.plot(sorted_predicted)
                p3 = plt.plot(sorted_vars)
                fig.suptitle(name, fontsize=20)
    if test and plot:
        plt.show(block=True)
    return (processors, models)


if __name__ == "__main__":
    pid = "curryst01"
    y_key = 'BLK'
    print "training %s for %s" % (pid, y_key)
    training_games = findAllTrainingGames(pid)
    core_processors, core_models = trainCoreModels(pid, training_games, y_key, test=True, plot=True)
    trend_processors, trend_models = trainTrendModels(pid, training_games, y_key, test=True, plot=True)

