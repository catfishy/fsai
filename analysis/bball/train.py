"""
TODO: build distributional projections
"""

import sys
import re
import collections
import random
import math
import copy
import traceback

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
import scipy

from analysis.bball.playerAnalysis import *
from statsETL.db.mongolib import *


TREND_FEATURE_FNS = []
CORE_FEATURE_FNS = [('plf','runPlayerFeatures'),
                    ('oef','runOppositionEarnedFeatures'),
                    ('if','runIntensityFeatures'), 
                    ('phf','runPhysicalFeatures'),
                    ('oaf','runOppositionAllowedFeatures'), 
                    ('opf','runOppositionPlayerFeatures')]
UNUSED_FNS = []

class DataPreprocessor(object):

    """
    take samples, with samples split into cont vars and cat vars
    expects cat vars to be one hot encoded to -1 or 1
    normalizes cont vars by minmax to (-1,1), then scaling to 0 mean and 1 stdvar
    concatenates cont and cat vars to product single sample list
    """

    def __init__(self, timestamps, cont_labels, cont_samples, cat_labels, cat_samples, Y, 
                 cat_kernel_splits=None, weigh_recent=False, weigh_outliers=False, 
                 clamp=None):
        self.cont_labels = cont_labels
        self.cat_labels = cat_labels
        self.cont_samples = np.array(cont_samples)
        self.cat_samples = np.array(cat_samples)
        if self.cont_labels is None:
            self.cont_labels = []
        if self.cat_labels is None:
            self.cat_labels = []

        self.Y = np.array(Y)
        self.timestamps = np.array(timestamps)
        self.cat_kernel_splits = cat_kernel_splits
        if self.cat_kernel_splits is None:
            self.cat_kernel_splits = []

        if not (len(self.cont_samples) == len(self.cat_samples) == len(self.Y)):
            raise Exception("Inconsistent samples")

        if len(self.cont_samples) < 2 and len(self.cat_samples) < 2:
            raise Exception("Need more than one data point")

        self.clamp = clamp if clamp else (-1,1)
        self.clamped = True if clamp else False

        self.labels = None
        self.samples = None
        self.scaler = StandardScaler()
        self.minmax = MinMaxScaler(feature_range=self.clamp)
        self.cat_minmax = MinMaxScaler(feature_range=self.clamp)
        self.imputer = Imputer()

        # sort and normalize
        self.trendY = None
        self.runningAverage = None
        self.sortSamples()
        self.findBadContCols()
        self.normalizeSamples()

        # calculate different target values
        self.trendY = self.calcTrendY(window_size=15)
        self.runningAverage = self.calcRunningAverage(window_size=15)

        # upsample
        self.upsample(recent=weigh_recent, outliers=weigh_outliers)

    def upsample(self, recent, outliers):
        to_add_cont = []
        to_add_cat = []
        to_add_Y = []
        to_add_timestamps = []
        to_add_trendY = []
        to_add_runningAvg = []

        if recent:
            top = 7
            top_2 = 4
            # calculate upsamples
            to_add_cont.append(self.cont_samples[-top:])
            #to_add_cont.append(self.cont_samples[-top_2:])
            to_add_cat.append(self.cat_samples[-top:])
            #to_add_cat.append(self.cat_samples[-top_2:])
            to_add_Y.append(self.Y[-top:])
            #to_add_Y.append(self.Y[-top_2:])
            to_add_timestamps.append(self.timestamps[-top:])
            #to_add_timestamps.append(self.timestamps[-top_2:])
            to_add_trendY.append(self.trendY[-top:])
            #to_add_trendY.append(self.trendY[-top_2:])
            to_add_runningAvg.append(self.runningAverage[-top:])
            #to_add_runningAvg.append(self.runningAverage[-top_2:])

        if outliers:
            by_value = defaultdict(list)
            for k,v in enumerate(self.Y):
                by_value[abs(v)].append(k)
            highest_values = sorted(by_value.items(), key=lambda x: x[0], reverse=True)
            top = max(1,int(len(highest_values) * 0.2))
            highest_values = highest_values[:top]
            indices = np.array([c for k,v in highest_values for c in v])
            to_add_cont.append(self.cont_samples[indices])
            to_add_cat.append(self.cat_samples[indices])
            to_add_Y.append(self.Y[indices])
            to_add_timestamps.append(self.timestamps[indices])
            to_add_trendY.append(self.trendY[indices])
            to_add_runningAvg.append(self.runningAverage[indices])

        # concat samples
        if to_add_cont:
            self.cont_samples = np.concatenate((self.cont_samples,[b for a in to_add_cont for b in a]))
            self.cat_samples = np.concatenate((self.cat_samples, [b for a in to_add_cat for b in a]))
            self.Y = np.concatenate((self.Y, [b for a in to_add_Y for b in a]))
            self.timestamps = np.concatenate((self.timestamps,[b for a in to_add_timestamps for b in a]))
            self.trendY = np.concatenate((self.trendY, [b for a in to_add_trendY for b in a]))
            self.runningAverage = np.concatenate((self.runningAverage, [b for a in to_add_runningAvg for b in a]))
            # re-sort
            self.sortSamples()

    def sortSamples(self):
        indices = [i for i,ts in sorted(enumerate(self.timestamps), key=lambda x: x[1])]
        self.cont_samples = np.array([self.cont_samples[indices[i]] for i in range(len(self.timestamps))])
        self.cat_samples = np.array([self.cat_samples[indices[i]] for i in range(len(self.timestamps))])
        self.Y = np.array([self.Y[indices[i]] for i in range(len(self.timestamps))])
        self.timestamps = np.array(list(sorted(self.timestamps)))
        if self.trendY is not None:
            self.trendY = np.array([self.trendY[indices[i]] for i in range(len(self.timestamps))])
        if self.runningAverage is not None:
            self.runningAverage = np.array([self.runningAverage[indices[i]] for i in range(len(self.timestamps))])

    def normalizeSamples(self):  
        samples = np.array(self.cont_samples)
        samples = self.imputer.fit_transform(samples)
        samples = self.minmax.fit_transform(samples)
        if not self.clamped:
            samples = self.scaler.fit_transform(samples)
        self.cont_samples = samples
        # remove bad columns from labels
        for bad_col in sorted(self.bad_cont_cols,reverse=True):
            self.cont_labels = scipy.delete(self.cont_labels, bad_col, 0)

        # normalize cat samples
        cat_samples = np.array(self.cat_samples)
        cat_samples = self.cat_minmax.fit_transform(cat_samples)
        self.cat_samples = cat_samples

    def findBadContCols(self):
        # find features that would be removed after imputing
        to_delete = []
        samples = np.array(self.cont_samples)
        for col_i in range(len(self.cont_labels)):
            col_vals = samples[:,col_i]
            bad_col = all([np.isnan(i) for i in col_vals])
            if bad_col:
                to_delete.append(col_i)
        self.bad_cont_cols = to_delete

    def getKernel(self):
        '''
        added rbf kernels for cont features
        mult rbf for cat features, split at splits 
        '''
        kernel = None
        kernel_pieces = []
        # for conts
        if len(self.cont_labels) > 0:
            for i in range(len(self.cont_labels)):
                name = self.cont_labels[i]
                name = name.replace('%','_percent')
                cont_kernel = GPy.kern._src.rbf.RBF(1, active_dims=[i], name=name)
                kernel_pieces.append(cont_kernel)
            
            #cont_kernel = GPy.kern.Matern52(len(self.cont_labels))
            #kernel_pieces.append(cont_kernel)
            # GPy.kern.rbf(len(self.cont_labels))

        # for cats
        cat_size = len(self.cat_labels)
        kernel_sizes = []
        if len(self.cat_kernel_splits) == 0:
            kernel_sizes.append(cat_size)
        else:
            kernel_sizes = self.cat_kernel_splits
        start_i = 0
        for size in kernel_sizes:
            if size > 0:
                cat_kernel = GPy.kern._src.rbf.RBF(size, active_dims=range(start_i, start_i + size), ARD=True)
                start_i += size
                kernel_pieces.append(cat_kernel)

        # make into one kernel
        kernel = None
        for p in kernel_pieces:
            if kernel is None:
                kernel = p
            else:
                kernel = kernel + p
        return kernel
        

    def getAllSamples(self):
        """
        DON'T CHANGE THIS ORDER
        """
        all_samples = []
        for cat_sample, cont_sample in zip(self.cat_samples, self.cont_samples):
            whole_sample = np.concatenate((cont_sample, cat_sample), axis=0)
            all_samples.append(whole_sample)
        return np.array(all_samples)

    def getY(self):
        return copy.deepcopy(self.Y)

    def getTrendY(self):
        return copy.deepcopy(self.trendY)

    def getRunningAverage(self):
        return copy.deepcopy(self.runningAverage)

    def calcTrendY(self, window_size=15):
        '''
        returns Y as a percentage above or below the windowed avg
        determines window by first looking backward, if not a
        large enough window, then increase window going into the future
        until all values are consumed or window size is reached
        '''
        in_vals = copy.deepcopy(self.Y)
        ratios = []
        for i,y in enumerate(in_vals):
            prev_values = in_vals[max(0,i-window_size):max(0,i)]
            if len(prev_values) < window_size:
                # try to push it forward
                length_needed = window_size - len(prev_values)
                future_values = in_vals[i:i+length_needed]
                prev_values = np.concatenate((prev_values,future_values))
            prev_avg = np.mean(prev_values)
            if prev_avg == 0.0 and y > 0.0:
                ratios.append(2.0) # DEFAULT TO 2.0 IF INF
            elif prev_avg == 0.0 and y == 0.0:
                ratios.append(1.0)
            else:
                ratios.append(y/prev_avg)
        return np.array(ratios)

    def calcRunningAverage(self, window_size=15):
        '''
        DEPRECATED
        '''
        in_vals = copy.deepcopy(self.Y)
        trend_y = []
        for i,v in enumerate(in_vals):
            trend_y.append(np.mean(in_vals[max(0,i+1-window_size):i+1]))
        trend_y = [trend_y[0]] + trend_y[:-1]
        return np.array(trend_y)

    def getFeatureLabels(self):
        return list(self.cont_labels) + list(self.cat_labels)

    def transform(self, cont_sample, cat_sample):
        if cont_sample:
            samples = np.array(cont_sample)
            samples = self.imputer.transform(samples)
            samples = self.minmax.transform(samples)
            if not self.clamped:
                samples = self.scaler.transform(samples)
            cont_sample = samples[0]
        if cat_sample:
            samples = np.array(cat_sample)
            samples = self.cat_minmax.transform(samples)
            cat_sample = samples[0]
        whole_sample = np.concatenate((cont_sample, cat_sample), axis=0)
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

    def __init__(self, kernel, X, Y, test_size=0.2, restarts=5):
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

    def train(self, zip_data=None):
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
        #self.model.ensure_default_constraints()
        #self.model.optimize_restarts(
        #    num_restarts=self.restarts, robust=True, parallel=True, num_processes=None)
        #self.model.constrain_positive('') # '' is a regex matching all parameter names
        self.model.unconstrain()
        #self.model.randomize()
        self.model['.*var'].constrain_positive()
        self.model['.*lengthscale'].constrain_positive()
        self.model.Gaussian_noise.constrain_positive()
        self.model.optimize()
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
    fn_names = TREND_FEATURE_FNS + CORE_FEATURE_FNS
    pfe = featureExtractor(pid, target_game=matched_game)
    features = {}
    for name, fn_name in fn_names:
        new_features = getattr(pfe, fn_name)()
        features[name] = new_features
    return features

def getProcessor(pid, games, fn_name, y_key, weigh_recent=False, weigh_outliers=False):
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
        try:
            cat_labels, cat_features, cont_labels, cont_features, cat_splits = getattr(fext, fn_name)()
        except Exception as e:
            print "Exception running %s for pid %s, game %s: %s" % (fn_name, pid, gid, e)
            print traceback.print_exc()
            continue

        ts = fext.timestamp()
        y = fext.getY(y_key)
        ts_X.append(ts)
        cat_X.append(cat_features)
        cont_X.append(cont_features)
        Y.append(y)

    # normalize the data
    proc = DataPreprocessor(ts_X, cont_labels, cont_X, cat_labels, cat_X, Y, 
                            cat_kernel_splits=cat_splits, weigh_recent=weigh_recent, 
                            weigh_outliers=weigh_outliers)
    return proc

def trainTrendModels(pid, training_games, y_key, weigh_recent=False, weigh_outliers=True, test=False, plot=False):
    processors = {}
    models = {}

    if test:
        test_size = 0.3
    else:
        test_size = 0.0

    for name, f in TREND_FEATURE_FNS:
        processors[name] = getProcessor(pid, training_games, f, y_key, weigh_recent=weigh_recent, weigh_outliers=weigh_outliers)
    for name,proc in processors.iteritems():
        X = proc.getAllSamples()
        trendY = proc.getTrendY()
        realY = proc.getY()
        print "name: %s, ykey: %s, model: %s, x_len: %s, y_len: %s" % (pid, y_key, name, len(X), len(realY))
        labels = proc.getFeatureLabels()
        ra = proc.getRunningAverage()
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
        framework.train(zip_data=zipped_ra)
        print framework.model
        # save the model
        models[name] = framework

        if test:
            predicted_trend = []
            predicted_vars = []
            for x_test in framework.X_test:
                means, variances = framework.model.predict(np.array([x_test]))
                predicted_trend.append(means[0][0])
                predicted_vars.append(math.sqrt(variances[0][0]))

            actual_trend = [_[0] for _ in framework.Y_test]
            ra = framework.data_test
            avgs = [_[0] for _ in ra]
            actual = [_[1] for _ in ra]
            #predicted = [avgs[i]*_ for i,_ in enumerate(predicted_trend)]
            corrs = np.corrcoef(actual_trend, predicted_trend)

            print "%s for %s" % (pid, y_key)
            print actual_trend
            print predicted_trend
            print predicted_vars
            print "%s: %s" % (name, corrs)


            if plot:
                indices = [i for i,a in sorted(enumerate(predicted_trend), key=lambda x:x[1])]
                sorted_actual = [actual[indices[i]] for i in range(len(indices))]
                sorted_actual_trends = [actual_trend[indices[i]] for i in range(len(indices))]
                #sorted_predicted = [predicted[indices[i]] for i in range(len(indices))]
                sorted_predicted_trends = [predicted_trend[indices[i]] for i in range(len(indices))]
                sorted_avgs = [avgs[indices[i]] for i in range(len(indices))]
                sorted_vars = [predicted_vars[indices[i]] for i in range(len(indices))]

                fig = plt.figure()
                p1 = plt.plot(sorted_actual_trends, marker='o', label='actual')
                p2 = plt.plot(sorted_predicted_trends, marker='o', label='predicted')
                p3 = plt.plot(sorted_vars, marker='o', label='vars')
                fig.suptitle(name, fontsize=20)
                plt.legend(loc='best', shadow=True)
    if test and plot:
        plt.show(block=True)
    return (processors, models)

def trainCoreModels(pid, training_games, y_key, weigh_recent=False, weigh_outliers=True, test=False, plot=False):
    processors = {}
    models = {}

    if test:
        test_size = 0.3
    else:
        test_size = 0.0

    for name, f in CORE_FEATURE_FNS:
        processors[name] = getProcessor(pid, training_games, f, y_key, weigh_recent=weigh_recent, weigh_outliers=weigh_outliers)
    for name,proc in processors.iteritems():
        X = proc.getAllSamples()
        trendY = proc.getTrendY()
        realY = proc.getY()
        print "name: %s, ykey: %s, model: %s, x_len: %s, y_len: %s" % (pid, y_key, name, len(X), len(realY))
        labels = proc.getFeatureLabels()
        ra = proc.getRunningAverage()
        kernel = proc.getKernel()

        framework = GPCrossValidationFramework(kernel, X, realY, test_size=test_size, restarts=10)
        framework.train(zip_data=None)
        models[name] = framework

        if test:
            predicted = []
            predicted_vars = []
            for x_test in framework.X_test:
                means, variances = framework.model.predict(np.array([x_test]))
                predicted.append(means[0][0])
                predicted_vars.append(math.sqrt(variances[0][0]))

            

            actual = [_[0] for _ in framework.Y_test]
            corrs = np.corrcoef(actual, predicted)

            print "%s for %s" % (pid, y_key)
            print actual
            print predicted
            print predicted_vars
            print "%s: %s" % (name, corrs)
            #print framework.model['.*lengthscale']

            if plot:
                indices = [i for i,a in sorted(enumerate(predicted), key=lambda x:x[1])]
                sorted_actual = [actual[indices[i]] for i in range(len(indices))]
                sorted_predicted = [predicted[indices[i]] for i in range(len(indices))]
                sorted_vars = [predicted_vars[indices[i]] for i in range(len(indices))]

                fig = plt.figure()
                p1 = plt.plot(sorted_actual, marker='o', label='actual')
                p2 = plt.plot(sorted_predicted, marker='o', label='predicted')
                p3 = plt.plot(sorted_vars, marker='o', label='vars')
                fig.suptitle(name, fontsize=20)
                plt.legend(loc='best', shadow=True)
    if test and plot:
        plt.show(block=True)
    return (processors, models)


if __name__ == "__main__":
    pid = "irvinky01"
    y_key = 'PTS'
    print "training %s for %s" % (pid, y_key)
    training_games = findAllTrainingGames(pid)
    proc = getProcessor(pid, training_games, 'runEncoderFeatures', y_key, weigh_recent=False, weigh_outliers=False)
    X = proc.getAllSamples()
    labels = proc.getFeatureLabels()

    for x in X:
        for a,b in zip(labels,x):
            print "%s: %s" % (a,b)

    sys.exit(1)



    realY = proc.getY()
    kernel = proc.getKernel()
    framework = GPCrossValidationFramework(kernel, X, realY, test_size=0.3, restarts=10)
    framework.train(zip_data=None)
    print framework.model

    predicted = []
    predicted_vars = []
    for x_test in framework.X_test:
        means, variances = framework.model.predict(np.array([x_test]))
        predicted.append(means[0][0])
        predicted_vars.append(math.sqrt(variances[0][0]))
    actual = [_[0] for _ in framework.Y_test]
    corrs = np.corrcoef(actual, predicted)
    print "%s for %s" % (pid, y_key)
    print actual
    print predicted
    print predicted_vars
    print corrs

    if plot:
        indices = [i for i,a in sorted(enumerate(predicted), key=lambda x:x[1])]
        sorted_actual = [actual[indices[i]] for i in range(len(indices))]
        sorted_predicted = [predicted[indices[i]] for i in range(len(indices))]
        sorted_vars = [predicted_vars[indices[i]] for i in range(len(indices))]

        fig = plt.figure()
        p1 = plt.plot(sorted_actual, marker='o', label='actual')
        p2 = plt.plot(sorted_predicted, marker='o', label='predicted')
        p3 = plt.plot(sorted_vars, marker='o', label='vars')
        fig.suptitle(name, fontsize=20)
        plt.legend(loc='best', shadow=True)


    sys.exit(1)
    core_processors, core_models = trainCoreModels(pid, training_games, y_key, weigh_outliers=False, test=True, plot=True)
    trend_processors, trend_models = trainTrendModels(pid, training_games, y_key, weigh_outliers=False, test=True, plot=True)
