"""
TODO: build distributional projections
"""

import sys
import os
import re
import collections
import random
import math
import copy
import traceback
import cPickle
import time
from datetime import datetime

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
from sklearn import cross_validation

from analysis.bball.playerAnalysis import *
from statsETL.db.mongolib import *
from analysis.util.autoencoder import train_dA, shared_dataset
from analysis.util.stacked_autoencoder import train_SdA


def loadModel(filepath):
    return cPickle.load(open(filepath, 'rb'))

def pickleModel(filepath, model):
    return cPickle.dump(model, open(filepath,'wb'))

def encodeCSVInput(csv_path, model_path, y_key='PTS', limit=5000):
    dproc = StreamDataPreprocessor(streamCSV(csv_file), limit=limit, clamp=(0,1))
    Y = dproc.getY(y_key)
    X = dproc.getAllSamples()
    labels = dproc.getFeatureLabels()

    # using AE to encode
    da = loadModel(model_path)
    encoded_X = [da.get_hidden_values(_).eval() for _ in X] 
    return (encoded_X, Y)

def encodePlayerInput(pid, model_path, pipeline_path, y_key='PTS', limit=None, time=None, min_count=None):
    stream = streamPlayerInput(pid, limit=limit, time=time, min_count=min_count)
    dproc = StreamDataPreprocessor(stream, pipeline_path=pipeline_path)
    Y = dproc.getY(y_key)
    X = dproc.getAllSamples()
    labels = dproc.getFeatureLabels()

    # using AE to encode
    da = loadModel(model_path)
    encoded_X = [da.get_hidden_values(_).eval() for _ in X]

    # normalize encoded values
    scaler = StandardScaler()
    encoded_X = scaler.fit_transform(encoded_X)

    return (encoded_X, Y)

def getNBATrainingData(file_path, pipeline_path=None, y_key=None, test_size=0.4, limit=10000, n_outs=None, mean_window=3):
    if y_key is None:
        y_key = 'PTS'
    dproc = StreamDataPreprocessor(streamCSV(file_path), limit=limit, clamp=3, feature_range=(0,1), mean_window=mean_window)

    if pipeline_path is not None:
        dproc.dumpPipeline(pipeline_path)

    if n_outs:
        Y, Means = dproc.getClassY(y_key, n_outs, clamp=3)
    else:
        Y, Means = dproc.getY(y_key)

    X = dproc.getAllSamples()
    labels = dproc.getFeatureLabels()
    
    # zip the means with y
    y_and_means = zip(Y, Means)

    # split into train/test
    X_train, X_test, Y_means_train, Y_means_test = cross_validation.train_test_split(X, y_and_means, test_size=test_size)
    Y_train = [_[0] for _ in Y_means_train]
    train_set = (X_train, Y_train)
    # split test set into validation/test
    X_valid, X_test, Y_means_valid, Y_means_test = cross_validation.train_test_split(X_test, Y_means_test, test_size=0.5)
    Y_valid = [_[0] for _ in Y_means_valid]
    Y_test = [_[0] for _ in Y_means_test]
    Means_test = [_[1] for _ in Y_means_test]
    valid_set = (X_valid, Y_valid)
    test_set = (X_test, Y_test)

    # put into shared vars
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [labels, (Means_test, Y_test), (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

    return rval

def trainAE(csv_path, model_path, pipeline_path, y_key='PTS', limit=5000, corruption=0.35, learning_rate=0.1, epochs=2000, batch_size=10, mean_window=3):
    # training a AE
    labels, mean_predictions, train_set, valid_set, test_set = getNBATrainingData(csv_path, pipeline_path=pipeline_path, y_key=y_key, limit=limit, mean_window=mean_window)
    da = train_dA(labels, train_set, corruption=corruption, learning_rate=learning_rate, training_epochs=epochs, batch_size=batch_size, output_folder='dA_plots')
    pickleModel(model_path, da)

def trainSAE(csv_path, model_path, pipeline_path, y_key='FG%', limit=5000, 
             training_epochs=20, training_lr=0.001, finetune_epochs=1000, finetune_lr=0.1,
             batch_size=1, hidden_layer_sizes=None, mean_window=3,
             corruption_type='mask', corruption_levels=None, n_outs=None):
    if n_outs is None:
        n_outs = 20
    labels, mean_predictions, train_set, valid_set, test_set = getNBATrainingData(csv_path, pipeline_path=pipeline_path, y_key=y_key, limit=limit, n_outs=n_outs, mean_window=mean_window)
    sda, valid_error, test_error = train_SdA(len(labels), train_set, valid_set, test_set, finetune_lr=finetune_lr, pretraining_epochs=training_epochs,
             pretrain_lr=training_lr, training_epochs=finetune_epochs, hidden_layer_sizes=hidden_layer_sizes, 
             corruption_type=corruption_type, corruption_levels=corruption_levels,
             batch_size=batch_size, n_outs=n_outs)
    pickleModel(model_path, sda)

    # get test value of using regress-to-mean prediction
    all_means, all_ys = mean_predictions
    all_means = np.array(all_means)
    all_ys = np.array(all_ys)
    mean_regress_error = np.mean(abs(all_means - all_ys) / float((n_outs - 1))) * 100.0

    return valid_error, test_error, mean_regress_error

class DataPreprocessor(object):

    def __init__(self, *args, **kwargs):
        raise Exception("Not Implemented")

    def normalizeSamples(self): 
        # find bad continuous columns
        self.findBadContCols()

        # normalize cont samples
        samples = np.array(self.cont_samples)
        samples = self.imputer.fit_transform(samples)
        samples = self.scaler.fit_transform(samples)
        if self.clamp is not None:
            # after standard scaling, convert all values abs(v) > clamp to (+/-)clamp value
            samples[samples > self.clamp] = self.clamp
            samples[samples < -self.clamp] = -self.clamp
        if self.minmax is not None:
            samples = self.minmax.fit_transform(samples)
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
        samples = self.cont_samples
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

    def getFeatureLabels(self):
        return list(self.cont_labels) + list(self.cat_labels)

    def transform(self, cont_sample, cat_sample):
        if cont_sample:
            samples = np.array(cont_sample)
            samples = self.imputer.transform(samples)
            samples = self.scaler.transform(samples)
            if self.clamp is not None:
                # after standard scaling, convert all values abs(v) > clamp to (+/-)clamp value
                samples[samples > self.clamp] = self.clamp
                samples[samples < -self.clamp] = -self.clamp
            if self.minmax is not None:
                samples = self.minmax.transform(samples)
            cont_sample = samples[0]
        if cat_sample:
            samples = np.array(cat_sample)
            samples = self.cat_minmax.transform(samples)
            cat_sample = samples[0]
        whole_sample = np.concatenate((cont_sample, cat_sample), axis=0)
        return whole_sample

    def dumpPipeline(self, filepath):
        data = {'clamp': self.clamp,
                'imputer': self.imputer,
                'scaler': self.scaler,
                'minmax': self.minmax,
                'cat_minmax': self.cat_minmax}
        result = cPickle.dump(data, open(filepath,'wb'))
        return result

    def loadPipeline(self, filepath):
        data = cPickle.load(open(filepath,'rb'))
        self.clamp = data['clamp']
        self.imputer = data['imputer']
        self.scaler = data['scaler']
        self.minmax = data['minmax']
        self.cat_minmax = data['cat_minmax']

class StreamDataPreprocessor(DataPreprocessor):

    TARGET_KEYS = ['PTS', 'TRB', 'AST', 'eFG%']

    def __init__(self, input_stream, limit=None, feature_range=None, clamp=None, mean_window=1, pipeline_path=None, lazy=False):
        # import data from stream
        self.cont_labels = None
        self.cat_labels = None
        self.cat_kernel_splits = None
        self.cat_samples = []
        self.cont_samples = []
        self.ids = []
        count = 0
        sys.stdout.write(str(count))
        for row in input_stream:
            id_dict, cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits = row
            if self.cont_labels is None:
                self.cont_labels = np.array(cont_labels)
            if self.cat_labels is None:
                self.cat_labels = np.array(cat_labels)
            if self.cat_kernel_splits is None:
                self.cat_kernel_splits = np.array(cat_feat_splits)
            self.cat_samples.append(cat_features)
            self.cont_samples.append(cont_features)
            self.ids.append(id_dict)
            count += 1
            sys.stdout.write('\r' + str(count) + ' ' * 20)
            sys.stdout.flush()
            if limit and count >= limit:
                sys.stdout.write('\n')
                break
        self.cont_samples = np.array(self.cont_samples)
        self.cat_samples = np.array(self.cat_samples)

        # basic checks
        if not (len(self.cont_samples) == len(self.cat_samples)):
            raise Exception("Inconsistent cat/cont sample counts")
        elif len(self.cont_samples) < 2 and len(self.cat_samples) < 2:
            raise Exception("Need more than one data point")

        # create normalization pipeline, and fit to data if necessary
        if pipeline_path is None:
            self.clamp = clamp
            self.feature_range = feature_range
            self.imputer = Imputer()
            self.scaler = StandardScaler()
            if self.feature_range is not None:
                self.minmax = MinMaxScaler(feature_range=self.feature_range)
                self.cat_minmax = MinMaxScaler(feature_range=self.feature_range)
            else:
                self.minmax = None
                self.cat_minmax = MinMaxScaler(feature_range=(-1,1))
            self.normalizeSamples()
        else:
            self.loadPipeline(pipeline_path)

        # get Y and trendY
        self.Y = None
        self.trendY = None
        self.Means = None
        if not lazy:
            self.calculateY(mean_window=mean_window)
            self.calculateTrendY()

    def calculateY(self, mean_window=1):
        ys = []
        means = []
        to_remove = []
        for i, id_dict in enumerate(self.ids):
            pid = id_dict['player_id']
            gid = id_dict['target_game']

            row = player_game_collection.find_one({"player_id" : pid, "game_id" : gid})
            if not row:
                to_remove.append(i)
                continue

            ts = row['game_time']
            season_start = datetime(year=ts.year, month=8, day=1)
            if season_start > ts:
                season_start = datetime(year=ts.year-1, month=8, day=1)
            history = list(player_game_collection.find({"player_id": pid, "game_time": {"$lt" : row['game_time'], "$gt" : season_start}, "player_team" : row['player_team']}, limit=mean_window, sort=[("game_time",-1)]))
            if len(history) == 0:
                to_remove.append(i)
                continue

            player_targets = {target_key : row.get(target_key) for target_key in self.TARGET_KEYS}
            if None in player_targets.values() or '' in player_targets.values():
                to_remove.append(i)
                continue

            player_history_avgs = self.averageStats(history, self.TARGET_KEYS, weights=None)
            if np.nan in player_history_avgs.values():
                to_remove.append(i)
                continue

            ys.append(player_targets)
            means.append(player_history_avgs)

        # remove the bad samples
        self.cat_samples = np.delete(self.cat_samples, to_remove, axis=0)
        self.cont_samples = np.delete(self.cont_samples, to_remove, axis=0)
        self.ids = np.delete(self.ids, to_remove, axis=0)

        self.Y = ys
        self.Means = means

    def averageStats(self, stats, allowed_keys, weights=None):
        if weights is not None and len(weights) != len(stats):
            raise Exception("Weights not same length as stats")
        trajectories = {k:[] for k in allowed_keys}
        for stat in stats:
            for k in allowed_keys:
                trajectories[k].append(stat.get(k))
        # average out the stats
        for k,v in trajectories.iteritems():
            filtered_values = []
            filtered_weights = []
            for i,value in enumerate(v):
                if value is not None and value != '':
                    new_weight = weights[i] if weights is not None else 1.0
                    filtered_values.append(value)
                    filtered_weights.append(new_weight)
            if len(filtered_values) > 0:
                trajectories[k] = np.average(filtered_values, weights=filtered_weights)
            else:
                trajectories[k] = np.nan
        return trajectories


    def calculateTrendY(self):
        pass

    def getY(self, key):
        if key not in self.TARGET_KEYS or self.Y is None or self.Means is None:
            raise Exception("Y not computed")
        to_return = [_[key] for _ in self.Y]
        to_return_means = [_[key] for _ in self.Means]
        return to_return, to_return_means

    def getClassY(self, key, n_outs, clamp=2):
        if key not in self.TARGET_KEYS or self.Y is None or self.Means is None:
            raise Exception("Y not computed")
        clamp = abs(clamp)
        raw_values = [_[key] for _ in self.Y]
        mean_values = [_[key] for _ in self.Means]
        # put into class buckets
        scaler = StandardScaler()
        # normalize
        norm_raw = scaler.fit_transform(raw_values)
        norm_means = scaler.transform(mean_values)
        # put into buckets
        buckets = np.linspace(-clamp,clamp,n_outs-1)
        placements = np.digitize(norm_raw, buckets)
        class_means = np.digitize(norm_means, buckets)
        '''
        # convert to one-hot encoding, activating the specified bucket
        to_return = []
        for p in placements:
            vector = np.zeros(n_outs)
            vector[p] = 1.0
            to_return.append(vector)
        return to_return
        '''
        return placements, class_means

    def getTrendY(self, key):
        if key not in self.TARGET_KEYS or self.trendY is None:
            raise Exception("Y not computed")
        to_return = [_[key] for _ in self.trendY]
        return to_return


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

    def __init__(self, X, Y, test_size=0.25, feature_labels=None, restarts=10):
        super(RandomForestValidationFramework, self).__init__(X, Y, test_size)
        self.trained = False
        self.feature_labels = feature_labels

    def save_model(self, model_file):
        pass

    def load_model(self, model_file):
        pass

    def train(self, zip_data=None):
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

        '''
        # for random forest
        model = make_pipeline(PolynomialFeatures(1), RandomForestRegressor(n_estimators=400, n_jobs=-1))
        model.fit(self.X_train, self.Y_train)
        '''
        # for gradient boosting
        model = make_pipeline(PolynomialFeatures(1), GradientBoostingRegressor(n_estimators=400))
        model.fit(self.X_train, self.Y_train)
        
        '''
        featimp = sorted(zip(self.feature_labels, model.feature_importances_), reverse=True, key=lambda x: x[1])
        for l, imp in featimp:
            print "%s : %s" % (l,imp)
        '''
        self.model = model
        self.trained = True

    def test(self, graph=False):
        train_predicted = self.model.predict(self.X_train)
        test_predicted = self.model.predict(self.X_test)
        train_error = mean_squared_error(self.Y_train, train_predicted)
        test_error = mean_squared_error(self.Y_test, test_predicted)

        train_zipped = sorted(zip(self.Y_train, train_predicted), key=lambda x: x[0])
        test_zipped = sorted(zip(self.Y_test, test_predicted), key=lambda x: x[0])

        train_real = [a for a,b in train_zipped]
        train_predicted = [b for a,b in train_zipped]
        test_real = [a for a,b in test_zipped]
        test_predicted = [b for a,b in test_zipped]

        if graph:
            fig2 = plt.figure()
            p2 = plt.plot(train_predicted)
            p3 = plt.plot(train_real)
            fig2.suptitle('TRAIN', fontsize=20)

            fig3 = plt.figure()
            p4 = plt.plot(test_predicted)
            p5 = plt.plot(test_real)
            fig3.suptitle('TEST', fontsize=20)

            plt.show(block=True)
        return (train_error, test_error)


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

        self.model = model_three
        self.trained = True

    def test(self, graph=False):
        train_predicted = self.model.predict(self.X_train)
        test_predicted = self.model.predict(self.X_test)
        train_error = mean_squared_error(self.Y_train, train_predicted)
        test_error = mean_squared_error(self.Y_test, test_predicted)

        train_zipped = sorted(zip(self.Y_train, train_predicted), key=lambda x: x[0])
        test_zipped = sorted(zip(self.Y_test, test_predicted), key=lambda x: x[0])

        train_real = [a for a,b in train_zipped]
        train_predicted = [b for a,b in train_zipped]
        test_real = [a for a,b in test_zipped]
        test_predicted = [b for a,b in test_zipped]

        if graph:
            fig1 = plt.figure()
            p1 = plt.plot(self.model.steps[1][1].coef_, label="coeffs")
            plt.setp(p1, color='b', linewidth=2.0)
            fig1.suptitle('COEFFS', fontsize=20)

            fig2 = plt.figure()
            p2 = plt.plot(train_predicted)
            p3 = plt.plot(train_real)
            fig2.suptitle('TRAIN', fontsize=20)

            fig3 = plt.figure()
            p4 = plt.plot(test_predicted)
            p5 = plt.plot(test_real)
            fig3.suptitle('TEST', fontsize=20)

            plt.show(block=True)
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


def optimizeMeanWindow(file_path, windows, limit, y_key, n_outs):
    dproc = StreamDataPreprocessor(streamCSV(file_path), limit=limit, clamp=3, feature_range=(0,1), lazy=True)
    for w in sorted(windows,reverse=True):
        dproc.calculateY(mean_window=w)
        all_ys, all_means = dproc.getClassY(y_key, n_outs, clamp=3)
        print all_ys
        print all_means
        all_means = np.array(all_means)
        all_ys = np.array(all_ys)
        regress_to_mean_error = np.mean(abs(all_means - all_ys) / float((n_outs - 1)))
        print "REGRESS TO MEAN ERROR (window: %s): %s" % (w, regress_to_mean_error)


def optimizeSAEHyperParameters(csv_file, model_folder, result_file, limit, batch, y_key, n_outs, nHLay_choices, nHUnit_choices, 
                               lRate_choices, lRateSup_choices, nEpoq_choices, v_choices):
    labels = ['nHLay', 'nHUnit', 'lRate', 'lRateSup', 'nEpoq', 'v_type', 'v', 'n_outs', 'batch_size', 'limit', 'valid_error', 'test_error', 'mean_regress_error']
    results = []
    for nHLay, nHUnit, lRate, lRateSup, nEpoq, v_choices in itertools.product(nHLay_choices, nHUnit_choices, lRate_choices, lRateSup_choices, nEpoq_choices, v_choices):
        n_type, n_choices = v_choices
        for v in n_choices:   
            model_file = os.path.join(model_folder,"%slayer_%sunit_%slrate_%ssuplrate_%sbatch%sepoq_%s%s_%sout_%s_model" % (nHLay, nHUnit, lRate, lRateSup, batch, nEpoq, v, n_type, n_outs, y_key))
            pipeline_file = os.path.join(model_folder,"%slimit_sae_pipeline" % (limit,))
            hidden_layer_sizes = [nHUnit] * nHLay
            corruption_levels = [v] * nHLay
            start = time.time()
            valid_error, test_error, mean_regress_error = trainSAE(csv_file, model_file, pipeline_file, y_key=y_key, limit=limit, 
                training_epochs=nEpoq, training_lr=lRate, finetune_epochs=1000, finetune_lr=lRateSup,
                batch_size=batch, n_outs=n_outs, hidden_layer_sizes=hidden_layer_sizes, 
                corruption_type=n_type, corruption_levels=corruption_levels,
                mean_window=0)
            end = time.time()
            print "Took %s seconds" % (end-start,)
            result = [nHLay, nHUnit, lRate, lRateSup, nEpoq, n_type, v, n_outs, batch, limit, valid_error, test_error, mean_regress_error]
            for k,v in zip(labels, results):
                print "%s: %s" % (k,v)
            results.append(result)
    # write results
    out_file = open(result_file, 'wb')
    writer = csv.writer(out_file)
    writer.writerow(labels)
    for r in results:
        writer.writerow(r)



if __name__ == "__main__":
    
    csv_file = "/usr/local/fsai/analysis/data/nba_stats_1.csv"
    model_folder = "/usr/local/fsai/analysis/data"
    model_file = '/usr/local/fsai/analysis/data/ae_2'
    sae_file = '/usr/local/fsai/analysis/data/sae_1'
    pipeline_file = '/usr/local/fsai/analysis/data/pipeline_3'
    
    # optimize mean prediction window
    '''
    windows = np.arange(0,1)
    optimizeMeanWindow(csv_file, windows, limit=1000, y_key='TRB', n_outs=15)
    sys.exit(1)
    '''

    # training SAE
    limit = 1000
    batch = 5
    y_key = 'TRB'
    n_outs = 20
    nHLay_choices = [1]
    nHUnit_choices = [2000]
    lRate_choices = [0.005]
    lRateSup_choices = [0.005]
    nEpoq_choices = [10]
    #v_choices = [('gaussian',[0.10]),('mask',[0.15])]
    v_choices = [('gaussian',[0.10])]
    result_file = "/usr/local/fsai/analysis/data/results.csv"
    optimizeSAEHyperParameters(csv_file, model_folder, result_file, limit, batch, y_key, n_outs, nHLay_choices, nHUnit_choices, 
                               lRate_choices, lRateSup_choices, nEpoq_choices, v_choices)
    sys.exit(1)


    # training an AE
    '''
    trainAE(csv_file, model_file, pipeline_file, y_key='TRB', limit=50000, learning_rate=0.15, epochs=1000, batch_size=50, corruption=0.2)
    sys.exit(1)
    '''

    # fitting a linear model off autoencoder input for player
    '''
    pid = "curryst01"
    X, Y = encodePlayerInput(pid, model_file, pipeline_file, y_key='eFG%', limit=100, time=datetime(year=2014,month=10,day=1), min_count=None)
    model = LinearRegressionValidationFramework(X, Y, test_size=0.25, restarts=10)
    model.train()
    print model.test(graph=True)
    sys.exit(1)
    '''

    # fitting a linear model to csv dataset directly
    '''
    dproc = StreamDataPreprocessor(streamCSV(csv_file), limit=3000, clamp=None)
    Y = dproc.getY('TRB')
    X = dproc.getAllSamples()
    labels = dproc.getFeatureLabels()
    model = LinearRegressionValidationFramework(X, Y, test_size=0.25, restarts=10)
    #model = RandomForestValidationFramework(X, Y, test_size=0.3, feature_labels=labels, restarts=10)
    model.train()
    print model.test(graph=True)
    sys.exit(1)
    '''
