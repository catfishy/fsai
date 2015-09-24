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
        raise Exception("Not Implemented")

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

    def getRunningAverage(self):
        return copy.deepcopy(self.runningAverage)

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

