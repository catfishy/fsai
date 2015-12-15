import sys
from statsETL.db.mongolib import *


# # team output vectors
# for row in nba_team_outputs_collection.find({}):
#     row_id = row['_id']
#     old_data = row['input']
#     new_data = old_data['output']
#     nba_conn.updateDocument(nba_team_outputs_collection, row_id, {'input': new_data}, upsert=False)   

# # team season avgs
# for row in nba_season_averages_collection.find({}):
#     row_id = row['_id']
#     if "season_stats" not in row and 'input' in row:
#         # already conformed
#         continue
#     elif "season_stats" not in row:
#         # delete row
#         nba_season_averages_collection.remove(row_id)
#         continue
#     out_data = row['season_stats']
#     vars_df = pd.read_json(out_data['vars'])
#     vars_df = pd.DataFrame(vars_df.ix['mean']).T
#     vars_df.reset_index(drop=True, inplace=True)
#     vars_df['id'] = 'vars'
#     means_df = pd.read_json(out_data['means'])
#     means_df = pd.DataFrame(means_df.ix['mean']).T
#     means_df.reset_index(drop=True, inplace=True)
#     means_df['id'] = 'means'
#     new_data = pd.concat((means_df, vars_df))
#     new_data.set_index('id', inplace=True)
#     new_data_json = new_data.to_json()
#     nba_season_averages_collection.update({'_id': row_id}, {'$unset': {'season_stats':1}})
#     nba_conn.updateDocument(nba_season_averages_collection, row_id, {'input': new_data_json}, upsert=False)   

# # # team input
# for row in nba_team_vectors_collection.find({}):
#     row_id = row['_id']
#     old_data = row['input']
#     if not isinstance(old_data,dict):
#         # already conformed
#         continue
#     means_df = pd.read_json(old_data['means'])
#     means_df = pd.DataFrame(means_df.ix['mean']).T
#     means_df.reset_index(drop=True, inplace=True)
#     means_df['id'] = 'means'
#     vars_df = pd.read_json(old_data['variances'])
#     vars_df = pd.DataFrame(vars_df.ix['mean']).T
#     vars_df.reset_index(drop=True, inplace=True)
#     vars_df['id'] = 'vars'
#     new_data = pd.concat((means_df, vars_df))
#     new_data.set_index('id', inplace=True)
#     new_data_json = new_data.to_json()
#     nba_conn.updateDocument(nba_team_vectors_collection, row_id, {'input': new_data_json}, upsert=False)   

# player input
for row in nba_player_vectors_collection.find({}, no_cursor_timeout=True):
    row_id = row['_id']
    old_data = row['input']
    safe_keys = ['days_rest', 'home/road', 'location', 'position']
    new_data = {_:old_data[_] for _ in safe_keys if _ in old_data}

    if len(old_data) == 0:
        print "empty"
        continue
    if 'own' in old_data:
        # already conformed
        try:
            df_own = pd.read_csv(old_data['own'])
            df_own.set_index('id', inplace=True)
            try:
                df_own.loc['expmean'] = df_own.loc['expmean'].apply(lambda x: x['0'])
                df_own.loc['expvar'] = df_own.loc['expvar'].apply(lambda x: x['0'])
            except Exception as e:
                pass
            new_data['own'] = df_own.to_csv()
        except Exception as e:
            #print "own not csv:"
            new_data['own'] = old_data['own']
        try:
            df_own = pd.read_json(old_data['own'])
            try:
                df_own.loc['expmean'] = df_own.loc['expmean'].apply(lambda x: x['0'])
                df_own.loc['expvar'] = df_own.loc['expvar'].apply(lambda x: x['0'])
            except Exception as e:
                pass
            new_data['own'] = df_own.to_csv()
        except Exception as e:
            #print "own not json:"
            new_data['own'] = old_data['own']
        if 'trend' in old_data:
            try:
                df_trend = pd.read_json(old_data['trend']) 
                df_trend_csv = df_trend.to_csv()
                new_data['trend'] = df_trend_csv
            except Exception as e:
                #print "trend conformed:"
                new_data['trend'] = old_data['trend']
        nba_conn.updateDocument(nba_player_vectors_collection, row_id, {'input': new_data}, upsert=False)  
    else:
        print "new conversion"
        # aggregate own
        try:
            expmean = pd.read_json(old_data['expmean'], typ='frame')
        except Exception as e:
            expmean = pd.DataFrame(pd.read_json(old_data['expmean'], typ='series')).T
        expmean['id'] = 'expmean'
        try:
            expvar = pd.read_json(old_data['expvar'],  typ='frame')
        except Exception as e:
            expvar=pd.DataFrame(pd.read_json(old_data['expvar'],  typ='series')).T
        expvar['id'] = 'expvar'
        sample_means=pd.DataFrame(pd.read_json(old_data['means']).ix['mean']).T
        sample_means['id'] = 'means'
        sample_vars=pd.DataFrame(pd.read_json(old_data['variances']).ix['mean']).T
        sample_vars['id'] = 'vars'
        sample_df = pd.concat([sample_means, sample_vars, expmean, expvar])
        sample_df.set_index('id', inplace=True)
        new_data['own'] = sample_df.to_csv()

    # aggregate trend if possible
    if 'trend' in old_data:
        new_trend = {}
        for k,v in old_data['trend'].iteritems():
            if isinstance(v,dict):
                print "dict"
                if v['var'] is None:
                    var = pd.DataFrame()
                else:
                    var = pd.DataFrame(pd.read_json(v['var'],typ='series')).T
                var['id'] = 'var'
                if v['mean'] is None:
                    mean = pd.DataFrame()
                else:
                    mean = pd.DataFrame(pd.read_json(v['mean'],typ='series')).T
                mean['id'] = 'mean'
                trend_df = pd.concat([mean, var])
                trend_df.set_index('id', inplace=True)
                print trend_df
                new_trend[k] = trend_df.to_csv()
            else:
                print "string"
                trend_df = pd.read_json(v)
                print trend_df
        break
        new_data['trend'] = new_trend

    nba_conn.updateDocument(nba_player_vectors_collection, row_id, {'input': new_data}, upsert=False)   

sys.exit(1)

# player splits

# player against
    # aggregate against
    # new_against = {}
    # for k,v in old_data['against'].iteritems():
    #     var = pd.DataFrame(pd.read_json(v['var']).ix['mean']).T
    #     var['id'] = 'var'
    #     mean = pd.DataFrame(pd.read_json(v['mean']).ix['mean']).T
    #     mean['id'] = 'mean'
    #     trend_mean = pd.DataFrame(pd.read_json(v['trend_mean']).ix['mean']).T
    #     trend_mean['id'] = 'trend_mean'
    #     trend_var = pd.DataFrame(pd.read_json(v['trend_var']).ix['mean']).T
    #     trend_var['id'] = 'trend_var'
    #     against_df = pd.concat([mean, var, trend_mean, trend_var])
    #     against_df.set_index('id', inplace=True)
    #     new_against[k] = against_df.to_json()
    # new_data['against'] = new_against





