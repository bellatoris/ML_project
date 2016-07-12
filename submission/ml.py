import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
from xgboost.sklearn import XGBClassifier

datadir = "./"
datafile = "my_data.csv"

def mapper(df):
    df["time_remaining"] = df["minutes_remaining"] * 60 + df["seconds_remaining"]
    x_mapper = DataFrameMapper([
        (u'loc_x', None),
        (u'loc_y', None),
        (u'minutes_remaining', None),
        (u'period', None),
        (u'seconds_remaining', None),
        (u'shot_distance', None),
        (u'playoffs', LabelBinarizer()),
        (u'action_type_num', None),
        (u'combined_shot_type_num', None),
        (u'season_num', None),
        (u'shot_type_num', None),
        (u'shot_zone_area_num', None),
        (u'shot_zone_basic_num', None),
        (u'shot_zone_range_num', None),
        (u'matchup', LabelBinarizer()),
        (u'shot_id', None),
        (u'opponent_num',None),
        (u'time_remaining', None),
        (u'last_moment', None),
        (u'hna', None),
        (u'action1', None),
        (u'action2', None),
        (u'action3', None),
        (u'action4', None),
        (u'action5', None),
        (u'action6', None),
        (u'action8', None),
        (u'action9', None),
        (u'action10', None),
        (u'action11', None),
        (u'action13', None),
        (u'action14', None),
        (u'action15', None),
        (u'action16', None)
        ])
    x_mapper.fit(df)
    y_mapper = DataFrameMapper([
        (u'shot_made_flag', None),
        ])
    y_mapper.fit(df)
    return x_mapper, y_mapper


def xgboost_mappedvec(df):
    x_mapper, y_mapper = mapper(df)
    train_df, test_df, valid_df = split(df)
    train_x_vec = x_mapper.transform(train_df.copy())
    train_y_vec = y_mapper.transform(train_df.copy())
    valid_x_vec = x_mapper.transform(valid_df.copy())
    valid_y_vec = valid_df['shot_made_flag']
    test_x_vec = x_mapper.transform(test_df.copy())

    clf = XGBClassifier(max_depth=7, learning_rate=0.01, n_estimators=620, subsample=0.92, colsample_bytree=0.53, seed=0)

    clf.fit(train_x_vec, train_y_vec)
    test_y_vec = clf.predict_proba(test_x_vec)[:, 1]
    result_y_vec = clf.predict_proba(valid_x_vec)[:, 1]
    ll = logloss(valid_y_vec, result_y_vec)
    print(ll)
    return test_y_vec

def split(df):
    n = 20000;
    train_df = df[~np.isnan(df["shot_made_flag"])]
    valid_df = train_df[n:len(train_df)];
    # train_df = train_df[1:n];
    test_df = df[np.isnan(df["shot_made_flag"])]
    return train_df, test_df, valid_df


def makesubmission(predict_y, savename="submission99.csv"):
    submit_df = pd.read_csv(datadir + "sample_submission.csv")
    submit_df["shot_made_flag"] = predict_y
    submit_df = submit_df.fillna(np.nanmean(predict_y))
    submit_df.to_csv(savename, index=False)


def logloss(act, pred):
    epsilon = 1e-15
    act = act.astype(float)
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

if __name__ == "__main__":
    df = pd.read_csv(datadir + datafile)

    # xgboost
    predict_y = xgboost_mappedvec(df)
    makesubmission(predict_y, savename="submit.csv")