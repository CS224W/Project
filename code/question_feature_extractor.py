# Author: Matt Terry <matt.terry@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function

import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from operator import itemgetter
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from dateutil.parser import parse as dt_parse
from src.DataReader import DataReader
from src.constants import SPLIT_TIME, STORE_LOCATION, DATA_ROOT, POSTS_FILE, VOTES_FILE

import logging
logging.basicConfig()
min_df = 2

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class PostStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, StatColumns):
        print('Extracting post stats ...')
        return [{'FavoriteCount': row[0],
                 'OwnerUserId': row[1],
                 'Score' : row[2]}
                for row in StatColumns]


class ColumnExtractor(BaseEstimator, TransformerMixin):
    """
    """

    def fit(self, x, y=None):
        return self

    def transform(self, dataFrames):
        print('Extracting Important Columns ...')
        cols = ['Title', 'Body', 'CreationDate', 'FavoriteCount', 'OwnerUserId', 'Score', 'Tags', 'AcceptedAnswerId']
        all_important_columns = dataFrames[cols]    

        #print (dataFrames[dataFrames.Id == 1])
        
        features = np.recarray(shape=(len(all_important_columns),),
                               dtype=[('title', object), ('body', object), ('stats_columns', object)])
        for i, row in enumerate(all_important_columns.values):

            features['title'][i] = row[0]
            features['body'][i] = row[1]
            features['stats_columns'][i] = row[3:6]
        
                
        print('Done extracting Important Columns ...')

        return features


pipeline = Pipeline([
    # Extract the subject & body
    ('titlebodystats', ColumnExtractor()),

    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from the post's subject line
            ('title', Pipeline([
                ('selector', ItemSelector(key='title')),
                ('tfidf', TfidfVectorizer(min_df)),
            ])),

            # Pipeline for standard bag-of-words model for body
            ('body_bow', Pipeline([
                ('selector', ItemSelector(key='body')),
                ('tfidf', TfidfVectorizer()),
                ('best', TruncatedSVD(n_components=50)),
            ])),

            # Pipeline for pulling ad hoc features from post's body
            ('post_stats', Pipeline([
                ('selector', ItemSelector(key='stats_columns')),
                ('stats', PostStats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),

        ],

        # weight components in FeatureUnion
        transformer_weights={
            'title': 0.8,
            'body_bow': 0.5,
            'post_stats': 1.0,
        },
    )),

    # Use a SVC classifier on the combined features
    ('svc', SVC(kernel='linear')),
])

# limit the list of categories to make running this example faster.

data_directory = DATA_ROOT
post_reader = DataReader(os.path.join(data_directory, POSTS_FILE), True)
post_reader.read_data()
pdf = post_reader._df
question_df = pdf[pdf.PostTypeId == 1]

selected_questions_df = question_df[question_df.AnswerCount > 0]
selected_questions_df = selected_questions_df[selected_questions_df.Tags.notnull()]
selected_questions_df = selected_questions_df[selected_questions_df.AcceptedAnswerId.notnull()]
selected_questions_df = selected_questions_df[selected_questions_df.OwnerUserId.notnull()]
selected_questions_df = selected_questions_df[selected_questions_df.FavoriteCount.notnull()]

selected_questions_df_train = selected_questions_df[selected_questions_df.CreationDate <= dt_parse(SPLIT_TIME)]

targets_df_train = selected_questions_df_train['AcceptedAnswerId']
targets_train = targets_df_train.values

selected_questions_df_test = selected_questions_df[selected_questions_df.CreationDate > dt_parse(SPLIT_TIME)]
targets_df_test = selected_questions_df_test['AcceptedAnswerId']
targets_test = targets_df_test.values

pipeline.fit(selected_questions_df_train, targets_train)
print('Running Prediction ...')
y = pipeline.predict(selected_questions_df_test)

print(classification_report(y, targets_test))