# Import libraries
import numpy as np
import pandas as pd

#used for feature analysis
import matplotlib.pyplot as plt

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"
# Note: The last column 'passed' is the target/label, all other are feature columns

# TODO: Compute desired values - replace each '?' with an appropriate expression/function call
n_students = len(student_data)
n_features = len(student_data.columns)
n_passed = len(student_data[student_data.passed == "yes"])
n_failed = n_students - n_passed
grad_rate = n_passed * 1.0 / n_students
print "Total number of students: {}".format(n_students)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Number of features: {}".format(n_features)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_all.head()  # print the first 5 rows

# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty
        
    # Check each column
    for col, col_data in X.iteritems():        
        #if we are dealing with absences then bin descretize it into bands of 10
        #if col_data.name == 'absences':
        #    col_data = col_data.apply(lambda x: (int) (x * 1.0 /25))
        
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'      
            
        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

X_all = preprocess_features(X_all)
print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))

# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
# Create a list with the index numbers randomised
random_index = np.random.permutation(num_all)
random_train_index = random_index[:num_train]
random_test_index = random_index[num_train:]

def permutate():
    X_train = X_all.loc[random_train_index]
    y_train = y_all[random_train_index]
    X_test = X_all.loc[random_test_index]
    y_test = y_all[random_test_index]
    print "Training set: {} samples".format(X_train.shape[0])
    print "Test set: {} samples".format(X_test.shape[0])
    return (X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = permutate()

# Note: If you need a validation set, extract it from within training data

import time

def train_classifier(clf, X_train, y_train):
    #print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    #print "Done!\nTraining time (secs): {:.3f}".format(end - start)
    return (end - start)

from sklearn.metrics import f1_score

def predict_labels(clf, features, target):
    #print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    #print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return (end - start, f1_score(target.values, y_pred, pos_label='yes'))

# Train and predict using different training set sizes
def train_predict(clf, X_train, y_train, X_test, y_test, tuned='a'):
    #print "------------------------------------------"
    #print "Training set size: {}".format(len(X_train))
    a = train_classifier(clf, X_train, y_train)
    b = predict_labels(clf, X_train, y_train)
    c = predict_labels(clf, X_test, y_test)
    #print "F1 score for training set: {}".format(b)
    #print "F1 score for test set: {}".format(c)
    results.append([clf.__class__.__name__,
        len(X_train),
        a,
        b[0], b[1],
        c[0], c[1], tuned])

results = []
def run_samples(clf, X_train, y_train, X_test, y_test):
    for max_items in [100, 200, 300]:
        train_predict(clf, X_train[:max_items], y_train[:max_items], X_test, y_test)



from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


#run_samples(MultinomialNB(), X_train, y_train, X_test, y_test)

from sklearn import metrics
from sklearn.grid_search import GridSearchCV



def train_score(clf, params, X_train, y_train):
    scorer = metrics.make_scorer(metrics.f1_score, pos_label='yes')
    gs = GridSearchCV(clf, params, scoring=scorer, cv=3)
    gs.fit(X_train, y_train)
    print '*' * 20
    print clf.__class__.__name__
    print "Params %s" % gs.best_params_
    print "Score %.4f" % gs.best_score_
    return gs.best_estimator_

good_cols = ['failures',
            'absences',
            'schoolsup',
            #'goout',b
            #'paid',
            #'guardian_other',
            # 'studytime',
            # 'reason_course',
            # 'famrel',
            # 'reason_other',
            # 'freetime'
            ]
all_cols = X_all.dtypes.index
for a in all_cols:
    if not (a in good_cols):
        print "Removing %s" % a
        #del X_train[a]
        #del X_test[a]


def run_class(clf1, params, X_train, y_train, X_test, y_test):
    run_samples(clf1, X_train, y_train, X_test, y_test)

    tuned_clf = train_score(clf1, params, X_train, y_train)
    train_predict(tuned_clf, X_train, y_train, X_test, y_test, 'tuned')

    #Drop unnecessary columns and see how we go
    X_train_few = X_train.copy()
    X_test_few = X_test.copy()
    for a in all_cols:
        if not (a in good_cols):
            del X_train_few[a]
            del X_test_few[a]

    tuned_clf = train_score(clf1, params, X_train_few, y_train)
    train_predict(tuned_clf, X_train_few, y_train, X_test_few, y_test, 'feature_drop')



for i in xrange(1,100):
    print '#%d' % i
    X_train, y_train, X_test, y_test = permutate()

    run_class(KNeighborsClassifier(), {'n_neighbors': range(1,10) }, X_train, y_train, X_test, y_test)
    run_class(DecisionTreeClassifier(), {'max_depth': range(1,5)}, X_train, y_train, X_test, y_test)
    run_class(SVC(), {'C': [.1, .25, .4, .5, .6, 1.0, 2.0, 4.0], 'shrinking': [True, False], 'kernel': ['rbf', 'rbf']}, X_train, y_train, X_test, y_test)


#train_predict(clf, X_train, y_train, X_test, y_test)

for l in results:
    print "{:20}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(*l)

from pandas import DataFrame
data = DataFrame(results)
data.columns = ['classifier', 'sample_size', 'train_time', 'predict_time', 'train_percent', 'test_time', 'test_percent', 'tuned']

sample = data.groupby(['classifier', 'sample_size','tuned'])

for a in ['train_time', 'predict_time', 'train_percent', 'test_time', 'test_percent']:
    print sample[a].mean()



