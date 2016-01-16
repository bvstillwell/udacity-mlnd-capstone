import numpy as np
import pandas as pd

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

y_all = y_all.replace(['yes', 'no'], [1, 0])


def bin_data(series, bins):
    data = np.digitize(series.values, bins)
    result = pd.Series(data, index=series.index, name=series.name) 
    return result


# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        #if we are dealing with absences then bin descretize it into bands of 10
        if col_data.name == 'absences':
          col_data = bin_data(col_data, [2,6,10,20])

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

unique_value_per_feature = [(col, X_all[col].unique()) for col in X_all.columns]
for a in unique_value_per_feature:
    print "%s %s" % a

print "y_all", y_all.unique()


# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
random_index = np.random.permutation(num_all-1)
random_train_index = random_index[:num_train]
random_test_index = random_index[num_train:] 


print max(random_train_index), min(random_train_index)
X_train = X_all.loc[random_train_index]
y_train = y_all[random_train_index]
X_test = X_all.loc[random_test_index]
y_test = y_all[random_test_index]
print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])



# Train a model
import time

def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)
    return end - start

# TODO: Choose a model, import it and instantiate an object
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5)

# Fit model to training data
# train_classifier(clf, X_train, y_train)  # note: using entire training set here
# print clf  # you can inspect the learned model by printing it

# Predict on training set and compute F1 score
from sklearn.metrics import f1_score

def predict_labels(clf, features, target):
    print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return (end-start, f1_score(target.values, y_pred, pos_label=1))

#train_f1_score = predict_labels(clf, X_train, y_train)
#print "F1 score for training set: {}".format(train_f1_score)

# # Predict on test data
# print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))

# Train and predict using different training set sizes
def train_predict(clf, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    print "Training set size: {}".format(len(X_train))
    train_time = train_classifier(clf, X_train, y_train)
    print "F1 score for training set: {}".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))

    # a = len(X_train)
    # b = train_time
    # c, d = predict_labels(clf, X_train, y_train)
    # e, f = predict_labels(clf, X_test, y_test)
    # results.append((clf.__class__.__name__, a, b, c, d, e, f))

# results = []

# # TODO: Run the helper function above for desired subsets of training data
# # Note: Keep the test set constant
# for max_items in [100, 200, 300]:
#     train_predict(clf, X_train[:max_items], y_train[:max_items], X_test, y_test)

# # TODO: Train and predict using two other models
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()

# for max_items in [100, 200, 300]:
#     train_predict(clf, X_train[:max_items], y_train[:max_items], X_test, y_test)


# # TODO: Train and predict using two other models
# from sklearn.svm import SVC
# clf = SVC()

# for max_items in [100, 200, 300]:
#     train_predict(clf, X_train[:max_items], y_train[:max_items], X_test, y_test)



# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier

# for max_items in [100, 200, 300]:
#     train_predict(DecisionTreeClassifier(max_depth=3), X_train[:max_items], y_train[:max_items], X_test, y_test)

# for l in results:
#     print "{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(*l)
#     #print ["%.4f" % a for a in l]




# parameters = {
#     #'criterion':('gini', 'entropy'), 
#     #'max_features': range(1, 31),
#     'max_depth': range(1,9),
#     'min_samples_split' : range(1,5)
#     }

# clf = DecisionTreeClassifier()


from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


scorer = metrics.make_scorer(metrics.f1_score)


def train_score(clf, params):
    scores_pre = cross_val_score(clf, X_all, y_all, scoring=scorer, cv=5)

    gs = GridSearchCV(clf, params, scoring=scorer, cv=5)
    gs.fit(X_all, y_all)

    scores_post = cross_val_score(gs.best_estimator_, X_all, y_all, scoring=scorer, cv=5)

    print '*' * 20
    print clf.__class__.__name__
    print "Params %s" % gs.best_params_
    print "Score %.4f" % gs.best_score_
    return gs.best_estimator_




from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(.5)
result = sel.fit(X_all)
remove = []
for x in xrange(0, len(sel.variances_)):
    if(sel.variances_[x] < 0.0):
        remove.append(X_all.dtypes.index[x])
for a in remove:
    print "Removing %s" % a
    del X_all[a]


good_cols = ['failures',
            'absences',
            #'schoolsup',
            #'goout',
            # 'paid',
            # 'guardian_other',
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
        #del X_all[a]



train_score(KNeighborsClassifier(), {'n_neighbors' : range(1,15), 'weights' : ['uniform', 'distance']})
clf = train_score(DecisionTreeClassifier(), {'max_depth': range(2,9), 'min_samples_split' : range(1,15)})
importance = sorted(enumerate(clf.feature_importances_), key=lambda x: x[1], reverse=True)
for (a, b) in importance:
    if b > .01:
        print "%.3f - %s" % (b, X_all.dtypes.index[a])

#train_score(SVC(), {'kernel': ['linear', 'poly', 'rbf']})
train_score(SVC(degree=1), {'kernel': ['linear', 'poly', 'rbf']})


