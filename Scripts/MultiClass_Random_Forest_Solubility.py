import pandas as pd
import numpy as np
from sklearn import preprocessing
from pickle import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image
from sklearn.metrics import classification_report

path1='/home/joao_ines/Documents/SLE/new_modeling/training_set_final.csv'
path2='/home/joao_ines/Documents/SLE/new_modeling/test_set_final.csv'

training_data = pd.read_csv(path1)
test_data = pd.read_csv(path2)

training_data.set_index(training_data.ID.values, inplace=True)
training_data.drop(columns=['ID'], axis=1, inplace=True)

test_data.set_index(test_data.ID.values, inplace=True)
test_data.drop(columns=['ID'], axis=1, inplace=True)

y_tr = training_data['class']
training_data = training_data.drop(columns=['Classification', 'class'], axis=1)

y_test = test_data['class']
test_data = test_data.drop(columns=['Classification', 'class'], axis=1)

path3='/home/joao_ines/Documents/SLE/new_modeling/scaler_minmax.pkl'

#minmax scaling
scaler = preprocessing.MinMaxScaler()
# fit scaler on the training dataset
scaler.fit(training_data)
# transform the training dataset
training_data_scaled = pd.DataFrame(scaler.transform(training_data), index=training_data.index, columns=training_data.columns)
# save the scaler
dump(scaler, open(path3, 'wb'))
# load the scaler
scaler = load(open(path3, 'rb'))
# transform the test dataset
test_data_scaled = pd.DataFrame(scaler.transform(test_data), index=test_data.index, columns=test_data.columns)

path4='Documents/SLE/new_ext_ts/x_ext_ts_scaled.csv'
path5='Documents/SLE/new_ext_ts/y_ext_set_final.csv'

#load externat test set
x_ext_test = pd.read_csv(path4)
x_ext_test.set_index(x_ext_test.ID.values, inplace=True)
x_ext_test.drop(columns=['ID'], axis=1, inplace=True)

y_ext_test = pd.read_csv(path5)
y_ext_test.set_index(y_ext_test.ID.values, inplace=True)
y_ext_test.drop(columns=['ID'], axis=1, inplace=True)

#proximity matrix function
def proximityMatrix(model, X, normalize=True):      

    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    a = terminals[:,0]
    proxMat = 1*np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:,i]
        proxMat += 1*np.equal.outer(a, a)

    if normalize:
        proxMat = proxMat / nTrees

    return proxMat

train = training_data_scaled
teste = test_data_scaled

#class weight is used to punish over-represented classes
class_weight = dict({0:0.9, 1:1.6, 2:2.9})

#define RF parameters
RF_model = RandomForestClassifier(bootstrap=True, class_weight=class_weight, criterion='gini',
                       max_depth=10, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.009, min_impurity_split=None,
                       min_samples_leaf=3, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=11, n_jobs=-1,
                       oob_score=True, random_state=10, verbose=0,
                       warm_start=False)
#fit RF model
RF_model.fit(train.values, y_tr)

# Extract single tree
estimator = RF_model.estimators_[1]


# Export as dot file
export_graphviz(estimator, out_file='/home/joao_ines/Documents/SLE/RF-tree/RF_tree_training.dot', 
                feature_names = train.columns.values,
                class_names = ['Soluble', 'Moderate Soluble', 'Insoluble'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
call(['dot', '-Tpng', '/home/joao_ines/Documents/SLE/RF-tree/RF_tree_training.dot', '-o', '/home/joao_ines/Documents/SLE/RF-tree/RF_tree_training1.png', '-Gdpi=600'])

#Out of bag score calculation
print('Out Of Bag score :' , RF_model.oob_score_)

# plot the 15 most relevant descriptors
feat_importances = pd.Series(RF_model.feature_importances_, index=train.columns)
feat_importances.nlargest(15).plot(kind='barh').figure.savefig('/home/joao_ines/Documents/SLE/feature_importance/RF_feature_importance.png', bbox_inches='tight')

#prediction of test set
y_pred = RF_model.predict(teste.values)

#Test set stats
path6='/home/joao_ines/Documents/SLE/stats/test_stats.csv'
path7='/home/joao_ines/Documents/SLE/new_modeling/test_predictions.csv'

test_stats = classification_report(y_test, y_pred, target_names=['Soluble', 'Moderate Soluble', 'Insoluble'], output_dict=True)
print(classification_report(y_test, y_pred, target_names=['Soluble', 'Moderate Soluble', 'Insoluble']))
pd.DataFrame(test_stats).transpose().to_csv(path6)
pd.DataFrame(y_pred).to_csv(path7)

#external test set predictions
path8='/home/joao_ines/Documents/SLE/stats/ext_test_stats.csv'
path9='/home/joao_ines/Documents/SLE/new_ext_ts/ex_test_predictions.csv'

ex_y_pred = RF_model.predict(x_ext_test.values)
#external test set stats
ext_test_stats = classification_report(y_ext_test['class'].values, ex_y_pred, target_names=['Soluble', 'Moderate Soluble', 'Insoluble'], output_dict=True)
print(classification_report(y_ext_test['class'].values, ex_y_pred, target_names=['Soluble', 'Moderate Soluble', 'Insoluble']))
pd.DataFrame(ext_test_stats).transpose().to_csv(path8)

pd.DataFrame(ex_y_pred).to_csv(path9)
