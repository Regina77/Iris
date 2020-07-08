from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import csv


#keep only the ones that are within +3 to -3 standard deviations from the mean
def clean_data(df, z_thresh=3):

    #Drop the rows that have missing data
    df.dropna()

    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=[np.number]).apply(lambda x: np.abs(stats.zscore(x)) < z_thresh, result_type='reduce').all(axis=1)
    df.drop(df.index[~constrains], inplace=True)




# Study the statistical relationship between the features and between the features and the target
def find_relationship(df):

    # Compute the correlation matrix
    corr = df.corr().iloc[0:4,0:4]
    plt.figure() 
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap='viridis',mask=mask, annot=True, annot_kws={"size": 20, "color":"w"})
    plt.savefig("feature_heat_map.png")

    # Extract the values for features and create a list called all_features
    all_features=[]
    features = df.values[:,0:4]
    features.shape

    # Add all features for each row
    for observation in features:
        all_features.append(observation[0] + observation[1] + observation[2] + observation[3])

    df['all_features'] = all_features 
    df = df.round(2)

    # Finding the relationship between features and targets
    plt.figure()
    sns.swarmplot(x=df['target'], y=df['all_features']).set_title('Relationship between features and the target')
    plt.savefig("features_vs_target.png")

    # Save to csv
    df.to_csv("Relationship.csv", index=False)



def build_model(df):
    x = df.values[:,0:4]
    y = df.values[:,4]

    # Split-out dataset, 20% of data is used for testing
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=1)

    # Build 2 models: logistic regression and linear discriminant analysis
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))

    # Evaluate both models
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

        # Test the testing data and evaluate the results using the appropriate metrics
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)

    # Save the result to csv
    results_df = pd.DataFrame({"Accuracy(LR)" : results[0], "Accuracy(LDA)": results[1]}).round(2)
    results_df.to_csv("Algorithm_Comparison.csv", index=False)

    # Visualize the results
    plt.figure()
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.savefig("Algorithm_Comparison.png")


if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['target'])
 
    # Describe the data statistically using panda
    iris_description = df.describe()

    clean_data(df)
    
    find_relationship(df)

    build_model(df)
