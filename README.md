# mlScikitTensorflow
## Requirement
* dependencies
    * jupyter==1.0.0
    * matplotlib==2.0.2
    * numexpr==2.6.3
    * numpy==1.13.1
    * pandas==0.20.3
    * Pillow==4.2.1
    * protobuf==3.4.0
    * psutil==5.3.1
    * scikit-learn==0.19.0
    * scipy==0.19.1
    * sympy==1.1.1
    * tensorflow==1.3.0

## Official Book and Guidance
* official book:

[Hands-On Machine Learning with Scikit-Learn & TensorFlow](http://shop.oreilly.com/product/0636920052289.do)

* official guidance in jupyter notebook:

[link](https://github.com/ageron/handson-ml)

# content
## Part I,  e Fundamentals of Machine Learning
Linear and Polynomial Regression, Logistic Regression, k-Nearest Neighbors, Support Vector Machines, Decision Trees, Random Forests, and Ensemble methods.

## Part II, Neural Networks and Deep Learning
feedforward neural nets, convolutional nets, recurrent nets, long short-term memory (LSTM) nets, and autoencoders.

# Other Materials
 * Joel Grus, *Data Science from Scratch* (O’Reilly). This book presents the fundamentals of Machine Learning, and implements some of the main algorithms in pure Python (from scratch, as the name suggests).
 * Stephen Marsland, *Machine Learning: An Algorithmic Perspective* (Chapman and Hall). This book is a great introduction to Machine Learning, covering a wide range of topics in depth, with code examples in Python (also from scratch, but using NumPy).
 * Stuart Russell and Peter Norvig, *Artificial Intelligence: A Modern Approach, 3rd Edition* (Pearson). This is a great (and huge) book covering an incredible amount of topics, including Machine Learning. It helps put ML into perspective.

# Challenge
[Kaggle](https://www.kaggle.com)

# Chapter 01
## Algorithms List
* Supervised learning
    * k-nearest neighbors
    * linear regression
    * logistic regression
    * support vector machines
    * decision trees and random forests
    * neural networks
* Unsupervised learning
    * Clustering
        * k-means
        * Hierarchical cluster analysis
        * expectation maximization
    * Visualization and dimensionality reduction
        * principal component analysis
        * kernel PCA
        * locally-linear embedding
        * t-distributed Strochastic neighbor embedding
    * Association rule learning
        * Apriori
        * Eclat
* Semisupervised learning
    * deep belif networks
* Reinforcement learning

## Main Challenges of ML
* insufficient quantity of training data
* nonrepresentative training data
* poor-quality data ( *pre-pocessing*
* irrelevant features ( *feature engineering*
* overfitting the training data ( *regularization - hyperparameter*
* underfitting the training data

## Testing and Validating
* 80% the training set and 20% the test set ( *generalization error*) 
* cross-validation
**no free lunch theorem**:
    if you make absolutely no assumption about the data, then there is no reason to prefer one model over any other.
    
# Chapter 02
## Scikit-Learn design
* consistency: all objects share a consistent and simple interface:
    * estimators - `.fit()`: estimate some parameters based on a dataset.
        * inspection `.<attributes>`: access the hyperparameters of the estimators
        
    * transformers - `.transform()`: transform a dataset
            **`fit_transform()`**
    * predictors - `.predict()`: make predictions given a dataset
        * `.score()`: measure the quality of the predictions given a test set
    
    * composition: existing building blocks are reused as much as possible
    * sensible defaults: SL provides reasonable default values for most parameters, making it easy to create a baseline working system quickly 

## Custom transformers
* create a class and implement three methods, add `Base:BaseEstimator` as a base class
    * `fit()`
    * `transform()`
    * `fit_transform()` - get it free by adding `Base:ßTransformerMixin` as a base class
## Transformation pipelines
* to help with sequences of transformations - `pipeline:Pipeline`
```
from sklearn.pipeline import Pipeline
num_pipeline = Pipeline([
        ('trans1', <transformer1>),
        ('trans2', <transformer2>)
        ...
    ])

new_data = num_pipeline.fit_transform(old_data)
```
* to gather the result from different attribute - `pipeline:FeatrureUnion`
```
from sklearn.pipeline import FeatureUnion

pipe1 = ...
pipe2 = ...

full pipeline = FeatureUnion(transformer_list=[
        ('pipe1',pipe1),
        ('pipe2',pipe2)
])
```

## Save model
* to save the trained model for late usage - `externals:joblib`:

```
from sklearn.externals import joblib

joblib.dump(my_model,'my_model.pkl')
my_model_loaded = joblib.load('my_model.pkl')
```

# Chapter 03 
## Implementing cross-validation
* more control over the cross-validation process `model_selection:StratifiedKFold`,`base:clone`
```
skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_idx, test_idx in skfolds.split(X_train, y_train):
    clone_model = clone(ML_model)
    X_train_folds = X_train[train_idx]
    y_train_folds = y_train[train_idx]
    X_test_fold = X_train[test_idx]
    y_test_fold = y_train[test_idx]
    
    clone_model.fit(X_train_folds, y_train_folds)
    y_pred = clone_model.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct/(len(y_pred))
```
