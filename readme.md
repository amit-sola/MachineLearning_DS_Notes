# Adding some important concepts/questions/notes helpful in interview preparation.

This site extracts most of the elements from [EliteDataScience](https://elitedatascience.com/machine-learning-interview-questions-answers) and [SpringBoard](https://www.springboard.com/blog/machine-learning-interview-questions/).

## I have divided this post into multiple sections, details of which I will be adding gradually.


### ML interview questions on  

1. [Algorithms and Theory](#1-algorithms-and-theory)
2. [ML Models](#1-ml-models)

## 1. Algorithms and Theory

**Q1- What’s the trade-off between bias and variance?**

Bias is an error due to generalization (under fitting), variance is the error due to too much complexity in the learning algorithm, i.e. over fitting.

Train set error  in row 1 and Dev set error in row 2
| High variance        | High bias           | High Bias and High Variance  | Low bias and Low variance  |
| :---: |:---:|:---:|:---:|
| 1%     |15% | 15% |0.5% |
| 11%     | 16%    |   30% |1% |

 

Ways to get rid of high bias in the neural network (Fit at least training set well): 
1) Increase number of neurons.
2) Make Bigger network.
3) Training Longer.
4) Different Architecture.

After you get good performance on training set, you can proceed to improve performance on dev set, by improving variance.
Ways to get improve variance in the neural network (Generalize well over the dev set): 
1) More data.
2) Better regularization.
3) Different architecture.

**Q2 How is KNN different from k-means clustering?**

##### KNN
KNN stands for K nearest neighbors. It is an supervised learning algorithm. The test data is labeled based on the class of K nearest neighbor of test data.

##### K Means
K means is an unsupervised clustering algorithm. Complete dataset is divided into K data clusters depending on mean distance between data points in a cluster.

**Q3- Explain how a ROC curve works.**

ROC curves stands for Receiver operating characteristic curve. Its a plot of *True positive rate as a function of False Positive rate*. An ROC curve is the most commonly used way to visualize the performance of a binary classifier, and AUC is (arguably) the best way to summarize its performance in a single number.
For multiple class outputs, we use one vs all strategy.

ROC curve and AUC is best explained on [Data School](http://www.dataschool.io/roc-curves-and-auc-explained/)

AUC, which stands for Area Under the Curve. AUC is literally just the percentage of area under curve to the total area. A good 	classifier with an AUC of around 0.8, a very poor classifier has an AUC of around 0.5, and best classifier has an AUC of close to 1.

 All the AUC metric cares about is how well your classifier separated the two classes, and thus it is said to only be sensitive to rank ordering. You can think of AUC as representing the probability that a classifier will rank a randomly chosen positive observation higher than a randomly chosen negative observation, and thus it is a useful metric even for datasets with highly unbalanced classes.

Finally one has to decide whether one would rather minimize the False Positive Rate or maximize the True Positive Rate.

**Q5- Define precision and recall.**

Confusion matrix can clear doubt about precision and recall.

Recall=TP/(TP+FN). Out of total positive out there, how many was our model able to truly classify as positive. Out of all total positive present how many was it able to recall.

Precision=TP/(TP+FP). Out of positive predicted, how many are actually positive. How precisely the model identified.


![Confusion Matrix](https://image.slidesharecdn.com/petutorial-150413084118-conversion-gate01/95/performance-evaluation-for-classifiers-tutorial-16-638.jpg?cb=1428914518)

**Q6. What is Bayes Theorem**

Its nothing but answering the question : If my test were positive, whats the probability to trust the test, i.e. whats the true positive rate.

P(A/X)=TP/(TP+FP)


![Bayes Theorem](https://i.imgur.com/yJbZWGA.png)

**Q7. What is Naive Bayes Classifier?**

A better explanation is given in [Quora](https://www.quora.com/Why-is-naive-Bayes-naive?share=1)


**Q8- What’s the difference between Type I and Type II error?**
Type I error are nothing but False Positives.
Type II error are nothing but False Negatives

**Q8. What’s the difference between a generative and discriminative model?**

Generative classifiers learn a model of the joint probability, p( x, y), of the inputs x
and the label y, and make their predictions by using Bayes rules to calculate p(ylx),
and then picking the most likely label y. Discriminative classifiers model the posterior
p(ylx) directly, or learn a direct map from inputs x to the class labels


**Q9 What cross-validation technique would you use on a time series dataset?**
A better explanation is given in [StackOverflow](https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection)
Instead of using standard k-folds cross-validation, you have to pay attention to the fact that a time series is not randomly distributed data — it is inherently ordered by chronological order. If a pattern emerges in later time periods for example, your model may still pick up on it even if that effect doesn’t hold in earlier years!

You’ll want to do something like forward chaining where you’ll be able to model on past data then look at forward-facing data.

fold 1 : training [1], test [2]
fold 2 : training [1 2], test [3]
fold 3 : training [1 2 3], test [4]
fold 4 : training [1 2 3 4], test [5]
fold 5 : training [1 2 3 4 5], test [6]

**Q10- How is a decision tree pruned?**

Pruning is a technique in machine learning that reduces the size of decision trees by removing sections of the tree that provide little power to classify instances. Pruning reduces the complexity of the final classifier, and hence improves predictive accuracy by the reduction of overfitting.

It is hard to tell when a tree algorithm should stop because it is impossible to tell if the addition of a single extra node will dramatically decrease error. This problem is known as the horizon effect. 


>*Pruning can occur in a top down or bottom up fashion.*

A top down pruning will traverse nodes and trim subtrees starting at the root, while a bottom up pruning will start at the leaf nodes. Below are several popular pruning algorithms.

Reduced error pruning is perhaps the simplest version: replace each node. If it doesn’t decrease predictive accuracy, keep it pruned.

**Q11- Which is more important to you– model accuracy, or model performance?**
Depends on the usecase.
For cancer detection you would want to be more accurate in pointing out the cancer patient.
For fraud detection where minority of cases would be fraud, performance is the major factor, since a model prediction all cases as not fraud will have better accuracy while bad performance.

**Q12- What is F1 Score ? How would you use it ?**
The F1 score is a measure of a model’s performance. It is a weighted average of the precision and recall of a model, with results tending to 1 being the best, and those tending to 0 being the worst. You would use it in classification tests where true negatives don’t matter much.
![F1](https://i.imgur.com/vxTaObT.png)

 The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
 Its mostly used in IR.
 The G-measure is the geometric mean.
 
 **Q13- How would you handle an imbalanced dataset?**
Look at [Machine Learning Masetry.](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
Tips
* Collect More Data
* Try Changing Your Performance Metric
    *   Confusion Matrix: A breakdown of predictions into a table showing correct predictions (the       diagonal) and the types of incorrect predictions made (what classes incorrect predictions        were assigned).
    *   Precision: A measure of a classifiers exactness.
    * Recall: A measure of a classifiers completeness
    *   F1 Score (or F-score): A weighted average of precision and recall.
    *   ROC Curves: Like precision and recall, accuracy is divided into sensitivity and specificity and models can be chosen based on the balance thresholds of these values.
*   Try Resampling Your Dataset
*   Try Generate Synthetic Samples.

**Q14 -When should you use classification over regression?**

Regression: the output variable takes continuous values.

Classification: the output variable takes class labels.

**Q15 -Explain Ensemble Bagging Boosting?**
Ensembles are a divide-and-conquer approach used to improve performance. The main principle behind ensemble methods is that a group of “weak learners” can come together to form a “strong learner”

The random forest takes this notion to the next level by combining trees with the notion of an ensemble. Thus, in ensemble terms, the trees are weak learners and the random forest is a strong learner.

To use Bagging or Boosting you must select a base learner algorithm. For example, if we choose a classification tree, Bagging and Boosting would consist of a pool of trees as big as we want. 
.
In the case of Bagging, any element has the same probability to appear in a new data set. However, for Boosting the observations are weighted and therefore some of them will take part in the new sets more often: 

![Bagging Vs Boosting](https://quantdare.com/wp-content/uploads/2016/04/bb3-1150x441.png)
![VS](https://quantdare.com/wp-content/uploads/2016/04/bb2-1150x441.png)
![](https://quantdare.com/wp-content/uploads/2016/04/bb4-1150x441.png)
![](https://quantdare.com/wp-content/uploads/2016/04/bb5-1150x410.png)
In Boosting algorithms each classifier is trained on data, taking into account the previous classifiers’ success. After each training step, the weights are redistributed. **Misclassified data increases its weights to emphasise the most difficult cases. In this way, subsequent learners will focus on them during their training.**

**How does the classification stage work?**
To predict the class of new data we only need to apply the N learners to the new observations.* In **Bagging** the result is obtained by averaging the responses of the N learners (or majority vote). However, **Boosting** assigns a second set of weights, this time for the N classifiers, in order to take a weighted average of their estimates.

**Which is the best, Bagging or Boosting?**

There’s not an outright winner; it depends on the data, the simulation and the circumstances.
Bagging and Boosting decrease the variance of your single estimate as they combine several estimates from different models. So the result may be a model with higher stability.

If the problem is that the single model gets a very low performance, Bagging will rarely get a better bias. However, Boosting could generate a combined model with lower errors as it optimises the advantages and reduces pitfalls of the single model.

By contrast, if the difficulty of the single model is over-fitting, then Bagging is the best option. Boosting for its part doesn’t help to avoid over-fitting; in fact, this technique is faced with this problem itself. For this reason, Bagging is effective more often than Boosting.


| Similarities       | Differences   |
| :---: |:---:|
| Both are ensemble methods to get N learners from 1 learner    |but, while they are built independently for Bagging, Boosting tries to add new models that do well where previous models fail |
| Both generate several training data sets by random sampling    | but only Boosting determines weights for the data to tip the scales in favor of the most difficult cases    | 
|Both make the final decision by averaging  the N learners (or taking the majority of them)|but it is an equally weighted average for Bagging and a weighted average for Boosting, more weight to those with better performance on training data.|
|Both are good at reducing variance and provide higher stability| but only Boosting tries to reduce bias. On the other hand, Bagging may solve the over-fitting problem, while Boosting can increase it |

**Q16 -How do you ensure you’re not overfitting with a model??**

*   Keep Model Simpler
*   Use Kfold cross validation
*   Use better Regularization
*   Use less features
*   Chose a different Model. Ensemble ?
![](https://qph.ec.quoracdn.net/main-qimg-9000c0e50e1a97d0d12e85dc93affa5f.webp)

**Q17 -What evaluation approaches would you work to gauge the effectiveness of a machine learning model?**
More reading: [How to Evaluate Machine Learning Algorithms (Machine Learning Mastery)](https://machinelearningmastery.com/how-to-evaluate-machine-learning-algorithms/)

*   Test and Train Datasets
*   Cross Validation
    *   For example, a 3-fold cross validation would involve training and testing a model 3 times
1: Train on folds 1+2, test on fold 3
2: Train on folds 1+3, test on fold 2
3: Train on folds 2+3, test on fold 1


**Q18- List All ML algorithms  Grouped by Learning Style .**
More read at [Machine Learning Mastery](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)
![MLALGO](https://s3.amazonaws.com/MLMastery/MachineLearningAlgorithms.png?__s=9ns6a4sepp6hkbcfd1xv)

Let’s take a look at three different learning styles in machine learning algorithms:
*   1. Supervised Learning
**Example problems are classification and regression.**
Example algorithms include Logistic Regression and the Back Propagation Neural Network.
*   2. Unsupervised Learning
A model is prepared by deducing structures present in the input data.
**Example problems are clustering, dimensionality reduction and association rule learning.**
Example algorithms include: the Apriori algorithm,PCA and k-Means.

*   3. Semi-Supervised Learning
Input data is a mixture of labeled and unlabelled examples.
There is a desired prediction problem but the model must learn the structures to organize the data as well as make predictions.
Example problems are classification and regression.
**Example algorithms are extensions to other flexible methods that make assumptions about how to model the unlabeled data.**

**Q19- What is a Regression Algorithm.**
Regression is concerned with modeling the relationship between variables that is iteratively refined using a measure of error in the predictions made by the model.
![Regression](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2013/11/Regression-Algorithms.png)

The most popular regression algorithms are:
*   Ordinary Least Squares Regression (OLSR)
*   Linear Regression
*   Logistic Regression
*   Ridge Regression
*   Stepwise Regression
*   Multivariate Adaptive Regression Splines (MARS)
*   Locally Estimated Scatterplot Smoothing (LOESS)

**Q20- What is your understanding of Probablistic Models**

*Frequentist and Bayesian Interpretations*
***
Frequentists interpret the **probability in terms of the outcome over multiple
experiments in which the occurence of the event is monitored**. So for the
coin example, the frequentist interpretation of the 0.5 probability is that if
we toss a coin many times, almost half of the times we will observe a head. A
drawback of this interpretation is the reliance on multiple experiments. But
consider a different scenario. If someone claims that the probability of the
polar ice cap melting by year 2020 is 10%, then the frequentist interpretation
breaks down, because this event can only happen zero or one times. Unless,
there are multiple parallel universes!
![Basic](https://i.imgur.com/zAAtWAl.png)
![Bayes](https://i.imgur.com/BcOZLRB.png)
