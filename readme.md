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




