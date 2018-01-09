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

