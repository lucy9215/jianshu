<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


# Machine Learning Summary 


[TOC]


总结一下Coursera的两个Machine Learning课程，方便以后回来看看。 

- Machine Learning Specialization - Washington University. 
  这个specialization讲的挺细，做了不少可视化，遗憾的是最后两门取消了。  
- Machine Learning - Stanford University
  毕竟是Andrew Ng的课, 两门课有重复的地方也有不一样的地方，有时间都看了也还是不错的。  


# Regression 

Regrassion predict连续的输出。 

## Linear Regression

叫做linear regression, 但是feature可以是非线性的，output y是feature们的线性组合。 

### Model representation 

> \\(y_{m\*1} = X_{m\*n} \* \theta_{n\*1} \\)

In matlab code:

```matlab
y = X*theta
```

- feature X, Size m*n, m is the number of data points, n is the number of feature.  
- label y, Size m*1, label of m data points. 
- weights theta, Size n*1, 

Regression就是在已知数据集feature矩阵(X)和数据label(y)的情况下估计feature的weights(theta). 即已知X,y,求解最fit数据集的theta。有了theta, 就能对数据集中的training example，如X(i,:), 做出估计的label值y, 估计的y也常表示为\\( \hat{y}\\). 

### Objective/Cost function 

模型建立好了，求解theta则是个优化问题。定义Objective/Cost function(目标函数), 求使Cost function值最小的theta. 

``` matlab 
predictions = X*theta;
J = (1/2)*mean((y-predictions).^2);
```

### Solve the Equation 

Linear regression 的objective function是个凸函数, 不存在除global minimum以外的极值。

#### Normal Equation 

```
theta = (X'*X)^-1*X'*y
```

- No need to choose a(learning rate)
- No need iterate
- Need to compute `(X'*X)^-1`
- Slow if n is very large 

适合在数据集不大的n<=10000的情况下使用。

**if `(X'*X)^-1` is non-invertible(singular/degenerate), it is possible that:**

- Redundant features(linearly dependent)
- Too many features (m < n)  

在这种情况下Octave/Matlab可以用pinv函数，基本上是ok的。

#### Gradient Descent 

```
gradient =  (1/m)*(X'*(X*theta-y));
theta = theta - alpha*gradient;
```

Compute the gradient(导数) of cost function, tune learning rate(alpha) to find an optimal theta.

#### Note 

- **Batch gradient descent:** each step of gradient descent uses all the training examples.
- **Mini batch gradient descent:** mini batch size for one step(good for parallel computing implementation).
- **Stochastic gradient descent:** one random training example for one step.  

#### More advanced optimization 

事实上很多优化的算法已经有现成的函数了, 那么given theta, we can compute:

- Cost: J_theta (a real number)
- Gradient of J_theta by theta: gradient (a n*1 vector)

Optimization algorithms:

- Gradient descent
- Conjugate gradient
- BFGS
- L-BFGS

Advantages:

- No need to manually pick learning rate(alpha).
- Often faster than gradient descent.

Disadvantages:  

- more complex

In matlab, write a function to compute cost and gradient:
```matlab
function [jVal, gradient] = costFunction(theta)
jVal = ...
gradient = ...
```

call a optimization function to get optimal theta:
```matlab
options =  optimset(‘GradObj’, ‘on’, ‘MaxIter’, ‘100’);
initialTheta = zeros(n, 1);
[optTheta, cost, exitFlag] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
```

#### Dealing with over fitting

##### Ridge regression

##### Lasso Regression


##### Regularized linear regression and logistic regression
(在Ng的课中只用到了l2-norm)







# Classification 

## Logistic regression 

将linear regression的output(y)通过sigmoid映射到(0, 1), 假设这是预测为1的概率(probability), 设上一个概率threshold(decision boundary)就能根据输出进行分类了。将该输出看作概率，那么logistic regression的Quality metric就是以Maximizing likelihood(probability of data)为目标来定义的了。即找出theta,使data(X)预测出label(y)的概率最大化。于是logistic regression model就能用maximum likelihood estimation(MLE)的方法来训练.  

### Model representation

假设threshold为0.5, 在输出概率p>0.5时预测1, 否则预测0, matlab代码如下:

```
probabilities = sigmoid(X*theta);
p(find(probabilities>=0.5)) = 1;
p(find(probabilities<0.5)) = 0;
```

非线性的decision boundary与linear regression一样，都是通过非线性feature的线性组合来形成的。

### Objective/Cost function 

Maximum likelihood estimation(MLE)最后为了计算方便(化乘法为加法)等价为了Maximun log likelihood:

```
J = mean((-y.*log(predictions)-(1-y).*log(1-predictions))) + lambda/(2*m)*sum(theta(2:n).^2);
gradient = (1/m)*X'*(predictions-y) + [0; lambda*theta(2:n)/m];
```

### Solve the optimization problem

With `J` and `gradient` you can solve this optimization problem using advanced optimization function mentioned in linear regression.

### Muti class classification(one-vs-all) 

需要多个分类，可以为每一个分类训练一个model, 故称为one-vs-all.

## Neural Networks - Non linear hypotheses

### Model representation 

Parameters:

- Input layer: \\(x_{m\*1}\\), a vector contains m features.
- Hidden layer: \\(a^{(j)}\_{s\_j\*1}\\), layer j has \\(s_j\\) features. And it has a intercept feature \\(a^{(j)}\_{0}=1\\). 因此其实有\\(s_j+1\\)个node, 也需要\\(s_j+1\\)个对应的weights.
- \\(\Theta^{(j)}\_{s\_{j+1}\*(s_j+1)}\\) 每layer都有一个weights matrix(j), 将\\(a^{(j)}\\)映射到\\(a^{(j+1)}\\). Size：\\(s\_{j+1}\*(s_j+1)\\), 由于每层input包含了intercept: \\(a^{(j)}\_0\\), 故一共是\\(s_j+1\\)维。输出到\\(a^{(j+1)}\\) 则有\\(s\_{j+1}\\)维。

Model: forward propogation(3 layer example, 1 input, 1 hidden, 1 output)

```matlab
% Add ones(intercept term) to the X data matrix
A1 = [ones(m, 1) X]; % input layer
Z2 = Theta1*A1'; 
A2 = [ones(m,1) sigmoid(Z2)']; % hidden layer
Z3 = Theta2*A2';
A3 = sigmoid(Z3)'; % output layer
```

### Objective/Cost function

#### Back propagation 

与logistic regression一致, cost function是maximum likelihood estimation, 由于是multi-class, 故还需要将每个class的cost都加起来。

```matlab
y_one_hot = ind2vec(y')';

J = mean(sum((-y_one_hot.*log(A3)-(1-y_one_hot).*log(1-A3)),2))...
    + lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
```

#### Gradient

```matlab
delta3 = A3 - y_one_hot;
%delta2 = delta3*Theta2.*A2.*(1-A2);
delta2 = delta3*Theta2(:,2:end).*sigmoidGradient(Z2'); %A2.*(1-A2);

Delta2 = delta3'*A2;
%Delta1 = delta2(:,2:end)'*A1;
Delta1 = delta2'*A1; %delta2(:,2:end)'*A1;

Theta2_grad(:, 2:end) = (1/m)*(Delta2(:, 2:end) + lambda*Theta2(:, 2:end));
Theta2_grad(:, 1) = (1/m)*Delta2(:, 1);
Theta1_grad(:, 2:end) = (1/m)*(Delta1(:, 2:end) + lambda*Theta1(:, 2:end));
Theta1_grad(:, 1) = (1/m)*Delta1(:, 1);
```










# Feature Normalization 

FEATURENORMALIZE(X) returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1. 
This is often a good preprocessing step to do when working with learning algorithms.

```matlab 
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

[m,n] = size(X);

mu = mean(X,1);
sigma = std(X,0,1);
for j = 1:n
    X_norm(:,j) = (X(:,j)-mu(j))/sigma(j);
end

```
