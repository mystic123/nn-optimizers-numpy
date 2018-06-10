Simple examples of optimizers implementation in numpy.

I based my implementation on examples from Deep Learning Specialization [(Coursera)](https://www.coursera.org/specializations/deep-learning) and [this](http://neuralnetworksanddeeplearning.com) tutorial.

Great overview about optimization algorithms is [here](http://ruder.io/optimizing-gradient-descent/index.html#tensorflow).

Results:

From this simple example, we can conclude:
1. Not-adaptive algorithms (SGD, Momentum, NAG) need high learning rate for this task. With small learning rates, progress is slow.
2. Adaptive algorithms (Adam, Adagrad, RMSProp) fail (diverge) with high learning rates
3. Best results are achieved by Adam, RMSProp and Adagrad (depending on lr)

![Results1](https://github.com/mystic123/DeepLearning/blob/master/Basics/plots/plot_0.001.png)
![Results2](https://github.com/mystic123/DeepLearning/blob/master/Basics/plots/plot_0.01.png)
![Results3](https://github.com/mystic123/DeepLearning/blob/master/Basics/plots/plot_0.1.png)
![Results4](https://github.com/mystic123/DeepLearning/blob/master/Basics/plots/plot_1.0.png)
![Results5](https://github.com/mystic123/DeepLearning/blob/master/Basics/plots/plot_10.0.png)
