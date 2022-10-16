# Amphisbaena

![monster](https://upload.wikimedia.org/wikipedia/commons/d/dc/%D0%90%D0%BC%D1%84%D0%B8%D1%81%D0%B1%D0%B5%D0%BD%D0%B0._%D0%9C%D0%B8%D0%BD%D0%B8%D0%B0%D1%82%D1%8E%D1%80%D0%B0_%D0%B8%D0%B7_%D0%90%D0%B1%D0%B5%D1%80%D0%B4%D0%B8%D0%BD%D1%81%D0%BA%D0%BE%D0%B3%D0%BE_%D0%B1%D0%B5%D1%81%D1%82%D0%B8%D0%B0%D1%80%D0%B8%D1%8F.png)

> Imagine the digits in the test set of the MNIST dataset
> (http://yann.lecun.com/exdb/mnist/) got cut in half vertically and shuffled
> around. Implement a way to restore the original test set from the two halves,
> whilst maximising the overall matching accuracy.

This repository is the training and inference code for a deep learning approach to solve the above.

## Approach

An obvious method to solve this problem is to train a neural network contrastively. That is, a digit for which the left and right halves are from the same image has a label of `1` and a label of `0` otherwise.

Then, for each possible pair of left and right halves, the model predicts a label. To restore the original digit, we need only solve the [assignment problem](https://en.wikipedia.org/wiki/Assignment_problem), which has an excellent [implementation in scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).

## Why "Amphisbaena"?

An amphisbaena is a mystical, two-headed snake. Unlike a siamese network, the image recognition arms are not
