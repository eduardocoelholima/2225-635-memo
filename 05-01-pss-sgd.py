import numpy as np
import scipy as sp

D = np.array([[255, 128, 128, 0, 0], [55, 128, 128, 128, 1], [192, 128, 128, 0, 0],
    [100, 128, 128, 100, 1], [30, 64, 128, 30, 2], [20, 64, 128, 0, 2]])
X = D[:, :4]
y = D[:, -1]
unique_ys = np.unique(y)
X = np.hstack((np.ones((X.shape[0], 1)), X))
X = X/D.max()  # normalization
Y = np.zeros((y.shape[0], y.max()+1))
Y[np.arange(y.shape[0]), y] = 1  # one-hot encoding
w = np.zeros((X.shape[1], Y.shape[1]))

alpha = 0.1
epochs = 2

# initialize ground_truth as a binary matrix setting positions where a == j
ground_truth = np.zeros((X.shape[0], Y.shape[1]))
for label in unique_ys:
    ground_truth[:, label] = np.where(y == label, 1, 0)

for epoch in range(epochs):
    print(f'--- epoch {epoch} ---')
    logits = X @ w
    probs = sp.special.softmax(logits, axis=1)

    alpha_grads = np.where(ground_truth == 1, alpha*(1-probs), alpha*(-probs)) # set delta using slp's x-entropy loss
    deltas = np.zeros((X.shape[1], Y.shape[1]))

    for label in unique_ys:
        map = alpha_grads[:, label].reshape(1, -1) * X.T
        # deltas[:, label] = np.mean(map.T, axis=0) # reduce delta to the mean delta over the samples
        deltas[:, label] = np.sum(map.T, axis=0)
    w += deltas # update weights at the end of the epoch

    for label in unique_ys:
        scores = X[:, :] @ w[:, label]
        predictions = np.where(scores>0, 1, 0)
        diff = np.abs(Y[:, label] - predictions)
        accuracy = np.mean(diff)
        print(f'  label {label} accuracy:{accuracy}')

print('done')
