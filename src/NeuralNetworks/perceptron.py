import numpy as np
import matplotlib.pyplot as plt

## Generate some random data of two clases
mean = [0, 0]
covariance = [[10,0], [0,10]]
n_samples = 1000
features = np.random.multivariate_normal(mean, covariance, n_samples)
classes = np.random.choice([-1, 1], size=n_samples)

features[classes==1] += 5

plt.plot(features[classes==-1,0], features[classes==-1,1], '.', alpha=0.5)
plt.plot(features[classes==1,0], features[classes==1,1], '.', alpha=0.5)
plt.axis('equal')
plt.grid()
plt.show()

## Iterate
def perceptron(features: np.ndarray, classes: np.ndarray, iterations=100):
    # initialise parameters
    w_t = np.random.uniform(size=1 + features.shape[1]) # add one more fake feature
    n_data = features.shape[0]
    labels = np.zeros(n_data)
    for k in range(iterations):
        # calculate output
        n_errors = 0
        for t in range(n_data):
            y_t = classes[t]
            x_t = np.concatenate([[1], features[t]])
            a_t = np.sign(w_t.T @ x_t)
            labels[t] = a_t

            if a_t != y_t:
                w_t += y_t * x_t
                n_errors += 1
            
        print("error rate: ", n_errors / n_data)
    return w_t, labels


## call the algorithm
w, labels = perceptron(features, classes, 100)

## plot the results
plt.plot(features[labels==-1,0], features[labels==-1,1], 'r.', alpha=0.5)
plt.plot(features[labels==1,0], features[labels==1,1], 'g.', alpha=0.5)
X = np.linspace(-10,10)
# in our parametrisation, we have
# a = w[0] + x w[1] + y w[2] 
# so as a = 0 is the decision boundary, we can solve for the y coordinate
Y = - (w[0] + X * w[1])/w[2]
plt.plot(X, Y)
plt.axis([-10,10,-10,10])
plt.grid()
plt.show()
