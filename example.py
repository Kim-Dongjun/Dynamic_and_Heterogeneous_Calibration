print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from hmmlearn import hmm

def normalize(vector):
    try:
        return vector/np.sum(vector,1).reshape(len(vector),1)
    except:
        return vector/np.sum(vector)

components = 8
#startprob = np.array([0.6, 0.3, 0.1, 0.0])
startprob = normalize(np.random.random(components))
# The transition matrix, note that there are no transitions possible
# between component 1 and 3
#transmat = np.array([[0.7, 0.2, 0.0, 0.1],
#                     [0.3, 0.5, 0.2, 0.0],
#                     [0.0, 0.3, 0.5, 0.2],
#                     [0.2, 0.0, 0.2, 0.6]])
transmat = normalize(np.random.random((components,components)))
# The means of each component
#means = np.array([[0.0,  0.0],
#                  [0.0, 11.0],
#                  [9.0, 10.0],
#                  [11.0, -1.0]])
#means = np.random.random((components,2))
means = np.array([[10.0,0.0],[10.0,10.0],[0.0,10.0],
                  [-10.0,10.0],[-10.0,0.0],[-10.0,-10.0],[0.0,-10.0],[10.0,-10.0]])
# The covariance of each component
#covars = .5 * np.tile(np.identity(2), (4, 1, 1))
covars = .5 * np.tile(np.identity(2), (components, 1, 1))

# Build an HMM instance and set parameters
model = hmm.GaussianHMM(n_components=components, covariance_type="full")

# Instead of fitting it from the data, we directly set the estimated
# parameters, the means and covariance of the components
model.startprob_ = startprob
model.transmat_ = transmat
model.means_ = means
model.covars_ = covars

# Generate samples
X, Z = model.sample(500)
model.fit(X)
Z2 = model.predict(X)

# Plot the sampled data
plt.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
         mfc="orange", alpha=0.7)

for i in range(len(X)):
    plt.text(X[i][0], X[i][1], str(Z2[i]))

# Indicate the component numbers
'''for i, m in enumerate(means):
    plt.text(m[0], m[1], 'Component %i' % (i + 1),
             size=17, horizontalalignment='center',
             bbox=dict(alpha=.7, facecolor='w'))'''
plt.legend(loc='best')
plt.show()


def useHMM(self, numTimeStep, differences, trueResult, dir, l, itrCalibration, numberHMMClusters, numOutputDim):
    model = hmm.GaussianHMM(n_components=numberHMMClusters, covariance_type="full")

    model.fit(differences)
    predictedRegime = model.predict(differences)
    print(predictedRegime)
    sys.exit()
    pass