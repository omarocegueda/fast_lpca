import numpy as np
import fast_lpca as flpca
from time import time
reload(flpca)

def slow_lpca(I, radius, out):
    n0 = I.shape[0]
    n1 = I.shape[1]
    n2 = I.shape[2]
    ndiff = I.shape[3]

    side = 1 + (2 * radius)
    nsamples = side * side * side
    out[...] = 0
    for i in range(radius, n0 - radius):
        for j in range(radius, n1 - radius):
            for k in range(radius, n2 - radius):
                X = np.zeros((nsamples, ndiff))
                M = np.zeros(ndiff)

                temp = I[i - radius: i + radius + 1,
                         j - radius: j + radius + 1,
                         k - radius: k + radius + 1, :]
                X = temp.reshape(nsamples, I.shape[3])
                # compute the mean and normalize
                M = np.mean(X, axis=0)
                X = X - np.array([M, ] * X.shape[0], dtype=np.float64)

                # Compute the covariance matrix C = X_transpose X
                C = np.transpose(X).dot(X)
                C = C / nsamples
                out[i, j, k, :, :] = C[:, :]


ndiff = 32
vside = 50
radius = 2

n = np.array([vside, vside, vside, ndiff], dtype=np.int32)
I = np.empty(tuple(n), dtype=np.float64)
I = np.random.random(I.size).reshape(tuple(n))


# First test accuracy of the sequential algorithm
out_slow = np.zeros(tuple(n)+(ndiff,), dtype=np.float64)
out_fast = np.zeros(tuple(n)+(ndiff,), dtype=np.float64)
flpca.fast_lpca(I, radius, out_fast)
slow_lpca(I, radius, out_slow)

dd = np.abs(out_slow - out_fast)
print("Max. difference: %e"%(dd.max(),))


# Now test performance

for radius in range(5):
    start = time()
    flpca.fast_lpca(I, radius, out_fast)
    end = time()
    elapsed = end - start
    print("Fast (radius = %d):%f"%(radius, elapsed))

for radius in range(5):
    start = time()
    slow_lpca(I, radius, out_slow)
    end = time()
    elapsed = end - start
    print("Slow (radius = %d):%f"%(radius, elapsed))


