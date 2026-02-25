import numpy as np
import matplotlib.pyplot as plt

#Original tutorial just took a sequence of data (0's and 1's typed by monkey, drift diffusion trajectory) and learns a linear relationship between 
#past values and the next value, then tests how well that relationship predicts new data
#It is AR(1) OR AR(r)

#Mine does the same but with dice rolls that are dependent (depend on the previous roll with a 70% chance of repeating the previous one). Clips to 1-6
#It is AR(r)

# Creates drift diffusion matrix it's for original tutorial's data. 
def ddm(T, x0, xinfty, lam, sig):
  t = np.arange(0, T, 1.)
  x = np.zeros_like(t)
  x[0] = x0

  for k in range(len(t)-1):
      x[k+1] = xinfty + lam * (x[k] - xinfty) + sig * np.random.standard_normal(size=1)

  return t, x

#Build the time delay matrices, since linear regression expects 2D matrix of predictors and vector of targets
def build_time_delay_matrices(x, r):
    x1 = np.ones(len(x)-r)
    x1 = np.vstack((x1, x[0:-r]))
    xprime = x
    for i in range(r-1):
        xprime = np.roll(xprime, -1)
        x1 = np.vstack((x1, xprime[0:-r]))

    x2 = x[r:]

    return x1, x2

#Modified for dice values, clip 1-6
#np.dot - linear regression prediction
#np.round - round to the nearest integer (since it's dice)
#np.clip - valid is in valid range
def AR_prediction(x_test, p):
    x1, x2 = build_time_delay_matrices(x_test, len(p)-1)

    preds = np.dot(x1.T, p)

    preds = np.round(preds)
    preds = np.clip(preds, 1, 6)

    return preds

#Compute the error rate 
def error_rate(x_test, p):
    x1, x2 = build_time_delay_matrices(x_test, len(p)-1)

    preds = AR_prediction(x_test, p)

    return np.count_nonzero(x2 - preds) / len(x2)

#Converts the series into past/future pairs (X1 = past, X2 = future)
#Fit linear regression from past to future using least squares
#Return X1, x2 and p (model coeff.)
def AR_model(x, r):
    x1, x2 = build_time_delay_matrices(x, r)

    p, res, rnk, s = np.linalg.lstsq(x1.T, x2, rcond=None)

    return x1, x2, p

# Generates dependent dice rolls
# Generate n dice rolls with 70% probability that the next roll repeats the previous
# x = array of dice rolls. Generate x[i+1] using x[i]
np.random.seed(0)

def generate_dependent_dice(n, repeat_prob=0.7):
    x = np.zeros(n)
    x[0] = np.random.randint(1,7)
    for i in range(n-1):
        if np.random.rand() < repeat_prob:
            x[i+1] = x[i] 
        else:
            x[i+1] = np.random.randint(1,7)
    return x

x = generate_dependent_dice(500, repeat_prob=0.7)
test = generate_dependent_dice(200, repeat_prob=0.7)

#Plots the generated dice data
plt.figure()
plt.step(x, '.-')
plt.title("Dice Training Data")
plt.xlabel("time")
plt.ylabel("roll")
plt.show()

# r = array of orders of AR models to use (1-20)
#err = array of same shape as r, to be filled with test error values
r = np.arange(1, 21)
err = np.ones_like(r, dtype=float)

#Fit the ar model for each order
for i, order in enumerate(r):
  x1, x2, p = AR_model(x, order)
  test_error = error_rate(test, p)
  err[i] = test_error

#Plots the error rate
plt.figure()
plt.plot(r, err, 'o-')
plt.xlabel("Model order (r)")
plt.ylabel("Error rate")
plt.title("Autoregression on Dice Rolls")
plt.show()


for rr, e in zip(r, err):
    print(f"r = {rr}, error = {e:.3f}")
