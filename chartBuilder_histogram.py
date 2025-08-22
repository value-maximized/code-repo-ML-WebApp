import random
import numpy as np
import math
import matplotlib.pyplot as plt
print(random.randrange(1,5))
print(random.lognormvariate(1, 5))

mean = 0
stdDev = 1
size = 1000

#Normal Distribution of Randomly Generated Numbers
#randomNumbers = np.random.normal(loc=mean, scale=stdDev, size=size)
#Exponential Distribution of Randomly Generated Numbers
randomNumbers = np.random.exponential(scale=1, size=size)

print(randomNumbers)

plt.figure(figsize=(10,6))

plt.hist(randomNumbers, bins=30, density = True, alpha=0.6, color='g', edgecolor='black')

plt.title("Histogram of Normally Distributed Random Numbers")
plt.xlabel('Value')
plt.ylabel('Probability Density')

plt.show()