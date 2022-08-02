import numpy as np
import matplotlib.pyplot as plt

dose = 0
x = []
with open('analyse/Real0.txt'.format(dose)) as f:
    for line in f:
        if line.startswith('Ratio'):
            x.append(1.0/float(line.strip().split(' ')[1]))

n, bins, patches = plt.hist(x, 50, density=True)
plt.title(str(dose))
plt.show()