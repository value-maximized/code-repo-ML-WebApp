import matplotlib.pyplot as plt
import numpy as np

y = np.array([35 , 25, 25, 15])
myLabels = ["A", "B","C","D"]
myExplode = [0.5,0,0,0]

plt.pie(y, labels= myLabels,explode=myExplode,shadow=True)
plt.show()