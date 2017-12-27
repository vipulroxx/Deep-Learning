import numpy as np
from sklearn import preprocessing

labels = np.array([1,2,5,3,2,1,4,2,1,3])
lb = preprocessing.LabelBinarizer()
lb.fit(labels)
print(lb.transform(labels))
