import numpy as np
import math
import matplotlib.pyplot as plt
import ipdb as pdb


data = np.genfromtxt("convergence.txt", delimiter=',', dtype="f4")
train_loss = data[21:133,1]
valid_loss = data[21:133,2]
recall = data[21:133,3]
epoch = data[21:133,0]
plt.rcParams.update({'font.size': 22})
plt.plot(epoch, train_loss, 'r', linewidth=4, label='Train_loss')
plt.plot(epoch, valid_loss, 'b', linewidth=4, label='Valid_loss')
plt.plot(epoch, recall, 'g', linewidth=4, label='mAP')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# pdb.set_trace()

