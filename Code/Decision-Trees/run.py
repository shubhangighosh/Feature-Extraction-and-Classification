import numpy as np
import matplotlib.pyplot as plt

minNumObj = np.array([2,50,100,150,200,300,500,700,1000,1500,1650,1700, 1750,1850,1950,2000,2500])
acc = np.array([100,99.3772,98.8434,99.3772,99.3772,99.3772,99.3772,99.3772,99.3772,97.242,97.242,100,100,100,61.21,61.21,41.2811])
n_leaves = np.array([25,17,23,17,17,17,17,17,9,2,2,8,8,8,2,2,1])
t_size = np.array([30,19,26,19,19,19,19,19,10,3,3,9,9,9,3,3,1])

acc_nr = np.array([100,99.3772,98.8434,98.8434,99.3772,99.3772,99.3772,99.3772,99.3772,99.3772,99.3772,99.3772,99.3772,99.3772,99.3772,99.3772,97.242])
n_leaves_nr = np.array([25,17,23,23,17,17,17,17,17,9,9,9,9,9,9,9,2])
t_size_nr = np.array([30,19,27,26,19,19,19,19,19,10,10,10,10,10,10,10,3])

#plotting reduced error pruning features
plt.title('Reduced Error Pruning')
plt.plot(minNumObj,n_leaves,'b-')

plt.plot(minNumObj,t_size,'g-')
plt.xlabel('minNumObj')
plt.legend(('No of Leaves','Tree Size'))
plt.plot(minNumObj,n_leaves,'bo')
plt.plot(minNumObj,t_size,'go')
plt.savefig('rederrPrune.png',format='png')
plt.show()

plt.title('Reduced Error Pruning Accuracy')
plt.plot(minNumObj,acc,'r-')

plt.plot(minNumObj,acc,'ro')
plt.xlabel('minNumObj')
plt.ylabel('Accuracy in percentage')
plt.savefig('rederrPrune_acc.png',format='png')
plt.show()

#plotting without reduced error pruning features
plt.title('Without Reduced Error Pruning')
plt.plot(minNumObj,n_leaves_nr,'b-')

plt.plot(minNumObj,t_size_nr,'g-')
plt.xlabel('minNumObj')
plt.legend(('No of Leaves','Tree Size'))
plt.plot(minNumObj,n_leaves_nr,'bo')
plt.plot(minNumObj,t_size_nr,'go')
plt.savefig('nonrederrPrune.png',format='png')
plt.show()

plt.title('Accuracy')
plt.plot(minNumObj,acc_nr,'r-')

plt.plot(minNumObj,acc_nr,'ro')
plt.xlabel('minNumObj')
plt.ylabel('Accuracy in percentage')
plt.savefig('nonrederrPrune_acc.png',format='png')
plt.show()

#comparison plot
plt.title('Improvement with Reduced Error Pruning')
plt.plot(minNumObj,t_size_nr,'g-')
plt.plot(minNumObj,t_size,'b-')
plt.legend(('Without Reduced Error Pruning','Reduced Error Pruning'))
plt.xlabel('minNumObj')
plt.ylabel('Tree Size')
plt.plot(minNumObj,t_size_nr,'go')
plt.plot(minNumObj,t_size,'bo')
plt.savefig('compare.png',format='png')
plt.show()


