#importing libraries
import numpy as np
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

#Data needn't be standardised for performing LDA, dir of max separation of means is the same

#reading training data
df=pd.read_csv('../../Dataset/DS3/train.csv', sep=',',header=None)
X_train = df.values
X_train = np.array(X_train)


df=pd.read_csv('../../Dataset/DS3/train_labels.csv', sep=',',header=None)
Y_train = df.values
Y_train = np.array(Y_train[:,0])

#By default 1 component only, because two classes
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, Y_train)

X_train_lda = clf.transform(X_train)

#creating array of coefficients
W = np.append(np.array([clf.intercept_]), np.array(clf.coef_).T, axis=0)
#turning in coefficients in a csv file
my_df = pd.DataFrame(W)
my_df.to_csv('coeffs.csv', index=False, header=False)


#reading test data
df=pd.read_csv('../../Dataset/DS3/test.csv', sep=',',header=None)
X_test = df.values
X_test = np.array(X_test)
X_test_lda = clf.transform(X_test)
print np.shape(X_test), np.shape(X_test_lda)

#not an iterative algorithm, so we don't need validation set

df=pd.read_csv('../../Dataset/DS3/test_labels.csv', sep=',',header=None)
Y_test = df.values
Y_test = np.array(Y_test[:,0])


#predicting using LDA
Y_predicted = clf.predict(X_test)


#Calculating best fit parameters
target_names = ['Class 0', 'Class 1']
print(classification_report(Y_test, Y_predicted, target_names=target_names))

#Weight vectors are given as coeffs
print clf.intercept_
x_lda =  clf.coef_[0][0]
y_lda =  clf.coef_[0][1]
z_lda =  clf.coef_[0][2]
print "LDA Projected Dimension is found to be along:"
print x_lda, y_lda, z_lda

X_train_class0 = X_train[np.where( Y_train == 1 )]
X_train_class1 = X_train[np.where( Y_train == 2 )]
X_test_class0 = X_test[np.where( Y_test == 1 )]
X_test_class1 = X_test[np.where( Y_test == 2 )]

X_train_lda0 = np.array(X_train_lda[np.where( Y_train == 1 )])
X_train_lda1 = np.array(X_train_lda[np.where( Y_train == 2 )])
X_test_lda0 = np.array(X_test_lda[np.where( Y_test == 1 )])
X_test_lda1 = np.array(X_test_lda[np.where( Y_test == 2 )])

#Classes are known to be uniformly distributed
length_tr = len(X_train_lda0[:,0])
length_test = len(X_test_lda0[:,0])

# create x,y
xx, yy = np.meshgrid(range(-2,6), range(-2,6))
# calculate corresponding z
z = (-x_lda*xx - y_lda*yy - clf.intercept_[0])/(z_lda)

#3D PLOT
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(X_train_class0[:,0], X_train_class0[:,1], X_train_class0[:,2], 'ro',label='Class 0 train',zorder=-2)
ax.plot(X_train_class1[:,0], X_train_class1[:,1], X_train_class1[:,2], 'bo',label='Class 1 train',zorder=-2)
ax.plot(X_test_class0[:,0], X_test_class0[:,1], X_test_class0[:,2], 'g^',label='Class 0 test',zorder=-1)
ax.plot(X_test_class1[:,0], X_test_class1[:,1], X_test_class1[:,2], 'y^',label='Class 1 test',zorder=-1)

plt3d = plt.gca()
plt3d.hold(True)
plt3d.plot_surface(xx, yy, z,alpha=0.4,zorder=-4)

plt.title('LDA 3D points')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
plt.savefig('lda3D.png',format='png')
plt.show()
#Blue(Class 1) points are above Red(Class 0) points
# create x,y
xx, yy = np.meshgrid(range(-7,7), range(-8,7))

z = (-x_lda*xx - y_lda*yy - clf.intercept_[0])/(z_lda)
# plot the surface
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, z,alpha=0.4,zorder=-4,linewidth=0)

#Projected Plot
mag = np.sqrt(x_lda**2.0+y_lda**2.0+z_lda**2.0)
x_lda = x_lda/mag
y_lda = y_lda/mag
z_lda = z_lda/mag
intercept = clf.intercept_[0]/mag
ax = plt.gca()
ax.hold(True)

ax.plot(x_lda*(X_train_lda0[:,0]-intercept), y_lda*(X_train_lda0[:,0]-intercept), z_lda*(X_train_lda0[:,0]-intercept), 'r.',label='Class 0 train',zorder=-2)
ax.plot(x_lda*(X_train_lda1[:,0]-intercept), y_lda*(X_train_lda0[:,0]-intercept), z_lda*(X_train_lda0[:,0]-intercept), 'b.',label='Class 1 train',zorder=-2)
ax.plot(x_lda*(X_test_lda0[:,0]-intercept), y_lda*(X_test_lda0[:,0]-intercept), z_lda*(X_test_lda0[:,0]-intercept), 'gx',label='Class 0 test',zorder=-1)
ax.plot(x_lda*(X_test_lda1[:,0]-intercept), y_lda*(X_test_lda0[:,0]-intercept), z_lda*(X_test_lda0[:,0]-intercept), 'yx',label='Class 1 test',zorder=-1)

plt.title('LDA Projected points')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
plt.savefig('ldaProjected.png',format='png')
plt.show()

