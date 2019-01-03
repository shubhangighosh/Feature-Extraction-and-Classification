#importing libraries
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

#reading training data
df=pd.read_csv('../../Dataset/DS3/train.csv', sep=',',header=None)
X_train = df.values
X_train = np.array(X_train)

df=pd.read_csv('../../Dataset/DS3/train_labels.csv', sep=',',header=None)
Y_train_class1 = df.values
Y_train_class1 = np.array(Y_train_class1)
Y_train_class1 = Y_train_class1 - 1
Y_train_class0 =  1 - Y_train_class1
Y_train = np.append(Y_train_class0, Y_train_class1, axis=1)

#standardising data 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

#getting PCA components
pca = PCA(n_components=1, svd_solver='full')
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train)



#fitting linear regression model to both indicator variables
reg = linear_model.LinearRegression()
reg.fit (X_train_pca, Y_train) 
#creating array of coefficients
W = np.append(np.array([reg.intercept_]), np.array(reg.coef_).T, axis=0)
#turning in coefficients in a csv file
my_df = pd.DataFrame(W)
my_df.to_csv('coeffs.csv', index=False, header=False)
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)

#reading test data
df=pd.read_csv('../../Dataset/DS3/test.csv', sep=',',header=None)
X_test = df.values
X_test = np.array(X_test)
X_test_pca = pca.transform(X_test)

#not an iterative algorithm, so we don't need validation set

df=pd.read_csv('../../Dataset/DS3/test_labels.csv', sep=',',header=None)
Y_test_class1 = df.values
Y_test_class1 = np.array(Y_test_class1)
Y_test_class1 = Y_test_class1 - 1
Y_test_class0 =  1 - Y_test_class1
Y_test = np.append(Y_test_class0, Y_test_class1, axis=1)

#predicting using Linear Regression object
Y_predicted = reg.predict(X_test_pca)
#finding argmax to predict class
Y_predicted_arr = np.argmax(Y_predicted, axis =1)
Y_true = np.argmax(Y_test, axis =1)

#Calculating best fit parameters
target_names = ['Class 0', 'Class 1']
print(classification_report(Y_true, Y_predicted_arr, target_names=target_names))

#PCA projected direction
print "The PCA projected direction was found to be along the (x,y,z) vector:"

print pca.components_
x_pca =  pca.components_[0][0]
y_pca =  pca.components_[0][1]
z_pca =  pca.components_[0][2]


#separating class points
X_train_class0 = X_train[np.where( Y_train_class1[:,0] == 0 )]
X_train_class1 = X_train[np.where( Y_train_class1[:,0] == 1 )]
X_test_class0 = X_test[np.where( Y_test_class1[:,0] == 0 )]
X_test_class1 = X_test[np.where( Y_test_class1[:,0] == 1 )]

X_train_pca0 = np.array(X_train_pca[np.where( Y_train_class1[:,0] == 0 )])
X_train_pca1 = np.array(X_train_pca[np.where( Y_train_class1[:,0] == 1 )])
X_test_pca0 = np.array(X_test_pca[np.where( Y_test_class1[:,0] == 0 )])
X_test_pca1 = np.array(X_test_pca[np.where( Y_test_class1[:,0] == 1 )])

#Classes are known to be uniformly distributed
length_tr = len(X_train_pca0[:,0])
length_test = len(X_test_pca0[:,0])

#point through which plane passes and normal to plane acc to pca direction


coeff1 = reg.coef_[0][0]
coeff2 = reg.coef_[1][0]
coeff = coeff1 - coeff2
intercept1 = reg.intercept_[0]
intercept2 = reg.intercept_[1]
intercept = intercept1 - intercept2
# create x,y
xx, yy = np.meshgrid(range(-2,6), range(-2,6))
# calculate corresponding z
z = (-coeff*x_pca*xx - coeff*y_pca*yy - intercept)/(coeff*z_pca)

#3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(X_train_class0[:,0], X_train_class0[:,1], X_train_class0[:,2], 'ro',label='Class 0 train',zorder=-2)
ax.plot(X_train_class1[:,0], X_train_class1[:,1], X_train_class1[:,2], 'bo',label='Class 1 train',zorder=-2)
ax.plot(X_test_class0[:,0], X_test_class0[:,1], X_test_class0[:,2], 'g^',label='Class 0 test',zorder=-1)
ax.plot(X_test_class1[:,0], X_test_class1[:,1], X_test_class1[:,2], 'y^',label='Class 1 test',zorder=-1)

plt3d = plt.gca()
plt3d.hold(True)
plt3d.plot_surface(xx, yy, z,alpha=0.2,zorder=2)

plt.title('PCA 3D points')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
plt.savefig('pca3D.png',format='png')
plt.show()

# plot the surface
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, z,alpha=0.2,zorder=2)

ax = plt.gca()
ax.hold(True)


#Projected Plot
ax.plot(x_pca*X_train_pca0[:,0], y_pca*X_train_pca0[:,0], z_pca*X_train_pca0[:,0], 'r.',label='Class 0 train',zorder=-2)
ax.plot(x_pca*X_train_pca1[:,0], y_pca*X_train_pca0[:,0], z_pca*X_train_pca0[:,0], 'b.',label='Class 1 train',zorder=-2)
ax.plot(x_pca*X_test_pca0[:,0], y_pca*X_test_pca0[:,0], z_pca*X_test_pca0[:,0], 'gx',label='Class 0 test',zorder=-1)
ax.plot(x_pca*X_test_pca1[:,0], y_pca*X_test_pca0[:,0], z_pca*X_test_pca0[:,0], 'yx',label='Class 1 test',zorder=-1)

plt.title('PCA Projected points')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
plt.savefig('pcaProjected.png',format='png')
plt.show()

