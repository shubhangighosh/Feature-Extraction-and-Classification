

#importing libraries
import numpy as np
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_validate


#Data needn't be standardised for performing LDA, dir of max separation of means is the same

#reading training data
df=pd.read_csv('../../Dataset/iris/iris.csv', sep=',',header=None)
X = df.values
X = np.array(X)
Y_str = np.array(X[:-1,4:])
X = np.array(X[:-1,2:4])

#appending class labels

Y = []
for item in Y_str:
    if item == 'Iris-setosa':
    	Y.append(1)
    elif item == 'Iris-versicolor':
    	Y.append(2)
    elif item == 'Iris-virginica':
    	Y.append(3)	
Y = np.array(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=5)



#By default 1 component only, cuz two classes
clf = LinearDiscriminantAnalysis(store_covariance=True)
clf.fit(X_train, Y_train)

X_train_lda = clf.transform(X_train)

#creating array of coefficients
W = np.append(np.array([clf.intercept_]), np.array(clf.coef_).T, axis=0)
#turning in coefficients in a csv file
my_df = pd.DataFrame(W)
my_df.to_csv('coeffs.csv', index=False, header=False)


#predicting using LDA
Y_predicted = clf.predict(X_test)


#Calculating best fit parameters
target_names = ['Iris-setosa', 'Iris-versicolor','Iris-virginica']
print 'LDA Classification Measures'
print(classification_report(Y_test, Y_predicted, target_names=target_names))

#separating class points

X_train1 = X_train[np.where( Y_train == 1 )]
X_train2 = X_train[np.where( Y_train == 2 )]
X_train3 = X_train[np.where( Y_train == 3 )]

X_test1 = X_test[np.where( Y_test == 1 )]
X_test2 = X_test[np.where( Y_test == 2 )]
X_test3 = X_test[np.where( Y_test == 3 )]

c1 = W[0][1]-W[0][0]
a1 = W[1][1]-W[1][0]
b1 = W[2][1]-W[2][0]

c2 = W[0][2]-W[0][1]
a2 = W[1][2]-W[1][1]
b2 = W[2][2]-W[2][1]

#Decision Boundary Plot

fig, ax = plt.subplots()
ymin = 0.0
ymax = 3.0
xmin = -(c1+b1*ymin)/a1
xmax = -(c1+b1*ymax)/a1

plt.plot([xmin, xmax], [ymin, ymax], 'k-')


xmin = -(c2+b2*ymin)/a2
xmax = -(c2+b2*ymax)/a2
plt.plot([xmin, xmax], [ymin, ymax], 'k-')

ax = plt.gca()
ax.hold(True)

ax.scatter(X_train1[:,0], X_train1[:,1],color='r',label='setosa-train')
ax.scatter(X_train2[:,0], X_train2[:,1],color='b',label='versicolor-train')
ax.scatter(X_train3[:,0], X_train3[:,1],color='g',label='virginica-train')

ax.scatter(X_test1[:,0], X_test1[:,1],color='y',label='setosa-test')
ax.scatter(X_test2[:,0], X_test2[:,1],color='c',label='versicolor-test')
ax.scatter(X_test3[:,0], X_test3[:,1],color='m',label='virginica-test')
plt.title('LDA Analysis')

ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal Width')
ax.set_ylim([0,3])
plt.legend(numpoints=1, ncol=3, fontsize=8)
plt.savefig('LDA.png',format='png')
plt.show()

#classifier(LinearDiscriminantAnalysis,X_train, X_test, Y_train, Y_test, 'LDA')
#QDA doesn't perform feature selection, so no transform for QDA

def qda_reg(X_train, X_test, Y_train, Y_test, title, alpha):
	qda = QuadraticDiscriminantAnalysis(store_covariance=True)
	qda.fit(X_train, Y_train)
	Y_predicted_qda = qda.predict(X_test)
	print title + " Classification Report"
	target_names = ['Iris-setosa', 'Iris-versicolor','Iris-virginica']
	print(classification_report(Y_test, Y_predicted_qda, target_names=target_names))

	

	# plotting decision boundary
	nx, ny = 200, 100
	x_min, x_max = (0.0,10.0)

	y_min, y_max = (0.0,3.0)
	xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),np.linspace(y_min, y_max, ny))
	Z = qda.decision_function(np.c_[xx.ravel(), yy.ravel()])
	Z2 = Z[:, 2].reshape(xx.shape)
	Z1 = Z[:, 1].reshape(xx.shape)
	Z0 = Z[:, 0].reshape(xx.shape)

	bound1 = Z1 - Z0
	bound2 = Z2 - Z1

	#plotting where difference of decision fn is 0
	CS = plt.contour(xx, yy, bound1, levels = [0], color='k')
	plt.contour(xx, yy, bound2, levels = [0], color='k')
	
	ax = plt.gca()
	ax.hold(True)

	ax.scatter(X_train1[:,0], X_train1[:,1],color='r',label='setosa-train')
	ax.scatter(X_train2[:,0], X_train2[:,1],color='b',label='versicolor-train')
	ax.scatter(X_train3[:,0], X_train3[:,1],color='g',label='virginica-train')

	ax.scatter(X_test1[:,0], X_test1[:,1],color='y',label='setosa-test')
	ax.scatter(X_test2[:,0], X_test2[:,1],color='c',label='versicolor-test')
	ax.scatter(X_test3[:,0], X_test3[:,1],color='m',label='virginica-test')
	plt.title(title + ' Analysis')

	ax.set_xlabel('Petal Length')
	ax.set_ylabel('Petal Width')
	ax.set_ylim([0,3])
	plt.legend(numpoints=1, ncol=3, fontsize=8)
	plt.savefig(title +'.png',format='png')
	plt.show()



#RDA
#alpha=0 --> LDA
#alpha=1 --> QDA
qda_reg(X_train, X_test, Y_train, Y_test, 'QDA', 0.0)
alpha = np.arange(0.0,1.01,0.01) #to be changed using cross-validation

#performing RDA and Cross-Validation
def rda(X,Y, alpha):
	rda = QuadraticDiscriminantAnalysis(store_covariance=True, reg_param = alpha)
	scores = cross_validate(rda, X, Y, cv=5)
	return np.mean(scores['test_score']), np.mean(scores['train_score']) 

cv_test_scores = []
cv_train_scores = []

#Plotting cross validation test score and training scores
for i in alpha:
	cv_test_scores.append(rda(X_train,Y_train,i)[0])
	cv_train_scores.append(rda(X_train,Y_train,i)[1])
cv_test_scores = np.array(cv_test_scores)
cv_train_scores = np.array(cv_train_scores)
plt.plot(alpha,cv_test_scores,'b-',linewidth=2)
plt.plot(alpha,cv_train_scores,'g-',linewidth=2)
plt.legend(('Test Score', 'Train Score'), loc='best')
ax = plt.gca()
plt.title('RDA Cross-Validation')
ax.set_xlabel('Regularisation Parameter')
ax.set_ylabel('Cross Validation Score')
plt.savefig('RDA_cv.png',format='png')
plt.show()	

#performing RDA and plotting decision boundary
qda_reg(X_train, X_test, Y_train, Y_test, 'RDA', 0.6)

