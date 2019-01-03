
#importing libraries
import numpy as np
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_validate
import pickle
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.externals import joblib




#To read required data from file and store in array
def read_data(path):
	f = open ( path , 'r')

	
	line = f.readline()
	while line.startswith("%"):
		last_pos = f.tell()
		line = f.readline()
	f.seek(last_pos)
	line = f.read().splitlines()

	size = line[0].split()
	
	line = line[1:]
	i1 = int(size[0])
	j1 = int(size[1])
	
	
	k=0
	arr = np.zeros((i1,j1))
	for j  in xrange(j1):
	        for i in xrange(i1):
	            arr[i][j]=int(line[k])
	            k+=1 
	f.close()                       

	return arr
#storing data required for this question in an array	
X_train = np.array(read_data('../../Dataset/DS2/data_students/Train_features'))

X_test = np.array(read_data('../../Dataset/DS2/data_students/Test_features'))
Y_train = np.array(read_data('../../Dataset/DS2/data_students/Train_labels'))
Y_test = np.array(read_data('../../Dataset/DS2/data_students/Test_labels'))


#standardising data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#shuffling data for better cross-validation
X_train, Y_train = shuffle(X_train, Y_train, random_state=5)
X_test, Y_test = shuffle(X_test, Y_test, random_state=5)
#validation size increased to improve reliability
X_fit, X_val, Y_fit, Y_val = train_test_split(X_train,Y_train, test_size=0.3, random_state=5)
X_val = np.concatenate((X_val,X_test),axis=0)
Y_val = np.concatenate((Y_val,Y_test),axis=0)

cv = []
C=np.logspace(-2.0,1.0,num=4)
#Linear kernel
def svm_lin(X_fit,Y_fit, X_val, Y_val,C):
	svm_clf = SVC(kernel='linear',C=C)
	#fitting and cross-validation
	svm_clf.fit(X_fit,np.argmax(Y_fit, axis =1))
	scores = cross_validate(svm_clf, X_val, np.argmax(Y_val, axis =1), cv=5)
	return np.mean(scores['test_score'])

#finding best fit parameters and score
lin_bf =0.0
for c in C:
	x = svm_lin(X_fit,Y_fit, X_val, Y_val,c)
	#print x
	if x>lin_bf:
		lin_bf = x
		lin_bfC = c
	if not cv:
		cv = [x]
	else:		
		cv.append(x)	
		
#plotting cv		
plt.title('Cross Validation for Linear kernel')
plt.semilogx(C,cv,'b-')
plt.semilogx(C,cv,'bo')
plt.xlabel('C')
plt.ylabel('Cross Validation Score')
plt.savefig('Lin_cv.png',format='png')
plt.show()

#printing bf -- best fit
print "The best fit parameters for Linear kernel is:"
print "C: %f" %(lin_bfC)
print "The best fit cross validation score for Linear kernel: %f percent" %(lin_bf*100.0)	

clf = SVC(kernel='linear',C=lin_bfC)
clf.fit(X_fit,np.argmax(Y_fit, axis =1))
joblib.dump(clf, 'svm_model1.pkl')

#RBF
def svm_rbf(X_fit,Y_fit, X_val, Y_val,C,gamma):
	svm_clf = SVC(kernel='rbf',C=C,gamma=gamma)
	#fitting and cv
	svm_clf.fit(X_fit,np.argmax(Y_fit, axis =1))
	scores = cross_validate(svm_clf, X_val, np.argmax(Y_val, axis =1), cv=5)
	return np.mean(scores['test_score'])
	
C=np.logspace(-2.0,10.0,num=13)
gamma=np.logspace(-9.0,3.0,num=13)

#plotting cv
cv = []
rbf_bf = 0.0
rbf_bfC = 0.0
rbf_bfg = 0.0
for g in gamma:
	
	for c in C:
		x = svm_rbf(X_fit,Y_fit, X_val, Y_val,c,g)
		#print x
		if x>rbf_bf:
			rbf_bf = x
			rbf_bfC = c
			rbf_bfg = g
		if not cv:
			cv = [x]
		else:		
			cv.append(x)
	
cv=np.array(cv)	

cv = cv.reshape((13,13))

plt.imshow(cv, cmap='hot', interpolation='nearest')
ax = plt.gca()
plt.title('RBF Hyper Parameter Tuning')
ax.set_xlabel('log10(C-2)')
ax.set_ylabel('log10(gamma-9)')
plt.savefig('RBF_cv.png',format='png')
plt.show()
#printing bf
print "The best fit parameters for RBF kernel are:"
print "C: %f gamma: %f" %(rbf_bfC,rbf_bfg)
print "The best fit cross validation score for RBF kernel: %f percent" %(rbf_bf*100.0)

clf = SVC(kernel='rbf',C=rbf_bfC,gamma=rbf_bfg)
clf.fit(X_fit,np.argmax(Y_fit, axis =1))
joblib.dump(clf, 'svm_model2.pkl')




#Polynomial
def svm_pol(X_fit,Y_fit, X_val, Y_val,C,deg):
	svm_clf = SVC(kernel='poly',C=C,degree=deg)
	#fitting and cv
	svm_clf.fit(X_fit,np.argmax(Y_fit, axis =1))
	scores = cross_validate(svm_clf, X_val, np.argmax(Y_val, axis =1), cv=5)
	return np.mean(scores['test_score'])

#cv and grid search
C=np.logspace(-2.0,2.0,num=5)

deg = np.array([1,2,3,4,5])
cv = []
pol_bf =0.0
for c in C:
	for d in deg:
		x = svm_pol(X_fit,Y_fit, X_val, Y_val,c,d)
		#print x
		if x>pol_bf:
			pol_bf = x
			pol_bfC = c
			#pol_bfg = g
			#pol_bfcoef = coef
			pol_bfd = d
		if not cv:
			cv = [x]
		else:		
			cv.append(x)	


cv=np.array(cv)	
cv = cv.reshape((5,5))
#plotting cv
plt.imshow(cv, cmap='hot', interpolation='nearest')
ax = plt.gca()
plt.title('Polynomial Hyper Parameter Tuning')
ax.set_xlabel('Degree-1')
ax.set_ylabel('log10(C-2)')
plt.savefig('Poly_cv.png',format='png')
plt.show()
#printing bf
print "The best fit parameters for Polynomial kernel are:"
print "C: %f Degree: %f" %(pol_bfC,pol_bfd)
print "The best fit cross validation score for Polynomial kernel: %f percent" %(pol_bf*100.0)	

clf = SVC(kernel='poly',C=pol_bfC,degree=pol_bfd)
clf.fit(X_fit,np.argmax(Y_fit, axis =1))
#saving model
joblib.dump(clf, 'svm_model3.pkl')


#Sigmoidal
def svm_sig(X_fit,Y_fit, X_val, Y_val,C,gamma,coeff0):
	svm_clf = SVC(kernel='sigmoid',C=C,gamma=gamma,coef0=coeff0)
	#fitting and cv
	svm_clf.fit(X_fit,np.argmax(Y_fit, axis =1))
	scores = cross_validate(svm_clf, X_val, np.argmax(Y_val, axis =1), cv=5)
	return np.mean(scores['test_score'])
#grid search
C=np.logspace(-2.0,10.0,num=13)
gamma=np.arange(0.1,1.1,0.25)
coef0=np.arange(0.0,1.1,0.25)

sig_bf =0.0
for c in C:
	for g in gamma:
		for coef in coef0:
			x = svm_sig(X_fit,Y_fit, X_val, Y_val,c,g,coef)
			#print x
			if x>sig_bf:
				sig_bf = x
				sig_bfC = c
				sig_bfg = g
				sig_bfcoef = coef
				
#printing bf
print "The best fit parameters for Sigmoidal kernel are:"
print "C: %f gamma: %f coef0: %f" %(sig_bfC,sig_bfg,sig_bfcoef)
print "The best fit cross validation score for Polynomial kernel: %f percent" %(sig_bf*100.0)	

#saving model
clf = SVC(kernel='sigmoid',C=sig_bfC,gamma=sig_bfg,coef0=sig_bfcoef)
clf.fit(X_fit,np.argmax(Y_fit, axis =1))
joblib.dump(clf, 'svm_model4.pkl')			