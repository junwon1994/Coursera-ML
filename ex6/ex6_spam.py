#  Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

import numpy as np
import scipy.io
from sklearn import svm

from processEmail import processEmail
from emailFeatures import emailFeatures
from getVocabList import getVocabList


def readFile(filename):
    f = open(filename)
    file_contents = ''.join(f.readlines())
    f.close()
    return file_contents


#  ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

print('Preprocessing sample email (emailSample1.txt)')

# Extract Features
file_contents = readFile('emailSample1.txt')
word_indices = processEmail(file_contents)

# Print Stats
print('Word Indices: \n', end='')
print(word_indices, end='')
print('\n\n', end='')

input('Program paused. Press Enter to continue.')

#  ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n.
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.

print('\nExtracting features from sample email (emailSample1.txt)\n', end='')

# Extract Features
file_contents = readFile('emailSample1.txt')
word_indices = processEmail(file_contents)
features = emailFeatures(word_indices)

# Print Stats
print('Length of feature vector: %d\n' % features.size, end='')
print('Number of non-zero entries: %d\n' % sum(features > 0), end='')

input('Program paused. Press Enter to continue.')

#  =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment
data = scipy.io.loadmat('spamTrain.mat')
X = data['X']
y = data['y'].flatten()

print('\nTraining Linear SVM (Spam Classification)\n', end='')
print('(this may take 1 to 2 minutes) ...\n', end='')

C = 0.1
clf = svm.SVC(C=C, kernel='linear', max_iter=200)
model = clf.fit(X, y)

p = model.predict(X)

print('Training Accuracy: %f\n' % (np.mean(p == y) * 100), end='')

#  =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment
data = scipy.io.loadmat('spamTest.mat')
Xtest = data['Xtest']
ytest = data['ytest'].flatten()

print('\nEvaluating the trained Linear SVM on a test set ...\n', end='')

p = model.predict(Xtest)

print('Test Accuracy: %f\n' % (np.mean(p == ytest) * 100), end='')

#  ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.

# Sort the weights and obtain the vocabulary list
weight = np.sort(model.coef_[0])[::-1]
idx = np.argsort(model.coef_[0])[::-1]
vocabList = getVocabList()

print('\nTop predictors of spam: \n', end='')
for i in range(15):
    print(' %-15s (%f) \n' % (vocabList[idx[i]], weight[i]), end='')

print('\n\n', end='')
print('\nProgram paused. Press enter to continue.')

#  =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples.
#  The following code reads in one of these emails and then uses your
#  learned SVM classifier to determine whether the email is Spam or
#  Not Spam

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!
filename = 'spamSample1.txt'

# Read and predict
file_contents = readFile(filename)
word_indices = processEmail(file_contents)
x = emailFeatures(word_indices)
p = model.predict(x.reshape(1, -1))

print('\nProcessed %s\n\nSpam Classification: %d\n' % (filename, p), end='')
print('(1 indicates spam, 0 indicates not spam)\n\n', end='')
