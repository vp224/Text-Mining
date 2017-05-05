from math import log, sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from os import listdir
from os.path import isfile, join
from pyspark import SparkContext

import os

def termfrequency(term, document):
	normalizeDocument = document.lower().replace(".", "").replace("\\", " ").split()
	return normalizeDocument.count(term.lower())/float(len(normalizeDocument))

def inverseDocumentFrequency(term, allDocuments):
    numDocumentsWithThisTerm = 0
    for doc in xrange(len(allDocuments)):
        if term.lower() in allDocuments[doc].lower().split():
            numDocumentsWithThisTerm = numDocumentsWithThisTerm + 1
 
    if numDocumentsWithThisTerm > 0:
        return 1.0 + log(float(len(allDocuments)) / numDocumentsWithThisTerm)
    else:
        return 1.0

word1 = "and"
word2 = "the"
vectors = []
onlyfiles = []
filenames = []
len_files = []


def removedPunct(data):
	a = data.lower().replace("\n", "").replace(".","").replace(",","").replace(";","").replace("\\","")


sc = SparkContext()
def word_count(words, onlyFiles):
	for file in onlyFiles:
		words = []
		count = 0
		data = sc.textFile("test.txt")
		# data = open(file)
		processed = data.map(removedPunct,data)
		# word = processed.map(spliter, processed)
		# processed.saveAsTextFile("NOPUNCT"+`i`)	
		for k in words:
			count += 1
			words.append(processed)

for f in os.listdir(os.curdir):
	if os.path.basename(f) != 'cosine.py' and isfile(f):
		text = open(f).read()
		if word1 in text or word2 in text:
			onlyfiles.append(text)
			filenames.append(os.path.basename(f))
			len_files.append(len(os.path.basename(f)))

for file in onlyfiles:
	word_count(word1, onlyfiles)
	i_d1 = termfrequency(word1, file)*inverseDocumentFrequency(word1, onlyfiles)
	j_d1 = termfrequency(word2, file)*inverseDocumentFrequency(word2, onlyfiles)
	if i_d1 != 0 or j_d1 != 0:
		vectors.append([0, 0, i_d1/(sqrt(i_d1**2 + j_d1**2)), j_d1/(sqrt(i_d1**2 + j_d1**2))])

soa = np.array(vectors)
X, Y, U, V = zip(*soa)
plt.figure()
ax = plt.gca()
limit = 10
colors = ['m','r', 'g', 'b', 'y']
quiver = []
for i in xrange(len(vectors)):
	quiver.append(ax.quiver(X[i], Y[i], U[i], V[i], angles='xy', scale_units='xy', scale=1.0, color=colors[i%5]))
	ax.quiverkey(quiver[i], U[i], V[i], 0, filenames[i])

plt.xlabel(word1)
plt.ylabel(word2)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.draw()
plt.show()