import nltk
import numpy
from nltk import MaxentClassifier
from nltk import NaiveBayesClassifier

#opens source csv, adds data into an array, and closes it
montyPython = open('pathOfCSV', 'r')
pythonLines = montyPython.readlines()
montyPython.close()

#list of split lines, including indices
listOfSplits = []

#list of split lines with no indices
finalList = []

#splits pythonLines by commas, so that each field on a line is a separate string
for lines in range (1,4000):
    listOfSplits.append(pythonLines[lines].split(","))
    
#removes the index from listOfSplits, so that just actor and dialogue are left
for each in listOfSplits:
    del each[0]
    finalList.append(each)

#returns line length
def line_length(line):
    return len(line[1])

#returns first word of line
def first_word(line):
    dialogue = line[1]
    words = dialogue.split(" ")
    return words[0]

#returns last word of line
def last_word(line):
    dialogue = line[1]
    words = dialogue.split(" ")
    return words[-1]

#returns longest word of line
def longest_word(line):
    dialogue = line[1]
    words = dialogue.split(" ")
    longestWord = 0
    for word in words:
        if len(word) > longestWord:
            longestWord = len(word)
        else:
            longestWord = longestWord
    return longestWord

#returns dict of all features with feature names
def dialogue_features(entry):
    features = {}
    features["lineLength"] = line_length(each)
    features["firstWord"] = first_word(each)
    features["lastWord"] = last_word(each)
    features["longestWord"] = longest_word(each)
    return features

#returns list of tuples with dict returned by dialogue_features and actor names
featureSet = []
for each in finalList:
    mappedFeatures = dialogue_features(each)
    featureSet.append((mappedFeatures, each[0]))

#defines training set, complete test set with actor name, and usable test set
trainingSet = featureSet[3000:]
prelimTestSet = featureSet[:3000]
testSet = []
for each in prelimTestSet:
    testSet.append(each[0])

#change these ints to change the test entries: should be safe up to 500 or so
#first one is used by bayes, second is used by maxent
a = 10
b = 111

print('Training Naive Bayes')
bayesClassifier = NaiveBayesClassifier.train(trainingSet)
print('Naive Bayes training complete')

print('Naive Bayes most important features:')
bayesClassifier.show_most_informative_features(5)

print('Dialogue:')
print(prelimTestSet[a])
print('Bayes classification:')
print(bayesClassifier.classify(testSet[a]))

print('Training Maximum Entropy')
maxEntClassifier = MaxentClassifier.train(trainingSet, max_iter=30)
print('Maximum Entropy training complete')

print('Maximum Entropy most important features:')
maxEntClassifier.show_most_informative_features(5)

print('Dialogue:')
print(prelimTestSet[b])
print('MaxEnt classification:')
print(maxEntClassifier.classify(testSet[b]))
          

