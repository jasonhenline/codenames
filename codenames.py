import itertools
import json
import numpy


class DataSet:
    def __init__(self, wordIndices, featureValues):
        self.wordIndices = wordIndices
        self.featureValues = featureValues
        self.norms = numpy.linalg.norm(self.featureValues, axis=1)
        self.wordsList = len(self.wordIndices)*[None]
        for word, index in self.wordIndices.items():
            self.wordsList[index] = word

    def getFeatureVector(self, word):
        index = self.wordIndices[word]
        return self.featureValues[index]

    @staticmethod
    def loadFromFile(fileName, wordCountLimit=None):
        with open(fileName) as fp:
            headerLine = fp.readline()
            wordCount, featureCount = [int(t) for t in headerLine.split()]
            wordCountLimit = min(
                wordCount, wordCountLimit) if wordCountLimit else wordCount
            wordIndices = {}
            featureValues = numpy.empty((wordCountLimit, featureCount))
            nextIndex = 0
            for _ in range(wordCountLimit):
                line = fp.readline()
                tokens = line.split()
                word = tokens[0].strip().lower()
                if word in wordIndices:
                    continue
                wordIndices[word] = nextIndex
                wordFeatureValues = numpy.array(
                    [float(t) for t in tokens[1:]])
                featureValues[nextIndex] = wordFeatureValues
                nextIndex += 1
            featureValues.resize((nextIndex, featureCount))
        return DataSet(wordIndices=wordIndices, featureValues=featureValues)

    def getCosineSimilarities(self, word):
        indexForThisWord = self.wordIndices[word]
        featuresForThisWord = self.featureValues[indexForThisWord]
        normForThisWord = self.norms[indexForThisWord]
        dots = numpy.dot(self.featureValues, featuresForThisWord)
        similarities = numpy.divide(numpy.divide(
            dots, self.norms), normForThisWord)
        return similarities

    @staticmethod
    def areTooSimilar(word1, word2):
        if word1 in word2 or word2 in word1:
            return True
        return False

    def getCosineSimilarity(self, word1, word2):
        v1 = self.getFeatureVector(word1)
        v2 = self.getFeatureVector(word2)
        return numpy.dot(v1, v2) / numpy.linalg.norm(v1) / numpy.linalg.norm(v2)

    def getWordWithMaximumMinimumSimilarity(self, wordsToClue, wordsNotToClue=[], illegalWords=[]):
        clueSimilarities = []
        for clueWord in wordsToClue:
            clueSimilarities.append(self.getCosineSimilarities(clueWord))
        minClueSimilarities = numpy.min(clueSimilarities, axis=0)

        if wordsNotToClue:
            notClueSimilarities = []
            for notClueWord in wordsNotToClue:
                notClueSimilarities.append(
                    self.getCosineSimilarities(notClueWord))
            maxNotClueSimilarities = numpy.max(notClueSimilarities, axis=0)

        # Now we have to find the best minimum similarity, but it cannot be
        # with any form of any word in the word list, and its minimum
        # similarity with a clue word must be better than its maximum
        # similarity to a notClueWord.
        bestMinimum = -1
        bestWord = None
        for index, word in enumerate(self.wordsList):
            if any(DataSet.areTooSimilar(word, clueWord) for clueWord in wordsToClue):
                continue
            minClueSimilarity = minClueSimilarities[index]
            if minClueSimilarity > bestMinimum:
                if wordsNotToClue and maxNotClueSimilarities[index] >= minClueSimilarity:
                    continue
                bestMinimum = minClueSimilarity
                bestWord = word
        return bestWord, bestMinimum


if __name__ == '__main__':
    dataSet = DataSet.loadFromFile(
        'wiki-news-300d-1M.vec', wordCountLimit=500_000)
    fileName = 'crud.json'
    while True:
        newFileName = input(
            f'Input the name of a config file (or just hit enter to use {fileName})\n')
        fileName = newFileName if newFileName else fileName
        config = json.load(open(fileName))
        wordsToClue = config['wordsToClue']
        wordsNotToClue = config['wordsNotToClue']
        for subsetSize in range(1, len(wordsToClue) + 1):
            bestClue = None
            bestSubset = None
            bestWorstSimilarity = -1
            for subset in itertools.combinations(wordsToClue, subsetSize):
                clue, worstSimilarityForClue = dataSet.getWordWithMaximumMinimumSimilarity(
                    wordsToClue=subset, wordsNotToClue=wordsNotToClue)
                if clue and worstSimilarityForClue > bestWorstSimilarity:
                    bestClue = clue
                    bestSubset = subset
                    bestWorstSimilarity = worstSimilarityForClue
            print('SUBSET =', bestSubset, ", BEST_CLUE =", bestClue,
                  ", WORST_SIMILARITY =", bestWorstSimilarity)
            print('WORDS TO CLUE:')
            for word in wordsToClue:
                similarity = dataSet.getCosineSimilarity(bestClue, word)
                print("similarity =", similarity, ', word =', word)
            print('WORDS TO AVOID:')
            for word in wordsNotToClue:
                similarity = dataSet.getCosineSimilarity(bestClue, word)
                print("similarity =", similarity, ', word =', word)
