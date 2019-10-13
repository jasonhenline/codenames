import numpy


class DataSet:
    def __init__(self, wordIndices, featureValues):
        self.wordIndices = wordIndices
        self.featureValues = featureValues
        self.norms = numpy.linalg.norm(self.featureValues, axis=1)
        self.wordsList = len(self.wordIndices)*[None]
        for word, index in self.wordIndices.items():
            self.wordsList[index] = word

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
        return bestWord


if __name__ == '__main__':
    dataSet = DataSet.loadFromFile(
        'wiki-news-300d-1M.vec', wordCountLimit=300_000)
    while True:
        wordsToClue = input(
            'Enter a space-separated list of words and I will give you the best match\n').split()
        wordsNotToClue = input(
            'Enter a space-separated list of words not to be clued\n'
        ).split()
        print(dataSet.getWordWithMaximumMinimumSimilarity(
            wordsToClue, wordsNotToClue=wordsNotToClue))
