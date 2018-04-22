def getVocabList():
    """reads the fixed vocabulary list in vocab.txt
    and returns a cell array of the words in vocabList.
    """

    # Read the fixed vocabulary list
    file = open('vocab.txt')

    # For ease of implementation,
    # we use a struct to map the integers => string
    # In practice, you'll want to use some form of hashmap
    vocabList = {}

    for line in file.readlines():
        idx, word = line.split()
        vocabList[int(idx)] = word

    file.close()

    return vocabList
