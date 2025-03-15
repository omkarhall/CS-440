"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

from collections import defaultdict


def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    counts = defaultdict(lambda: defaultdict(lambda: 0))
    tags = defaultdict(lambda: 0)
    
    for sentence in train:
        for word, tag in sentence:
            counts[word][tag] += 1
            tags[tag] += 1
            
    res = []
    for sentence in test:
        temp = []
        for word in sentence:
            if word in counts:
                tag = max(counts[word], key=counts[word].get)
            else:
                tag = max(tags, key=tags.get)
            temp.append((word, tag))
        res.append(temp)
    return res