"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: initial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    tag_count = defaultdict(lambda: 0)
    next_tag_count = defaultdict(lambda: 0)
    
    for sentence in sentences:
        init_prob[sentence[0][1]] += 1
        for i in range(len(sentence)):
            word, tag = sentence[i]
            emit_prob[tag][word] += 1
            tag_count[tag] += 1
            if i < len(sentence) - 1:
                next_tag_count[tag] += 1
                next_tag = sentence[i+1][1]
                trans_prob[tag][next_tag] += 1
    
    for tag in init_prob:
        init_prob[tag] /= len(sentences)
        
    for tag in trans_prob:
        for word in trans_prob[tag]:
            trans_prob[tag][word] = (trans_prob[tag][word] + epsilon_for_pt) / (next_tag_count[tag] + epsilon_for_pt * (len(trans_prob[tag]) + 1))
            
    for tag in emit_prob:
        for word in emit_prob[tag]:
            emit_prob[tag][word] = (emit_prob[tag][word] + emit_epsilon) / (tag_count[tag] + emit_epsilon * (len(emit_prob[tag]) + 1))
        emit_prob[tag]["UNKNOWN"] = (emit_epsilon) / (tag_count[tag] + emit_epsilon * (len(emit_prob[tag]) + 1))
    
    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.

    for tag in emit_prob:
        if i == 0:
            t = trans_prob["START"].get(tag, epsilon_for_pt)
            e = emit_prob[tag].get(word, emit_prob[tag]["UNKNOWN"])
            log_prob[tag] = math.log(t) + math.log(e)
            predict_tag_seq[tag] = [tag]
        else:
            max_prob = float('-inf')
            max_tag = None

            for prev_tag in prev_prob:
                t = trans_prob[prev_tag].get(tag, epsilon_for_pt)
                e = emit_prob[tag].get(word, emit_prob[tag]["UNKNOWN"])

                temp = prev_prob[prev_tag] + math.log(t) + math.log(e)

                if temp > max_prob:
                    max_prob = temp
                    max_tag = prev_tag

            log_prob[tag] = max_prob
            predict_tag_seq[tag] = prev_predict_tag_seq[max_tag] + [tag]
    
    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        tag = max(log_prob, key=log_prob.get)
        seq = predict_tag_seq[tag]
        predicts.append(list(zip(sentence, seq)))
        
        
    return predicts