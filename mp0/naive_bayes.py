# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naive_bayes(train_labels, train_data, dev_data, laplace=0.01, pos_prior=0.8, silently=False):
    from collections import defaultdict

    print_values(laplace,pos_prior)
    
    pos_word_count = Counter() # Number of times words appear in positive reviews
    neg_word_count = Counter() # Number of times words appear in negative reviews
    
    for i in range(len(train_data)):
        if train_labels[i] == 1:
            pos_word_count.update(train_data[i])
        else:
            neg_word_count.update(train_data[i])

    total_pos_words = sum(pos_word_count.values()) # Total number of words in positive reviews
    total_neg_words = sum(neg_word_count.values()) # Total number of words in negative reviews
    log_pos_word_probs = defaultdict(int)
    log_neg_word_probs = defaultdict(int)

    for word in pos_word_count:
        log_pos_word_probs[word] = math.log((laplace + pos_word_count[word]) / (total_pos_words + laplace * (1 + len(pos_word_count))))
    
    for word in neg_word_count:
        log_neg_word_probs[word] = math.log((laplace + neg_word_count[word]) / (total_neg_words + laplace * (1 + len(neg_word_count))))
    
    log_unknown_pos_word_prob = math.log(laplace / (total_pos_words + laplace * (1 + len(pos_word_count))))
    log_unknown_neg_word_prob = math.log(laplace / (total_neg_words + laplace * (1 + len(neg_word_count))))

    yhats = []
    for review in tqdm(dev_data, disable=silently):
        log_pos_prob = math.log(pos_prior)
        log_neg_prob = math.log(1 - pos_prior)
        
        for word in review:
            if word in pos_word_count:
                log_pos_prob += log_pos_word_probs[word]
            else:
                log_pos_prob += log_unknown_pos_word_prob

            if word in neg_word_count:
                log_neg_prob += log_neg_word_probs[word]
            else:
                log_neg_prob += log_unknown_neg_word_prob
                
        if log_pos_prob > log_neg_prob:
            yhats.append(1)
        else:
            yhats.append(0) 
            
    return yhats
    
