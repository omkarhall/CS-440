import numpy as np

def initialize(data, dim):
    '''
    Initialize embeddings for all distinct words in the input data.
    Most of the dimensions will be zero-mean unit-variance Gaussian random variables.
    In order to make debugging easier, however, we will assign special geometric values
    to the first two dimensions of the embedding:

    (1) Find out how many distinct words there are.
    (2) Choose that many locations uniformly spaced on a unit circle in the first two dimensions.
    (3) Put the words into those spots in the same order that they occur in the data.

    Thus if data[0] and data[1] are different words, you should have

    embedding[data[0]] = np.array([np.cos(0), np.sin(0), random, random, random, ...])
    embedding[data[1]] = np.array([np.cos(2*np.pi/N), np.sin(2*np.pi/N), random, random, random, ...])

    ... and so on, where N is the number of distinct words, and each random element is
    a Gaussian random variable with mean=0 and standard deviation=1.

    @param:
    data (list) - list of words in the input text, split on whitespace
    dim (int) - dimension of the learned embeddings

    @return:
    embedding - dict mapping from words (strings) to numpy arrays of dimension=dim.
    '''
    distinct_words = list(dict.fromkeys(data))
    N = len(distinct_words)
    spacing = 2 * np.pi/N
    
    embedding = {}
    for i in range(0, N):
        temp = np.array([np.cos(i * spacing), np.sin(i * spacing)])
        randoms = np.random.normal(0, 1, size = (dim - 2))
        embedding[distinct_words[i]] = np.concatenate((temp, randoms))
    return embedding

def gradient(embedding, data, t, d, k):
    '''
    Calculate gradient of the skipgram NCE loss with respect to the embedding of data[t]

    @param:
    embedding - dict mapping from words (strings) to numpy arrays.
    data (list) - list of words in the input text, split on whitespace
    t (int) - data index of word with respect to which you want the gradient
    d (int) - choose context words from t-d through t+d, not including t
    k (int) - compare each context word to k words chosen uniformly at random from the data

    @return:
    g (numpy array) - loss gradients with respect to embedding of data[t]
    '''
    if (len(embedding) == 1):
        x = np.dot(embedding[data[t]],embedding[data[t]])
        return 2 * d * (2 / (1 + np.exp(x)) - 1) * embedding[data[t]]
    
    g = np.zeros_like(embedding[data[t]])
    for c in range(-d, d+1):
        if c != 0 and 0 <= t+c < len(data):  
            x = np.dot(embedding[data[t]].T, embedding[data[t+c]])
            g += (1 / (1 + np.exp(-x)) - 1) * embedding[data[t+c]]
        
    for _ in range(1, k+1):
        rand_word = np.random.choice(data)
        x = np.dot(embedding[data[t]].T, embedding[rand_word])
        g += (1 / (1 + np.exp(-x))) * embedding[rand_word]
        
    return g
           
def sgd(embedding, data, learning_rate, num_iters, d, k):
    '''
    Perform num_iters steps of stochastic gradient descent.

    @param:
    embedding - dict mapping from words (strings) to numpy arrays.
    data (list) - list of words in the input text, split on whitespace
    learning_rate (scalar) - scale the negative gradient by this amount at each step
    num_iters (int) - the number of iterations to perform
    d (int) - context width hyperparameter for gradient computation
    k (int) - noise sample size hyperparameter for gradient computation

    @return:
    embedding - the updated embeddings
    '''
    for _ in range(num_iters):
        i = np.random.randint(len(data))
        embedding[data[i]] -= learning_rate * gradient(embedding, data, i, d, k)
    return embedding
    

