import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory): #add more comments to the entire code
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """
    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    doc_words = [] # list for words in filepath
    rel_data = [] # list for relevant words w/ cutoff

    with open(filepath, 'r') as doc:
        for word in doc:
            word = word.strip()
            doc_words.append(word) # adds words from doc. to above list

    for i in doc_words:
        if i in vocab:
            rel_data.append(i) # if word is in vocab add to rel_data list
        if i not in vocab:
            rel_data.append(None) # if word is not in vocab add to rel_data list as None

    for i in rel_data:
        bow[i] = rel_data.count(i) # counts the # of appearances of each word in rel_data list

    return bow

#Needs modifications
def prior(training_data, label_list): #fix this method
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """
    smooth = 1 # smoothing factor
    logprob = {}

    c0 = 0 # counter for label_list[0]
    c1 = 0 # counter for label_list[1]

    for x in training_data:
        if x['label'] == label_list[0]: # if label in training_table == first label_list item
            c0 += 1 # increment label_list[0] counter
        if x['label'] == label_list[1]: # if label in training_table == second label_list item
            c1 += 1 # increment label_list[1] counter
    
    p0 = (c0 + smooth) / (c0 + c1 + smooth + smooth) # calculates P(label) for label_list[0]
    logprob[label_list[0]] = math.log(p0) # calculates log P(label)
    p1 = (c1 + smooth) / (c0 + c1 + smooth + smooth) # calculates P(label) for label_list[1]
    logprob[label_list[1]] = math.log(p1) # calculates log P(label)

    return logprob

#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """
    smooth = 1 # smoothing factor
    word_prob = {}

    length = len(vocab) # |V| - length of the list vocab

    wc = 0 # word count counter

    for i in training_data: # calculates word count for a particular label (i.e. year)
        if i['label'] == label:
            for x in i['bow']:
                 wc += i['bow'][x]

    v = {}
    for x in vocab: # creates a dictionary for all vocab. keys and init. all values to 0
        v[x] = 0

    v[None] = 0 # creates a None key and init. its value to 0

    for x in v: # c(word) - counts the # of times a word appears in a label
        for i in training_data:
            if i['label'] == label:
                if x in i['bow']:
                    v[x] += i['bow'][x]

    for x in v: # calculates log P(word | label)
        v[x] = math.log((v[x] + smooth) / (wc + length + smooth))

    word_prob = v

    return word_prob


##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    
    vocab = create_vocabulary(training_directory, cutoff) # creates vocab
    training_data = load_training_data(vocab, training_directory) # creates training_data
    label_list = ['2020', '2016']

    # creates train for a given training_directory & cutoff
    retval['vocabulary'] = vocab
    retval['log prior'] = prior(training_data, label_list)
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, training_data, label_list[0])
    retval['log p(w|y=2016)'] = p_word_given_label(vocab, training_data, label_list[1])

    return retval

#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}

    vocab = model['vocabulary'] # gets vocab from parameter train
    bow = create_bow(vocab, filepath) # creates bow (i.e. bag of words)

    p_2020 = 0
    p_2016 = 0

    for x in bow: # finds all instances of a particular word & adds their prob. together (2020)
        p_2020 += (model['log p(w|y=2020)'][x]) * bow[x]
 
    for x in bow: # finds all instances of a particular word & adds their prob. together (2016)
        p_2016 += (model['log p(w|y=2016)'][x]) * bow[x]
 
    p_2020 += model['log prior']['2020'] # adds log prior to the prob. found above (2020)
    p_2016 += model['log prior']['2016'] # adds log prior to the prob. found above (2016)

    # finds the max. of the two prob.
    if p_2020 > p_2016:
        m = '2020'
    if p_2016 > p_2020:
        m = '2016'

    # adds relevant information into retval dictionary
    retval['predicted y'] = m
    retval['log p(y=2016|x)'] = p_2016
    retval['log p(y=2020|x)'] = p_2020

    return retval