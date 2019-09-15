from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
from nltk import word_tokenize
import os

def read_stop_words(filename):
    fobj = open(filename)
    stopwords = fobj.read() # return one string
    stopwords = stopwords.split('\n')
    return stopwords

def tokenize_file(fname, stop_words):
    fobj = open(fname, encoding="ISO-8859-1")
    try:
        content = fobj.read()
    except Exception as e:
        print("Length of content: ", len(content))
        print(e)
    stemmer = PorterStemmer()
    soup = BeautifulSoup(content, "html.parser")    
    body_tag = soup.find('body')
    if body_tag:
        body_text = body_tag.text
        token_list = word_tokenize(body_text)
        token_list = [ token for token in token_list if token.isalpha() and token.isascii()]    # remove speciel characters
        token_list = [ token.lower() for token in token_list]
        token_list = [ token for token in token_list if token not in stop_words ] # list comprehension
        
        final_tokens = []   # final_tokens will act as keys to the output dictionary, which is a mapping of tokens to a list of their positions
        for token in token_list:
            stemmed_token = stemmer.stem(token)
            final_tokens.append(stemmed_token)

        positions = {}  # postions = {term1: [], term2: []}
        for pos, tok in enumerate(final_tokens):
            if tok not in positions.keys():
                positions[tok] = [pos]
            else:
                positions[tok].append(pos)
        return final_tokens, positions 
    else:
        print("no text found in document")
        fobj.close()
        return [], {}

def assign_doc_ids(filenames):
    doc_ids = {}    # dictionary
    # doc_id = {"file1":0, ..}
    for ind, fname in enumerate(filenames):
        doc_ids[fname] = ind + 1
    return doc_ids

def write_doc_ids(doc_ids):
    os.chdir("..")
    fobj = open("docids.txt", "w")
    entry = ""
    for fname in doc_ids.keys():
##        fobj.write(id)
##        fobj.write("\t")
##        fobj.write(doc_ids[id])
##        fobj.write("\n")
        entry = entry + str(doc_ids[fname])
        entry = entry + "\t"
        entry = entry + fname
        entry = entry + "\n"
    fobj.write(entry)
    fobj.close()
    os.chdir("corpus")

def write_term_ids(term_ids):
    os.chdir("..")
    fobj = open("termids.txt", "w")
    entry = ""
    for tok in term_ids.keys():
        entry = entry + str(term_ids[tok]) + "\t" + tok + "\n"
    fobj.write(entry)
    fobj.close()


def write_index(term_index):
    fobj = open("term_index.txt", "w")
    entry = ""
    for term_id in term_index.keys():
        entry += str(term_id)
        entry += " " + str(term_index[term_id]['tf'])
        entry += " " + str(term_index[term_id]['df']) + " "

        doc_ids = term_index[term_id]['doc_ids']
        for doc_id in doc_ids.keys():
            
            for pos in doc_ids[doc_id]:
                entry += str(doc_id) + ","
                entry += str(pos) + " "
        entry += "\n"
    fobj.write(entry)
    fobj.close()


def make_index_without_hash(term_index):
    simple_index = []
    for term_id in term_index.keys():
        temp_row = []
        temp_row.append(term_id)
        doc_ids = term_index[term_id]['doc_ids'].keys()
        temp_row.extend(doc_ids)
        simple_index.append(temp_row)
    return simple_index
        

def write_simple_index(simple_index):
    fobj = open("simple_index.txt", "w")
    entry = ""
    for row in simple_index:
        for n in row:
            entry += str(n) + " "
        entry += "\n"
    fobj.write(entry)
    fobj.close()
    print("good bye")

############### MAIN #################
stop_words = read_stop_words("stoplist.txt")

dir_name = "corpus"     # relative path
os.chdir(dir_name)
file_names = os.listdir()[:5]
doc_ids = assign_doc_ids(file_names)
write_doc_ids(doc_ids)

term_ids = {} ## term_ids = {word1:id, word2:id}
term_id = 0    # unique
term_index = {} # term_index = {term_id: {'tf':tf, 'df':df', 'doc_ids':{id1:[pos] id2:[pos]} }, ..


for i, fname in enumerate(file_names):
    print("Working with ", fname)
    current_doc_id = doc_ids[fname]
    tokens, positions = tokenize_file(fname, stop_words)    # positions is a dictionary with keys as tokens and values as position lists

    
    
    for unique_tok in positions.keys():
        # make term ids
        if unique_tok not in term_ids.keys():
            term_ids[unique_tok] = term_id
            term_id += 1
        current_term_id = term_ids[unique_tok]
        # make term_index
        if current_term_id not in term_index.keys():
            term_index[current_term_id] = {}
            term_index[current_term_id]['tf'] = len(positions[unique_tok])
            term_index[current_term_id]['df'] = 1 
            term_index[current_term_id]['doc_ids'] = {}
            term_index[current_term_id]['doc_ids'][current_doc_id] = positions[unique_tok]
            
        else:   # this else is for iterations of outer loop
            term_index[current_term_id]['tf'] += len(positions[unique_tok])
            term_index[current_term_id]['df'] += 1
            term_index[current_term_id]['doc_ids'][current_doc_id] = positions[unique_tok]
            
    
write_term_ids(term_ids)
write_index(term_index)
simple_index = make_index_without_hash(term_index)
print(term_index)
write_simple_index(simple_index)
