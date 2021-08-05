import json
import random
import nltk
import csv
from itertools import islice
import numpy as np
import pandas as pd
nltk.download('punkt')

TOKENIZER_FILTER = (("'",""),('"',''),(',',' , '),('/',' / '),('\\',' \\ '),('.',' . '),('-',' - '),('+',''),(':',' : '))
LABEL_DIC = {'CRIME':0, 
             'ENTERTAINMENT':1, 
             'WORLD NEWS':2, 
             'IMPACT':3, 
             'POLITICS':4, 
             'WEIRD NEWS':5, 
             'BLACK VOICES':6, 
             'WOMEN':7, 
             'COMEDY':8, 
             'QUEER VOICES':9, 
             'SPORTS':10, 
             'BUSINESS':11, 
             'TRAVEL':12, 
             'MEDIA':13, 
             'TECH':14, 
             'RELIGION':15, 
             'SCIENCE':16, 
             'LATINO VOICES':17, 
             'EDUCATION':18, 
             'COLLEGE':19, 
             'PARENTS':20, 
             'ARTS & CULTURE':21, 
             'STYLE':22, 
             'GREEN':23, 
             'TASTE':24, 
             'HEALTHY LIVING':25, 
             'THE WORLDPOST':26, 
             'GOOD NEWS':27, 
             'WORLDPOST':28, 
             'FIFTY':29, 
             'ARTS':30, 
             'WELLNESS':31, 
             'PARENTING':32, 
             'HOME & LIVING':33, 
             'STYLE & BEAUTY':34, 
             'DIVORCE':35, 
             'WEDDINGS':36, 
             'FOOD & DRINK':37, 
             'MONEY':38, 
             'ENVIRONMENT':39, 
             'CULTURE & ARTS':40}
LABEL_LIST=['CRIME',
            'ENTERTAINMENT',
            'WORLD NEWS', 
            'IMPACT', 
            'POLITICS', 
            'WEIRD NEWS', 
            'BLACK VOICES', 
            'WOMEN', 
            'COMEDY', 
            'QUEER VOICES', 
            'SPORTS', 
            'BUSINESS', 
            'TRAVEL', 
            'MEDIA', 
            'TECH', 
            'RELIGION', 
            'SCIENCE', 
            'LATINO VOICES', 
            'EDUCATION', 
            'COLLEGE', 
            'PARENTS', 
            'ARTS & CULTURE', 
            'STYLE', 
            'GREEN', 
            'TASTE', 
            'HEALTHY LIVING', 
            'THE WORLDPOST', 
            'GOOD NEWS', 
             'WORLDPOST', 
             'FIFTY', 
             'ARTS', 
             'WELLNESS', 
             'PARENTING', 
             'HOME & LIVING', 
             'STYLE & BEAUTY', 
             'DIVORCE', 
             'WEDDINGS', 
             'FOOD & DRINK', 
             'MONEY', 
             'ENVIRONMENT', 
             'CULTURE & ARTS']


def print_all_class():
    i = 0
    organized_data = []
    data_row = []
    for label in LABEL_DIC:
        if i < 5:
            data_row.append(label)
            data_row.append(LABEL_DIC[label])
            i += 1
        else:
            organized_data.append(data_row)
            data_row = []
            i = 0
    od = pd.DataFrame(organized_data, head(category, id, category, id,categoryname, id, category, id, category, id))
    return od  
    
def getvector(word, model):
    if model.vocab.get(word.lower(),"NaN") == "NaN":
        lst = nltk.stem.LancasterStemmer()
        if model.vocab.get(lst.stem(word).lower(),"NaN") == "NaN":
            if word[-1] == "s" and model.vocab.get(word[:-1].lower(),"NaN") != "NaN":
                return model[word[:-1].lower()]
            else:
                return model["unk"]
        else:
            return model[lst.stem(word).lower()]
    else:
        return model[word.lower()]

def word2vector(reviews, model, max_length=1000):
    vector_data = np.zeros((len(reviews), max_length,100))
    x_length = []
    i = 0
    for review in reviews:
        j = 0 
        if len(review) > max_length:
            print("The length of the reviews is %dwhich is larger than max_length (%d)"%(len(review),max_length))
            print(review)
            print("-"*10)
        for word in review:
            vector_data[i,j] = getvector(word,model)
            if str(vector_data[i,j,0])=='nan':
                print(word)
                break
            j += 1
        x_length.append(j)
        i += 1
    return vector_data, np.asarray(x_length)

class NewsCategory:
    def __init__(self,filepath='News_Category_Dataset_v2.json', batch_size=100, 
                 max_length=500, shuffle = True):
        self.file_path = filepath
        self.batch_size = batch_size 
        self.start_pos = 0
        self.batch_index = []
        self.index = 0
        self.max_length = max_length
        self.data, self.label = self._read_file(self.file_path)
        self.max_recorder = len(self.data)
        self.index = [i for i in range(self.max_recorder)]
        if shuffle:
            random.shuffle(self.index)
        
    def _read_file(self,filename):
        data = []
        label = []
        with open(filename, "r") as f: 
            jsonData =  f.readline()
            while jsonData != "":
                jsonText = json.loads(jsonData)
                tokenized_word = self.tokenizer_clean(jsonText['headline'],TOKENIZER_FILTER)
                if len(jsonText['headline']) != 0:
                    if len(tokenized_word) <= self.max_length:
                        data.append(tokenized_word)
                    else: 
                        data.append(tokenized_word[:int(self.max_length/2)]+tokenized_word[-int(self.max_length/2):])
                    label.append(LABEL_DIC[jsonText['category']])
                jsonData =  f.readline()
        #print(len(data))
        #print(len(label))
        return data,label
    
    def tokenizer_clean(self, words, tokenizer_filter):
        for item in tokenizer_filter:
            words = words.replace(item[0],item[1])
            
        if words.find("2008.")!= -1:
            print(words.find("."))
            words.replace('.',' . ')
            print(words)
            
        tokenized_words = nltk.word_tokenize(words)
        return tokenized_words
    
    def get_batch_data(self):
        batch_data = []
        batch_label = []
        if self.start_pos + self.batch_size < len(self.index):
            for i in range(self.batch_size):
                batch_data.append(self.data[self.index[self.start_pos+i]])
                batch_label.append(self.label[self.index[self.start_pos+i]])
            self.start_pos += self.batch_size
        else:
            for i in range(len(self.index)-self.start_pos):
                batch_data.append(self.data[self.index[self.start_pos+i]])
                batch_label.append(self.label[self.index[self.start_pos+i]])
            for i in range(self.batch_size+self.start_pos-len(self.index)):
                batch_data.append(self.data[self.index[i]])
                batch_label.append(self.label[self.index[i]])
            self.start_pos = i
        return batch_data, batch_label
    
class NewsCategoryTrainTestSet:
    def __init__(self,filepath="news_cat_train_test_data.csv", batch_size=100, 
                 max_length=500, shuffle = True):
        self.file_path = filepath
        self.batch_size = batch_size 
        self.train_start_pos = 0
        self.test_start_pos = 0
        self.batch_index = []
        self.index = 0
        self.max_length = max_length
        self._read_file(self.file_path)
        self.train_size = len(self.train_label)
        self.test_size = len(self.test_label)
        self.train_index = [i for i in range(self.train_size)]
        if shuffle:
            random.shuffle(self.train_index)
    
    def _remove_quotation_mark(self, word_list):
        
        new_data = []
        #i=0
        for word in word_list:
            #if i < 2:
            #    print(word)
            #i+=1
            if len(word) < 3:
                continue
            new_data.append(word.strip()[1:-1])
        return new_data
    
    def _read_file(self,filename):
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        
        with open(filename, "r") as fo: 
            reader = csv.reader(fo)
            #i = 0
            for row in islice(reader,1,None):
                if row[2] == "train":
                    new_data = self._remove_quotation_mark(row[0].strip()[1:-1].split(","))
                    #if i < 5:
                    #    print(row[0].strip()[1:-1].split(","))
                    #    i += 1
                    self.train_data.append(new_data)
                    self.train_label.append(int(row[1]))
                elif row[2] == "test":
                    new_data = self._remove_quotation_mark(row[0].strip()[1:-1].split(","))
                    self.test_data.append(new_data)
                    self.test_label.append(int(row[1]))
        print("Total %d recorders are read. %d train data and %d test data."%(len(self.train_label)+len(self.test_label),
                                                                              len(self.train_label),
                                                                              len(self.test_label)))
    def batch_train_set(self):
        batch_data = []
        batch_label = []
        if self.train_start_pos + self.batch_size < self.train_size:
            for i in range(self.batch_size):
                batch_data.append(self.train_data[self.train_index[self.train_start_pos+i]])
                batch_label.append(self.train_label[self.train_index[self.train_start_pos+i]])
            self.train_start_pos += self.batch_size
        else:
            for i in range(self.train_size-self.train_start_pos):
                batch_data.append(self.train_data[self.train_index[self.train_start_pos+i]])
                batch_label.append(self.train_label[self.train_index[self.train_start_pos+i]])
            for i in range(self.batch_size+self.train_start_pos-self.train_size):
                batch_data.append(self.train_data[self.train_index[i]])
                batch_label.append(self.train_label[self.train_index[i]])
            self.train_start_pos = i
        return batch_data, batch_label
                
    def batch_test_set(self):
        batch_data = []
        batch_label = []
        if self.test_start_pos + self.batch_size < self.test_size:
            for i in range(self.batch_size):
                batch_data.append(self.test_data[self.test_start_pos+i])
                batch_label.append(self.test_label[self.test_start_pos+i])
            self.test_start_pos += self.batch_size
        else:
            for i in range(self.test_size-self.test_start_pos):
                batch_data.append(self.test_data[self.test_start_pos+i])
                batch_label.append(self.test_label[self.test_start_pos+i])
            for i in range(self.batch_size+self.test_start_pos-self.test_size):
                batch_data.append(self.test_data[i])
                batch_label.append(self.test_label[i])
            self.test_start_pos = i
        return batch_data, batch_label   
                