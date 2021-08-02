import json
import random
import nltk
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
    
    
                