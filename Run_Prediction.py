import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import nltk
import csv

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('C:/Users/Wangdi/Documents/Notebook/GloVe/glove.6B.100d.txt')
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)
model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

from NewsCategoryData import NewsCategory

def getvector(word, model=model):
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

def word2vector(reviews, model=model, max_length=1000):
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
# Run cnn model

def run_cnn_model(ckpt_dir, model_file):
    state_size = 64
    max_length = 100
    batch_size = 64

    #MODEL_FILE = 'news_category-100000.meta'
    #ckpt_dir = './ckpt_save/'

    data = NewsCategory(batch_size=batch_size,max_length=max_length,shuffle=False)
    prediction_label = []
    max_recorder = len(data.data)

    with tf.Session() as sess:
        graph = tf.get_default_graph()
        saver = tf.train.import_meta_graph(ckpt_dir + model_file)
        model_file=tf.train.latest_checkpoint(ckpt_dir)
        saver.restore(sess,model_file)

        x = graph.get_tensor_by_name("Input/x:0")
        y_ = graph.get_tensor_by_name("Input/y_:0")
        x_length = graph.get_tensor_by_name("Input/x_length:0")
        y = graph.get_tensor_by_name("Output/y:0")
        prediction = graph.get_tensor_by_name("Prediction/prediction:0")
        #cal_accuracy = graph.get_tensor_by_name("Prediction/cal_accuracy:0")
        i = 0
        avg_accuracy = 0

        while i < int(max_recorder/batch_size)+1:
            batch_data, batch_label = data.get_batch_data()  
            batch_data_vec, data_length = word2vector(batch_data, model,max_length)
            logits, pred_cate = sess.run([y, prediction],feed_dict={x:batch_data_vec, y_:batch_label, x_length:data_length})
            for j in range(len(pred_cate)):
                prediction_label.append(pred_cate[j])
            i += 1
        return prediction_label[:max_recorder]
    
def one_cnn_model(title,ckpt_dir, model_file):
    graph = tf.Graph()
    max_length = 100
    with graph.as_default():
        saver = tf.train.import_meta_graph(ckpt_dir + model_file)
        print("Load graph.")
        x = graph.get_tensor_by_name("Input/x:0")
        x_length = graph.get_tensor_by_name("Input/x_length:0")
        y = graph.get_tensor_by_name("Output/y:0")
        with tf.Session() as sess:
            model_file=tf.train.latest_checkpoint(ckpt_dir)
            print("Load model.")
            saver.restore(sess,model_file)
            batch_data_vec, data_length = word2vector([title], model, max_length)
            logits= sess.run([y],feed_dict={x:batch_data_vec, x_length:data_length})
    return logits[0]

#Run naive bayes model    
def read_naive_bayes_word_vector(file_name):
    print("Load the naive bayes word vectors. It takes a few minutes.")
    fo = open(file_name, "r+", encoding='utf-8')
    reader = csv.reader(fo)
    word_matrix = []
    word_list = {}
    index = 0
    for row in reader:
        word_list[row[0]] = index
        index += 1
        vector = []
        for data in row[1:]:
            vector.append(float(data))
        word_matrix.append(vector)
    return np.array(word_matrix),word_list

def naive_bayes_model(title_list,word_matrix,word_list): 
    prediction_list = []
    for title in title_list:
        prediction = one_naive_bayes_model(title,word_matrix,word_list)
        prediction_list.append(prediction)
    return np.array(prediction_list)

def one_naive_bayes_model(title,word_matrix,word_list): 
    x = [] 
    for word in title:
        if word.lower() in word_list:
            x.append(word_matrix[word_list[word.lower()]])
        else:
            x.append(np.array([0.0001 for i in range(41)]))
    x = np.asarray(x)
    x1 = x.prod(axis=0)
    x2 = x1.sum(keepdims=True) 
    #print('x',x)
    #print('x1', x1)
    #print('x2', x2)
    predict = x1/x2
    #print('predict', predict)
    return predict

def run_naive_bayes_model(word_matrix_file):
    data = NewsCategory()
    
    max_recorder = data.max_recorder
    word_matrix, word_list = read_naive_bayes_word_vector(word_matrix_file)
    
    prediction_list = []
    for title in data.data:
        x = [] 
        for word in title:
            x.append(word_matrix[word_list[word.lower()]])
        predict = np.asarray(x).prod(axis=0).argmax() 
        prediction_list.append(predict)
    return prediction_list

#Run combined model
def one_combined_model(cnn_logits, naive_bayes_input, ckpt_dir, graph_file):
    graph = tf.Graph()
    with graph.as_default(): 
        saver = tf.train.import_meta_graph(ckpt_dir + graph_file)
        x_logits = graph.get_tensor_by_name("Combined_Input/x_logits:0")
        x_naive_bayes = graph.get_tensor_by_name("Combined_Input/x_naive_bayes:0")
        prediction = graph.get_tensor_by_name("Combined_Prediction/prediction:0")
        with tf.Session() as sess:
            model_file=tf.train.latest_checkpoint(ckpt_dir)
            print(model_file)
            saver.restore(sess, model_file)   
            print('Restore model.')
            prediction= sess.run([prediction],
                                feed_dict={x_logits:cnn_logits,
                                           x_naive_bayes:[naive_bayes_input]})
            return prediction