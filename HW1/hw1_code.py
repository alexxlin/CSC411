import numpy
import sklearn
import csv
import sys
import random
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree
import graphviz
import pydot
import math

# csv.field_size_limit(sys.maxsize)
def load_data():
    
    # Loading fake new headlines file
    real_file = open("data/clean_real.txt", "r")
    fake_file = open("data/clean_fake.txt", "r")
    real_titles = real_file.read().split("\n")
    fake_titles = fake_file.read().split("\n")
    
    all_titles = real_titles + fake_titles
    
    true_labels = numpy.ones(len(real_titles))
    false_labels = numpy.zeros(len(fake_titles))
                      
    all_labels = []
    all_labels.extend(true_labels)
    all_labels.extend(false_labels)
    
    
    vectorizer = CountVectorizer()
    vectorizer._validate_vocabulary()
    matrix = vectorizer.fit_transform(all_titles)
    #This gets all occuring words from the data set into a list
    features = vectorizer.get_feature_names()
    
    #Split up the data randomly into 70% of training set, 15% of validation set 
    #and 15% of testing set.    
    titles_train, titles_test, labels_train, labels_test = train_test_split(matrix, all_labels, test_size = 0.3, random_state = 13)
    titles_test, titles_validation, labels_test, labels_validation = train_test_split(titles_test, labels_test, test_size = 0.3, random_state = 23)
    
    data_set = {"training_headlines": titles_train,
                "validation_headlines": titles_validation,
                "testing_headlines": titles_test,
                "training_labels": labels_train,
                "validation_labels": labels_validation,
                "testing_labels": labels_test,
                "features": features}
    return data_set

def select_model(data_set):
    max_depth = [5, 10, 20, 50, 100]
    gini_accuracy = []
    info_gain_accuracy = []
    
    #extract data from data set
    training_headlines = data_set["training_headlines"]
    training_labels = data_set["training_labels"]
    validation_headlines = data_set["validation_headlines"]
    validation_labels = data_set["validation_labels"]
    

    for depth in max_depth:
        #Training the classifier using Gini coefficient
        clf_info_gain = DecisionTreeClassifier(criterion = "entropy", max_depth = depth)
        clf_info_gain.fit(training_headlines, training_labels)
        
        #Get predicted labels, and validate it with actual labels to compute
        #accuracy
        validation_predicted_info = clf_info_gain.predict(validation_headlines)
        info_gain_accuracy.append(calculate_accuracy(validation_predicted_info, validation_labels))
        
        #Training the classifier using Gini coefficient
        clf_gini = DecisionTreeClassifier(criterion = "gini", max_depth = depth)
        clf_gini.fit(training_headlines, training_labels)
        
        #Get predicted labels, and validate it with actual labels to compute
        #accuracy        
        validation_predicted_gini = clf_gini.predict(validation_headlines)
        gini_accuracy.append(calculate_accuracy(validation_predicted_gini, validation_labels))
        
    for i in range(len(max_depth)):
        
        #diplay accuracy using info_gain criteria
        print("The accuracy using info gain criteria and max depth of " + str(max_depth[i]) + " is " + str(info_gain_accuracy[i]))
              
        #diplay accuracy using info_gain criteria
        print("The accuracy using gini criteria and max depth of " + str(max_depth[i]) + " is " + str(gini_accuracy[i]))              

#This function gets the whole decision tree graph, and saves it as a png file
def visualization(classiflier, data_set):
    trained_classiflier = classiflier.fit(data_set["training_headlines"], data_set["training_labels"])
    tree.export_graphviz(trained_classiflier, feature_names = data_set["features"], 
                         class_names = ["false headlines", "real headlines"], 
                         out_file='textVisual.dot')                                                                        
    (graph,) = pydot.graph_from_dot_file('textVisual.dot' )
    im_name = '{}.png'.format("visualization")
    graph.write_png(im_name)
    return

def compute_information_gain(data_set, words):
    
    #extract data
    training_headlines = data_set["training_headlines"]
    training_labels = data_set["training_labels"]    
    total_words = data_set["features"]
    info_gain_dic = {}
    
    for word in words:
        #number of real headlines that contain the word
        real_positive = 0
        #number of real headlines that don't contain the word
        real_negative = 0
        #number of fake headlines that contain word
        fake_positive = 0
        #number of fake headlines that don't contain the word
        fake_negative = 0
        
        word_index = total_words.index(word)
        for i in range(len(training_labels)):
            if training_headlines[i, word_index] > 0:
                if training_labels[i] == 1:
                    real_positive += 1
                else:
                    fake_positive += 1
            else:
                if training_labels[i] == 1:
                    real_negative += 1
                else:
                    fake_negative += 1                
    
        #Calculate the entropy of whethere it is fake/real, without having
        #information about the specific word
        entropy_before = calculate_entropy(real_positive + real_negative, fake_positive + fake_negative)
        
        #Calculate the entropy of whethere it is fake/real, given we know the 
        #word is in headline
        entropy_positive = calculate_entropy(real_positive, fake_positive)
        
        #Calculate the entropy of whethere it is fake/real, given we know the word is
        #NOT in headline
        entropy_negative = calculate_entropy(real_negative, fake_negative)
        
        #Calculate the conditional entropy for fake/real headline, given information
        #about the word
        positive_probability = (real_positive + fake_positive)/len(training_labels)
        negative_probability = (real_negative + fake_negative)/len(training_labels)        
        conditional_entropy = positive_probability * entropy_positive + negative_probability * entropy_negative
        
        #Compute infomational gain due to the specific word
        info_gain = entropy_before - conditional_entropy
        info_gain_dic[word] = info_gain
        
    return info_gain_dic

################ helper functions ############################

def calculate_accuracy(actual_label, predicted_label):
    
    accurate_count = 0
    for i in range(len(actual_label)):
        if actual_label[i] == predicted_label[i]:
            accurate_count += 1
    
    return accurate_count/len(actual_label)

def calculate_entropy(count_event_one, count_event_two):
    
    total_events = count_event_one + count_event_two
    event_one_probability = count_event_one/total_events
    event_two_probability = count_event_two/total_events
    
    entropy = - event_one_probability * math.log(event_one_probability, 2.0) - event_two_probability * math.log(event_two_probability, 2.0)
    return entropy
    

#running the functions to get results for question 2
data_set = load_data()
select_model(data_set)
classifier = DecisionTreeClassifier(criterion = "entropy", max_depth = 100)
visualization(classifier, data_set)
split_words = ["the", "if", "clinton", "changed", "trade"]
info_gain = compute_information_gain(data_set, split_words)
for word in split_words:
    print("The information gain in fake/real headline due to '" + word + "' is " + str(info_gain[word]))
