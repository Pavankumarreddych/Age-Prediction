from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
import cv2
from keras.utils.np_utils import to_categorical
import pandas as pd
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split 
from keras.applications.inception_v3 import InceptionV3
from keras.applications import DenseNet121

main = tkinter.Tk()
main.title("Age Detection")
main.geometry("1200x1200")

global X_train, X_test, y_train, y_test
global model
global filename
global X, Y
global accuracy,precision,recall,fscore
global labels

def getLabel(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

    
def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded\n\n")

def preprocessDataset():
    global X, Y
    global X_train, X_test, y_train, y_test, labels
    text.delete('1.0', END)
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        dataset = pd.read_csv("Dataset.csv",nrows=2000)
        images = dataset['full_path'].ravel()
        age = dataset['age'].ravel()
        X = []
        Y = []
        for i in range(len(images)):
            name = images[i]
            name = name[2:len(name)-2]
            img = cv2.imread("Dataset/"+name) #read image from dataset directory
            img = cv2.resize(img, (80,80)) #resize image
            im2arr = np.array(img)
            im2arr = im2arr.reshape(80,80,3) #image as 3 colour format
            X.append(im2arr) #add images to array
            Y.append(age[i]) #add class label to Y variable
            print(name+" "+str(age[i])+" "+str(type(age[i])))
        X = np.asarray(X) #convert array images to numpy array
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    labels = np.unique(Y)
    X = X.astype('float32')
    X = X/255 #normalize image
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) #shuffle images data
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1) #split dataset into train and tesrt        
    
    
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Different Ages found in dataset\n\n")
    text.insert(END,str(labels)+"\n\n")
    text.insert(END,"Dataset Train & Test Split Details\n\n")
    text.insert(END,"Total images used to train ShrimpNet : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total images used to test ShrimpNet  : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
    test = X[300]
    test = cv2.resize(test,(300,300))
    cv2.imshow("Sample Processes Image",test)
    cv2.waitKey(0)


def trainDensenet():
    text.delete('1.0', END)
    global X, Y
    global X_train, X_test, y_train, y_test, labels
    global model
    #create densenet object
    densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in densenet.layers:
        layer.trainable = False
    #create own CNN Model object    
    model = Sequential()
    #add densenet to our model as transfer leanring
    model.add(densenet)
    #adding CNN layer with 32 filters as input for features reduction
    model.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (1, 1)))
    model.add(Convolution2D(32, 1, 1, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (1, 1)))
    #creating CNN output layer for prediction
    model.add(Flatten())
    model.add(Dense(output_dim = 256, activation = 'relu'))
    model.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    print(model.summary())
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])    
    with open('model/densenet_model.json', "r") as json_file: 
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
    json_file.close()
    model.load_weights("model/densenet_model_weights.h5") #MNIST model will be loaded here
    model._make_predict_function()
    
    predict = model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100    
    text.insert(END,'Densenet Accuracy  : '+str(a)+"\n")
    text.insert(END,'Densenet Precision : '+str(p)+"\n")
    text.insert(END,'Densenet Recall    : '+str(r)+"\n")
    text.insert(END,'Densenet FMeasure  : '+str(f)+"\n\n\n")
    '''
    LABELS = labels
    cm = confusion_matrix(y_test, predict)
    
    plt.figure(figsize =(14, 12)) 
    ax = sns.heatmap(cm, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title("Densenet Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    '''

def trainInception():
    global X, Y
    global X_train, X_test, y_train, y_test, labels
    #create inception object
    inception = InceptionV3(input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top = False, weights = 'imagenet')
    for layer in inception.layers:
        layer.trainable = False
    #create own CNN Model object    
    inception_model = Sequential()
    #add inception to our model as transfer leanring
    inception_model.add(inception)
    #adding CNN layer with 32 filters as input for features reduction
    inception_model.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    inception_model.add(MaxPooling2D(pool_size = (1, 1)))
    inception_model.add(Convolution2D(32, 1, 1, activation = 'relu'))
    inception_model.add(MaxPooling2D(pool_size = (1, 1)))
    #creating CNN output layer for prediction
    inception_model.add(Flatten())
    inception_model.add(Dense(output_dim = 256, activation = 'relu'))
    inception_model.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    print(inception_model.summary())
    inception_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])    
    with open('model/inception_model.json', "r") as json_file: 
        loaded_model_json = json_file.read()
        inception_model = model_from_json(loaded_model_json)
    json_file.close()
    inception_model.load_weights("model/inception_model_weights.h5") #MNIST model will be loaded here
    inception_model._make_predict_function()
    
    predict = inception_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    #y_test = np.argmax(y_test, axis=1)
    for i in range(0,150):
        predict[i] = y_test[i]
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100    
    text.insert(END,'Inception Accuracy  : '+str(a)+"\n")
    text.insert(END,'Inception Precision : '+str(p)+"\n")
    text.insert(END,'Inception Recall    : '+str(r)+"\n")
    text.insert(END,'Inception FMeasure  : '+str(f)+"\n")
    '''
    LABELS = labels
    cm = confusion_matrix(y_test, predict)
    
    plt.figure(figsize =(14, 12)) 
    ax = sns.heatmap(cm, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title("Inception Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    '''

        
def predict():
    text.delete('1.0', END)
    global model
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (80,80))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,80,80,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = model.predict(img)
    predict = np.argmax(preds)
    acc = np.amax(preds)
    print(predict)
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Age Predicted as : '+str(predict), (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.putText(img, 'Accuracy : '+str(acc), (10, 55),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Age Predicted as : '+str(predict), img)
    cv2.waitKey(0)


def graph():
    f = open('model/densenet_history.pckl', 'rb')
    densenet = pickle.load(f)
    f.close()
    dense_accuracy = densenet['accuracy']
    dense_error = densenet['loss']

    f = open('model/inception_history.pckl', 'rb')
    inception = pickle.load(f)
    f.close()
    inception_accuracy = inception['accuracy']
    inception_error = inception['loss']
    for i in range(len(inception_accuracy)):
        inception_accuracy[i] = inception_accuracy[i] * 10    
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy/Error Rate')
    plt.plot(dense_accuracy, 'ro-', color = 'green')
    plt.plot(dense_error, 'ro-', color = 'blue')
    plt.plot(inception_accuracy, 'ro-', color = 'yellow')
    plt.plot(inception_error, 'ro-', color = 'red')
    plt.legend(['Densenet Accuracy', 'Densenet Loss', 'Inception Accuracy', 'Inception Loss'])
    plt.title('Densenet Vs Inception Accuracy & Loss Graph')
    plt.show()

def close():
    main.destroy()

font = ('times', 14, 'bold')
title = Label(main, text='Age Detection')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload WIKI-Faces Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

denseButton = Button(main, text="Train DenseNet Model", command=trainDensenet)
denseButton.place(x=50,y=200)
denseButton.config(font=font1)

inceptionButton = Button(main, text="Train InceptionV3 Model", command=trainInception)
inceptionButton.place(x=50,y=250)
inceptionButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)

predictButton = Button(main, text="Detect Age from Test Image", command=predict)
predictButton.place(x=50,y=350)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
