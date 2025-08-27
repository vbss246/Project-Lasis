from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

main = tkinter.Tk()
main.title("CLASSIFICATION OF FRACTURED BONES USING MACHINE LEARNING")
main.geometry("1300x1200")

global filename
global X, Y
global X_train, X_test, y_train, y_test
labels = ['Chest', 'Elbow', 'Finger', 'Hand', 'Head', 'Shoulder', 'Wrist']
global classifier

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir = ".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+' dataset loaded\n')
        
def featuresExtraction():
    text.delete('1.0', END)
    global X, Y
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    text.insert(END,"Available Bones images in dataset is : "+str(labels)+"\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total available features in each image : "+str(X.shape[1])+"\n")
    test = X[3]
    test = test.reshape(32,32,3)
    test = cv2.resize(test,(250,250))
    cv2.imshow("Sample Image from dataset after features extraction",test)
    cv2.waitKey(0)

def trainTestGenerator():
    text.delete('1.0', END)
    global X, Y
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total images used to train Random Forest : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total images used to test Random Forest : "+str(X_test.shape[0])+"\n")

def randomForest():
    text.delete('1.0', END)
    global X, Y
    global X_train, X_test, y_train, y_test
    global classifier
    if os.path.exists('model/model.txt'):
        with open('model/model.txt', 'rb') as file:
            classifier = pickle.load(file)
        file.close()
    else:
        classifier = RandomForestClassifier(n_estimators=200, random_state=0)
        classifier.fit(X_train, y_train)
        with open('model/model.txt', 'wb') as file:
            pickle.dump(classifier, file)
        file.close()
    predict = classifier.predict(X_test)
    random_acc = accuracy_score(y_test,predict)*100
    text.insert(END,"Random Forest Bone Classification Accuracy on Test Images : "+str(random_acc)+"\n\n")
    text.insert(END,"Random Forest Bone Classification Report on Test Images\n\n")
    text.insert(END,str(classification_report(y_test, predict))+"\n")


def predict():
    global classifier
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (32,32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(32,32,3)
    test = np.asarray(im2arr)
    test = test.astype('float32')
    test = test/255
    test = test.ravel()
    imgData = []
    imgData.append(test)
    test = np.asarray(imgData)
    print(test.shape)
    predict = classifier.predict(test)[0]
    print(predict)

    img = cv2.imread(filename)
    img = cv2.resize(img, (400,400))
    cv2.putText(img, 'Bone Classified as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow('Bone Classified as : '+labels[predict], img)
    cv2.waitKey(0)
    


font = ('times', 16, 'bold')
title = Label(main, text='CLASSIFICATION OF FRACTURED BONES USING MACHINE LEARNING')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Dataset Collection or Upload", command=uploadDataset)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='lawn green', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

featuresButton = Button(main, text="Features Extraction", command=featuresExtraction)
featuresButton.place(x=700,y=200)
featuresButton.config(font=font1)

traintestButton = Button(main, text="Train & Test Data Generator", command=trainTestGenerator)
traintestButton.place(x=700,y=250)
traintestButton.config(font=font1) 

rfButton = Button(main, text="Build Random Forest Model", command=randomForest)
rfButton.place(x=700,y=300)
rfButton.config(font=font1)

predictButton = Button(main, text="Upload Test Image & Classify Bone", command=predict)
predictButton.place(x=700,y=350)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='RoyalBlue2')
main.mainloop()
