import pickle
import os
from tkinter import *
import numpy as np
from pymongo import MongoClient
import mlflow
from mlflow.tracking.client import MlflowClient
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import requests

root = Tk()

connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
dbClient = MongoClient(connectionString)
db = dbClient['mlflowexperiments']

def sampleForActiveLearning(sample):
	print('Sampled for AL:{}'.format(sample))

def getPredictions(enteredText):
	experimentName = 'Alpha'
	client = MlflowClient()

	collection = db['deployedModels']
	runid = collection.find({'task':'cyberbullying_classification'})[0]['cyberbullying_classification']
	
	model_name = '{}_{}'.format(experimentName, runid)
	stage='Staging'
	modeluri = 'models:/{}/{}'.format(model_name, stage)
	
	local_dir = './tmp'
	if not os.path.exists(local_dir):
		os.mkdir(local_dir)
	tokenizer_path = client.download_artifacts(runid, "tokenizer.pkl", local_dir)
	tokenizer = pickle.load(open(tokenizer_path,'rb'))

	comments = np.array([enteredText])
	run = client.get_run(runid)
	testSequences = tokenizer.texts_to_sequences(comments)
	paddedTestSequences = pad_sequences(testSequences,padding = run.data.params['seq_padding_style'],truncating = run.data.params['seq_truncating_style'],maxlen = int(run.data.params['max_length']))

	prediction = requests.post(url='http://127.0.0.1:5004/invocations',json={"inputs":paddedTestSequences.tolist()}, headers={'Content-Type':'application/json',})
	p = np.array(prediction.json())

	# p[0] contains the probability scores
	return p[0]

def predictClick():
	listboxResults.delete(0, END)
	enteredText = myLabel12.get()

	obscene, insulting, hateful, bullying = preds = getPredictions(enteredText)

	# If all the probability scores are below 0.8, sample is chosen for Active Learning
	if not (preds>=0.8).sum():
		sampleForActiveLearning(enteredText)

	if obscene>0.5 or insulting>0.5 or hateful>0.5 or bullying>0.5:
		enteredText = 'Disturbing Remark'

	listboxResults.insert(END,'Obscene:{0:.2f}    '.format(obscene) )
	listboxResults.insert(END,'Insulting:{0:.2f}    '.format(insulting) )
	listboxResults.insert(END,'Hateful:{0:.2f}    '.format(hateful) )
	listboxResults.insert(END,'Bullying:{0:.2f}    '.format(bullying) )
	
	myListbox32.insert(END,enteredText)


myLabel00 = Label(root, text="                    ")
myLabel01 = Label(root, text="                    ")
myLabel02 = Label(root, text="Enter your text here")
myLabel03 = Label(root, text="                    ")
myLabel04 = Label(root, text="                    ")

myLabel10 = Label(root, text="                    ")
myLabel11 = Label(root, text="                    ")
myLabel12 = Entry(root, width = 50)
myLabel13 = Label(root, text="                    ")
myLabel14 = Label(root, text="                    ")

myLabel20 = Label(root, text="                    ")
myLabel21 = Label(root, text="                    ")
myLabel22 = Button(root, text = "Predict", command = predictClick)
myLabel23 = Label(root, text="                    ")
myLabel24 = Label(root, text="                    ")

myLabel00.grid(row = 0, column = 0)
myLabel01.grid(row = 0, column = 1)
myLabel02.grid(row = 0, column = 2)
myLabel03.grid(row = 0, column = 3)
myLabel04.grid(row = 0, column = 4)
myLabel10.grid(row = 1, column = 0)
myLabel11.grid(row = 1, column = 1)
myLabel12.grid(row = 1, column = 2)
myLabel13.grid(row = 1, column = 3)
myLabel14.grid(row = 1, column = 4)
myLabel20.grid(row = 2, column = 0)
myLabel21.grid(row = 2, column = 1)
myLabel22.grid(row = 2, column = 2)
myLabel23.grid(row = 2, column = 3)
myLabel24.grid(row = 2, column = 4)

myFrame32 = Frame(root)
my_scrollbar = Scrollbar(myFrame32, orient=VERTICAL)
myListbox32 = Listbox(myFrame32, width=25, yscrollcommand=my_scrollbar.set)
my_scrollbar.config(command = myListbox32.yview)
my_scrollbar.pack(side=RIGHT, fill=Y)
listboxResults = Listbox(myFrame32, width = 25)
myFrame32.grid(row=3, column=2, pady = (0,10))

listboxResults.pack(side=LEFT)
myListbox32.pack(side=RIGHT)

root.mainloop()
