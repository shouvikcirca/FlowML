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
from tkinter import messagebox
from functools import partial

root = Tk()

connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
dbClient = MongoClient(connectionString)
db = dbClient['mlflowexperiments']

def getClickValues():
	ln = len(newSamples)
	targets = []
	comments = []
	for k in range(ln):
		comments.append(newSamples[k])
		targets.append(int(radioButtonIndices[k].get()))
	print(targets)
	print(comments)
	
	collection = db['trainingdata']
	preInsertionData = collection.find({"comments": {'$exists': True}})[0]
	preInsertionCommentData = preInsertionData["comments"]
	preInsertionTargetData = preInsertionData["targets"]

	for i in range(ln):
		collection.update_one({"_id":preInsertionData['_id']},{'$push':{'comments': comments[i]}})
		collection.update_one({"_id":preInsertionData['_id']},{'$push':{'targets': targets[i]}})
	
	postInsertionData = collection.find({"comments": {'$exists': True}})[0]
	postInsertionCommentData = postInsertionData["comments"]
	postInsertionTargetData = postInsertionData["targets"]

	commentsInserted = 0
	targetsInserted = 0
	if len(postInsertionCommentData) == len(preInsertionCommentData) + len(comments):
		print("New Comment data inserted")
		commentsInserted = 1
	else:
		print("Error inserting new Comment data")
	
	if len(postInsertionTargetData) == len(preInsertionTargetData) + len(targets):
		print("New Target data inserted")
		targetsInserted = 1
	else:
		print("Error inserting new Target data")

	if commentsInserted and targetsInserted:
		collection = db['activelearning']
		data = collection.find({"data": {'$exists': True}})
		collection.update_one({"_id":data[0]['_id']},{'$set':{'data': []}})
		print("No Active Learning samples remaining")




def getUnlabelledSamples():
	collection = db['activelearning']
	data = collection.find({"data": {'$exists': True}})
	print('Fetched existing Active Learning samples')
	
	existingEntries = data[0]['data']
	return existingEntries

def insertRowInGui(ind):
	newFrame = Frame(myFrame02)
	newFrame.pack()

	lb = Listbox(newFrame, width=50, height = 2)
	lb.insert(END,' '+newSamples[i])
	lb.pack()

	rb = Radiobutton(newFrame, text = 'Non-Derogatory', var = radioButtonIndices[ind], value = 0)
	rb.pack(side = RIGHT)

	rb = Radiobutton(newFrame, text = 'Bullying', var = radioButtonIndices[ind], value = 1)
	rb.pack(side = RIGHT)
	
	rb = Radiobutton(newFrame, text = 'Obscene', var = radioButtonIndices[ind], value = 2)
	rb.pack(side = RIGHT)
	
	rb = Radiobutton(newFrame, text = 'Insulting', var = radioButtonIndices[ind], value = 3)
	rb.pack(side = RIGHT)
	
	rb = Radiobutton(newFrame, text = 'Racist', var = radioButtonIndices[ind], value = 4)
	rb.pack(side = RIGHT)

if __name__ == '__main__':

	newSamples = getUnlabelledSamples()
	if len(newSamples):
		myLabel00 = Label(root, text="                    ")
		myLabel01 = Label(root, text="                    ")
		myLabel03 = Label(root, text="                    ")
		myLabel04 = Label(root, text="                    ")

		myLabel00.grid(row = 0, column = 0)
		myLabel01.grid(row = 0, column = 1)
		myLabel03.grid(row = 0, column = 3)
		myLabel04.grid(row = 0, column = 4)
		
		myFrame02 = Frame(root, height = 20)#grid(row = 3, column = 2)
		myFrame02.grid(row = 0, column = 2, sticky = 'nsew',pady = (10,10))
		
		my_scrollbar = Scrollbar(myFrame02, orient=VERTICAL)
		my_scrollbar.pack(side=RIGHT, fill=Y)
		
		radioButtonIndices = {}
		for i in range(len(newSamples)):
			radioButtonIndices[i] = StringVar()
			insertRowInGui(i)

		submitButton = Button(root, text = "Submit", command = getClickValues)
		submitButton.grid(row = 1, column = 2, pady = (5,5))

	else:
		myLabel00 = Label(root, text="                    ")
		myLabel01 = Label(root, text="                    ")
		myLabel02 = Label(root, text="No new samples")
		myLabel03 = Label(root, text="                    ")
		myLabel04 = Label(root, text="                    ")

		myLabel00.grid(row = 0, column = 0)
		myLabel01.grid(row = 0, column = 1)
		myLabel02.grid(row = 0, column = 2)
		myLabel03.grid(row = 0, column = 3)
		myLabel04.grid(row = 0, column = 4)


root.mainloop()
