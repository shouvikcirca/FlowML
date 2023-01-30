from tkinter import *
import numpy as np

root = Tk()
verdicts = []

def predictClick():
	listboxResults.delete(0, END)
	enteredText = myLabel12.get()

	probs = np.random.randint(10, size=(4))
	obscene = probs[0]/probs.sum()
	insulting = probs[1]/probs.sum()
	hateful = probs[2]/probs.sum()
	bullying = probs[3]/probs.sum()

	if obscene>0.5 or insulting>0.5 or hateful>0.5 or bullying>0.5:
		enteredText = 'Disturbing Remark'

	#verdicts.append(enteredText)
	#renderedVerdicts = '\n'.join(verdicts)

	
	listboxResults.insert(END,'Obscene:{0:.2f}    '.format(obscene) )
	listboxResults.insert(END,'Insulting:{0:.2f}    '.format(insulting) )
	listboxResults.insert(END,'Hateful:{0:.2f}    '.format(hateful) )
	listboxResults.insert(END,'Bullying:{0:.2f}    '.format(bullying) )
	
	myListbox32.insert(END,enteredText)
	
	#myLabel30 = Label(root, text='Obscene:{0:.2f}    '.format(obscene))
	#myLabel31 = Label(root, text='Insulting:{0:.2f}    '.format(insulting))
	
	#myListbox32.insert(END, enteredText)
	#myLabel33 = Label(root, text='Hateful:{0:.2f}    '.format(hateful))
	#myLabel34 = Label(root, text='Bullying:{0:.2f}    '.format(bullying))
	
	#myLabel30.grid(row = 3, column = 0)
	#myLabel31.grid(row = 3, column = 1)
	#myLabel33.grid(row = 3, column = 3)
	#myLabel34.grid(row = 3, column = 4)

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

#myLabel.pack()
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

listboxResults.pack(side=LEFT)#grid(row = 0, column = 0)
myListbox32.pack(side=RIGHT)#grid(row = 0, column = 1)

root.mainloop()
