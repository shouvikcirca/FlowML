from pymongo import MongoClient
import numpy as np

connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
dbClient = MongoClient(connectionString)
db = dbClient['mlflowexperiments']
collection = db['trainingdata']

data = collection.find({"comments": {'$exists': True}})

################ Load data ############
fileNames = ['BullyingCom.txt','InsultCom.txt','ObsceneCom.txt','RacistCom.txt']

comments = []
targets = []

for fName in fileNames:
	filePath = '../register_model/{}'.format(fName)
	with open(filePath, 'r') as r:
			s = r.read().split('\n')
			s = [i.split('\t') for i in s]
			s = s[:-1]
			comments.extend(s)	
	
targets = [int(item[0]) for item in comments if len(item) == 2]	
comments = [item[1] for item in comments if len(item) == 2]

for comment in comments:
	collection.update_one({"_id":data[0]['_id']},{'$push':{'comments': comment}})
#for target in targets:
#	collection.update_one({"_id":data[0]['_id']},{'$push':{'targets': target}})

commentsL = collection.find({"comments": {'$exists': True}})[0]['comments']
targetsL = collection.find({"comments": {'$exists': True}})[0]['targets']

#print(len(commentsL), len(targetsL))

npTargets = np.array(targetsL)
print((npTargets == 1).sum()/len(targetsL),
		(npTargets == 2).sum()/len(targetsL),
		(npTargets == 3).sum()/len(targetsL),
		(npTargets == 4).sum()/len(targetsL)
		)
################################################

##### Reset data ##################################

collection.update_one({"_id":data[0]['_id']},{'$set':{'comments': []}})
collection.update_one({"_id":data[0]['_id']},{'$set':{'targets': []}})

#################################################
