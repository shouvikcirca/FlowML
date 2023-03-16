from datasets import Dataset, Features, Value
from pymongo import MongoClient

connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
dbClient = MongoClient(connectionString)
db = dbClient['mlflowexperiments']
collection = db['trainingdata']

dbData = collection.find({"comments": {'$exists': True}})[0]
data = {"comments": dbData['comments'], "targets": dbData['targets'] }

features = Features({'comments': Value('string'),'targets': Value('int32')})
dataset = Dataset.from_dict(data, features = features)

print(dataset[0:12])




