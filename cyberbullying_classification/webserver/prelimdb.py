import pymongo
from pymongo import MongoClient


connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
client = MongoClient(connectionString)

db = client['mlflowexperiments']
collection = db["taskhyperparams"]

hyperparams = {
        'name':'Text Classification', 
        'hp':[
            ['learning_rate', 'float'],
            ['vocab_length','int'],
            ['seq_padding_style','string',['post', 'pre']],
            ['embedding_dim','int'],
            ['bs','int'],
            ['epochs','int'],
        ]
    }

collection.insert_one(hyperparams)
