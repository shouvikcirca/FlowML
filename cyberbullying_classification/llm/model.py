from datasets import Dataset, Features, Value
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tokenizers import Tokenizer

connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
dbClient = MongoClient(connectionString)
db = dbClient['mlflowexperiments']
collection = db['trainingdata']

dbData = collection.find({"comments": {'$exists': True}})[0]
data = {"comments": dbData['comments'], "targets": dbData['targets'] }

features = Features({'comments': Value('string'),'targets': Value('int32')})
dataset = Dataset.from_dict(data, features = features)

def get_training_corpus():
	return (train_dataset[i:i+5]['comments'] for i in range(0, train_dataset.num_rows, 5))


train_dataset, test_dataset = [Dataset.from_dict(i) for i in train_test_split(dataset, test_size=0.2, stratify=dataset['targets'])]
training_corpus = get_training_corpus()

old_tokenizer = AutoTokenizer.from_pretrained('gpt2')
new_tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 1000)

tokens = new_tokenizer.tokenize(train_dataset[0]['comments'])
print(tokens)



















"""
ones = dataset.filter(lambda x: x['targets'] == 1)
twos = dataset.filter(lambda x: x['targets'] == 2)
threes = dataset.filter(lambda x: x['targets'] == 3)
fours = dataset.filter(lambda x: x['targets'] == 4)


print(ones.num_rows/dataset.num_rows)
print(twos.num_rows/dataset.num_rows)
print(threes.num_rows/dataset.num_rows)
print(fours.num_rows/dataset.num_rows)


# To convert to Pandas format
#dataset.set_format('pandas')

# To reset to datasets format
#dataset.reset_format()

print('-----------------')
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, stratify=dataset['targets'])


ones = Dataset.from_dict(train_dataset).filter(lambda x: x['targets'] == 1)
twos = Dataset.from_dict(train_dataset).filter(lambda x: x['targets'] == 2)
threes = Dataset.from_dict(train_dataset).filter(lambda x: x['targets'] == 3)
fours = Dataset.from_dict(train_dataset).filter(lambda x: x['targets'] == 4)


print(ones.num_rows/Dataset.from_dict(train_dataset).num_rows)
print(twos.num_rows/Dataset.from_dict(train_dataset).num_rows)
print(threes.num_rows/Dataset.from_dict(train_dataset).num_rows)
print(fours.num_rows/Dataset.from_dict(train_dataset).num_rows)

print('-----------------')

ones = Dataset.from_dict(test_dataset).filter(lambda x: x['targets'] == 1)
twos = Dataset.from_dict(test_dataset).filter(lambda x: x['targets'] == 2)
threes = Dataset.from_dict(test_dataset).filter(lambda x: x['targets'] == 3)
fours = Dataset.from_dict(test_dataset).filter(lambda x: x['targets'] == 4)


print(ones.num_rows/Dataset.from_dict(test_dataset).num_rows)
print(twos.num_rows/Dataset.from_dict(test_dataset).num_rows)
print(threes.num_rows/Dataset.from_dict(test_dataset).num_rows)
print(fours.num_rows/Dataset.from_dict(test_dataset).num_rows)
"""
