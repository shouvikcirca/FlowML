from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForSequenceClassification
from tokenizers import Tokenizer
import evaluate

connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
dbClient = MongoClient(connectionString)
db = dbClient['mlflowexperiments']
collection = db['trainingdata']

classLabels = [str(i) for i in range(5)]
cLabel = ClassLabel(names = classLabels)

dbData = collection.find({"comments": {'$exists': True}})[0]
data = {"comments": dbData['comments'], "label": dbData['targets'] }
features = Features({'comments': Value('string'),'label': Value('int32')})
dataset = Dataset.from_dict(data, features = features)

#split_dataset = dataset.train_test_split(test_size=0.2)# Not using because it does not contain the option to stratify
train_dataset, test_dataset = [Dataset.from_dict(i) for i in train_test_split(dataset, test_size=0.2, stratify=dataset['label'])]

train_dataset = dataset.cast_column("label", cLabel)
test_dataset = dataset.cast_column("label", cLabel)

wholeset = DatasetDict({"train": train_dataset, "test": test_dataset})

def tokenize_function(sample):
	return tokenizer(sample['comments'], padding = 'max_length', truncation = True)

checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_dataset = wholeset.map(tokenize_function, batched=True)
#print(tokenized_dataset['train'][0])
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)



def compute_metrics(eval_preds):
	metric = evaluate.load("glue","mrpc")
	logits, labels = eval_preds
	predictions = np.argmax(logits, axis=-1)
	return metric.compute(predictions = predictions, references = labels)

training_args = TrainingArguments("modelDirectory", evaluation_strategy = "epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5)




trainer = Trainer(
		model,
		training_args,
		train_dataset = tokenized_dataset['train'],
		eval_dataset = tokenized_dataset['test'],
		data_collator = data_collator,
		tokenizer = tokenizer,
		compute_metrics = compute_metrics
		)

trainer.train()





