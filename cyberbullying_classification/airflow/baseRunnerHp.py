import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedShuffleSplit
import mlflow
from mlflow.tracking.client import MlflowClient
import click
import mlflow.keras
import pickle
from mlflow.models.signature import infer_signature
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from functools import partial

#mlflow.set_tracking_uri('http://localhost:5000')
#os.environ['MLFLOW_TRACKING_URI'] = 'postgresql+psycopg2://postgres:password@localhost:5432/mlflowdb'
#os.environ['MLFLOW_S3_ENDPOINT_URL'] =  'http://10.180.146.26:9000'
#os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
#os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'
#mlflow.keras.autolog(log_models = False)


@click.command()
@click.option("--experiment-name", default='default', type=str)
@click.option("--learning-rate-start", default=0.1, type=float)
@click.option("--learning-rate-end", default=0.3, type=float)
@click.option("--vocab-length-start", default=80, type=float)
@click.option("--vocab-length-end", default=100, type=float)
@click.option("--seq-padding-style", default="post", type=str)
#@click.option("--seq-truncating-style", default="post", type=str)
@click.option("--embedding-dim-start", default=50, type=float)
@click.option("--embedding-dim-end", default=50, type=float)
@click.option("--bs-start", default=64, type=float)
@click.option("--bs-end", default=64, type=float)
@click.option("--epochs-start", default=5, type=float)
@click.option("--epochs-end", default=5, type=float)
#@click.option("--max-length-start", default=50, type=int)
#@click.option("--max-length-end", default=50, type=int)
#@click.option("--run-id", type=str)
def runBaseRunner(experiment_name, learning_rate_start,learning_rate_end, vocab_length_start, vocab_length_end, seq_padding_style, embedding_dim_start, embedding_dim_end, bs_start, bs_end, epochs_start, epochs_end):#, run_id):#, max_length_start, max_length_end)

	search_space = {
		'learning_rate':tune.uniform(learning_rate_start, learning_rate_end),
		'vocab_length':tune.uniform(vocab_length_start, vocab_length_end),
		'seq_padding_style':tune.uniform(0,1),
		'seq_truncating_style':tune.uniform(0,1),
		'embedding_dim':tune.uniform(embedding_dim_start, embedding_dim_end),
		'bs':tune.uniform(bs_start, bs_end),
		'epochs':tune.uniform(epochs_start, epochs_end),
		'max_length':tune.uniform(50, 60)
		}

	algo = BayesOptSearch(utility_kwargs={'kind':'ucb', 'kappa':2.5, 'xi':0.0})
	#algo = ConcurrencyLimiter(algo, max_concurrent=4)

	num_samples = 5

	tuner = tune.Tuner(
			partial(train, data_dir=experiment_name),
			tune_config = tune.TuneConfig(
				metric='val_loss',
				mode='min',
				search_alg = algo,
				num_samples = num_samples
				),
			param_space=search_space
			)

	
	results = tuner.fit()
	print(results.get_best_result(metric='val_loss', mode='min').config)


def returnExperimentName():
	return thisExperimentName # global variable declared at the top and assigned in 'runBaseRunner()'

def trunc_or_round(num):
	return int(num) if num-int(num)<=0.5 else int(num+1)


class ModelBuilder:
	def __init__(self, learning_rate, vocab_length, seq_padding_style, seq_truncating_style, embedding_dim, bs, epochs, max_length):
		self.learning_rate = learning_rate
		self.vocab_length = vocab_length
		self.embedding_dim = embedding_dim
		self.max_length = max_length

	def build_model(self):
		model = tf.keras.Sequential([
			tf.keras.layers.Embedding(self.vocab_length, self.embedding_dim, input_length=self.max_length),
			tf.keras.layers.GlobalAveragePooling1D(),
			tf.keras.layers.Dense(6, activation='relu'),
			tf.keras.layers.Dense(4, activation='softmax')
		])

		model.compile(
			loss='categorical_crossentropy', 
			optimizer=tf.keras.optimizers.Adam(learning_rate = self.learning_rate), 
			metrics=["accuracy"]
		)
		return model



def extractor(filePath):
	print('-----')
	print(filePath)
	print(os.listdir(filePath))
	print('comment extractor was hit')
	files = [i for i in os.listdir(filePath) if '.txt' in i] # Only .txt files contain data
	print('-----')
	print(files)
	print('-----')
	comments = []
	targets = []
	for f in files:
		with open(filePath+f, 'r') as r:
			s = r.read().split('\n')
			s = [i.split('\t') for i in s]
			comments.extend(s)	

	targets = [int(item[0]) for item in comments if len(item) == 2]	
	comments = [item[1] for item in comments if len(item) == 2]
	return comments, targets


def train(config, data_dir):

	styleMap = {0:'post',1:'pre'}

	learning_rate = config['learning_rate']

	# Separating string hyperparameters
	seq_padding_style = styleMap[trunc_or_round(config['seq_padding_style'])]
	seq_truncating_style = styleMap[trunc_or_round(config['seq_truncating_style'])]

	# Separating integer hyperparameters
	vocab_length = trunc_or_round(config['vocab_length'])
	embedding_dim = trunc_or_round(config['embedding_dim'])
	bs = trunc_or_round(config['bs'])
	epochs = trunc_or_round(config['epochs'])
	max_length = trunc_or_round(config['max_length'])


	runHyperparams = {
			"learning_rate":learning_rate,
			"seq_padding_style":seq_padding_style,
			"seq_truncating_style":seq_truncating_style,
			"vocab_length":vocab_length,
			"embedding_dim":embedding_dim,
			"batchSize":bs,
			"epochs":epochs,
			"max_length":max_length
			}


	print('The experiment we are trying to execute is {}'.format(data_dir))
	signature = ''
	client = MlflowClient()
	exp_id = mlflow.get_experiment_by_name(data_dir).experiment_id 


	run = client.create_run(exp_id)#, tags = {"dataPath":hyperparams["dataPath"]})

	val_accuracy = -float('inf')
	val_loss = 0
	train_accuracy = -float('inf')
	model_to_save = None
	with mlflow.start_run(run_id = run.info.run_id) as active_run:

		mlflow.log_params(runHyperparams)
		r = client.get_run(active_run.info.run_id)
		
		filePath = '/home/centos/airflow/{}'.format(data_dir)#getDataPath()
		#print('{} is the filePath being used'.format(filePath))
		
		print('About to hit comment extractor')
		comments, targets = extractor(filePath)
		print('Just hit comment extractor')

		#Converting to numpy arrays
		comments = np.array(comments)
		targets = [i-1 for i in targets]
		targets = np.array(targets)

		print('comments: {}'.format(comments.shape))
		print('targets: {}'.format(targets.shape))
		splitter = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state = 12)

		for train_index, test_index in splitter.split(comments, targets):
			trainX = comments[train_index]
			trainY = targets[train_index]
			testX = comments[test_index]
			testY = targets[test_index]
			tokenizer = Tokenizer(num_words=vocab_length, oov_token="<OOV>")

			tokenizer.fit_on_texts(trainX) 
			
			#Saving tokenizer because it will be needed for inference
			with open('tokenizer.pkl','wb') as wp:
				pickle.dump(tokenizer, wp, pickle.HIGHEST_PROTOCOL)
			client.log_artifact(r.info.run_id, 'tokenizer.pkl')
			os.remove('tokenizer.pkl')
			
			trainSequences = tokenizer.texts_to_sequences(trainX)
			paddedTrainSequences = pad_sequences(trainSequences, padding=seq_padding_style, truncating=seq_truncating_style, maxlen=max_length)
			testSequences = tokenizer.texts_to_sequences(testX)
			paddedTestSequences = pad_sequences(testSequences, padding=seq_padding_style, truncating=seq_truncating_style, maxlen=max_length)

			trainY = tf.keras.utils.to_categorical(trainY, 4)
			testY = tf.keras.utils.to_categorical(testY, 4)

			# Defining signature for the model
			signature = infer_signature(paddedTestSequences , testY)
	
			model = ModelBuilder(learning_rate, vocab_length, seq_padding_style, seq_truncating_style, embedding_dim, bs, epochs, max_length) 
			model = model.build_model()
			history = model.fit(paddedTrainSequences, trainY, batch_size=bs, epochs=epochs, verbose=1)	
			cur_train_accuracy=model.history.history['accuracy'][-1]
			cur_val_accuracy=model.evaluate(paddedTestSequences, testY, batch_size=bs)
			if cur_val_accuracy[1] > val_accuracy:
				val_accuracy = cur_val_accuracy[1]
				train_accuracy = cur_train_accuracy
				model_to_save = model
				val_loss = cur_val_accuracy[0]
			
		client.log_metric(run_id = active_run.info.run_id, key='trainAccuracy', value=train_accuracy)
		client.log_metric(run_id = active_run.info.run_id, key='valAccuracy', value=val_accuracy)
		client.log_metric(run_id = active_run.info.run_id, key='valLoss', value=val_loss)
		client.set_terminated(active_run.info.run_id)
		mlflow.keras.log_model(keras_model = model_to_save, artifact_path='artifact_{}'.format(active_run.info.run_id), signature = signature)
		return {'val_loss':val_loss}

def extractor(filePath):
	filePath+='/'
	files = [i for i in os.listdir(filePath) if '.txt' in i] # Only .txt files contain data
	comments = []
	targets = []
	for f in files:
		with open(filePath+f, 'r') as r:
			s = r.read().split('\n')
			s = [i.split('\t') for i in s]
			comments.extend(s)	

	targets = [int(item[0]) for item in comments if len(item) == 2]	
	comments = [item[1] for item in comments if len(item) == 2]
	return comments, targets


if __name__ == "__main__":
	runBaseRunner()


#Creating an object because the hyperparameters have to be input by the user. So we need to maintain state
#model_blueprint = ModelBuilder(config.learning_rate, config.vocab_length, config.seq_padding_style, config.seq_truncating_style, config.embedding_dim, config.bs, config.epochs, config.max_length)
#tuner = keras_tuner.RandomSearch(
#		model_blueprint.build_model(),
#		objective = 'val_loss',
#		max_trials = 5)

#tuner.search(paddedTrainSequences, trainY, epochs=20, validation_data=(paddedTestSequences, testY))
#print(dir(tuner))
