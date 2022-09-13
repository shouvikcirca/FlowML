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

mlflow.set_tracking_uri('http://localhost:5000')
#os.environ['MLFLOW_TRACKING_URI'] = 'postgresql+psycopg2://postgres:password@localhost:5432/mlflowdb'
#os.environ['MLFLOW_S3_ENDPOINT_URL'] =  'http://10.180.146.26:9000'
#os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
#os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'

#mlflow.keras.autolog(log_models = False)

class ModelBuilder:
	def __init__(self, learning_rate, vocab_length, seq_padding_style, seq_truncating_style, embedding_dim, bs, epochs, max_length, run_id):
		self.learning_rate = learning_rate
		self.vocab_length = vocab_length
		self.seq_padding_style = seq_padding_style
		self.seq_truncating_style = seq.truncating_style
		self.embedding_dim = embedding_dim
		self.bs = bs
		self.epochs = epochs
		self.max_length = max_length
		self.run_id = run_id

	def build_model(self, hp):
		model = keras.models.Sequential()
		model.add(tf.keras.layers.Embedding(
		vocab_length = hp.Int('vocab_length', min_value = self.vocab_length_start, max_value=self.vocab_length_end, step=1),embedding_dim = embedding_dim, input_length = max_length))
		model.add(tf.keras.layers.GlobalAveragePooling1D())
		model.add(tf.keras.layers.Dense(6, activation='relu'))
		model.add(tf.keras.layers.Dense(4, activation='softmax'))

		model.compile(
			loss='categorical_crossentropy', 
			optimizer=tf.keras.optimizers.Adam(learning_rate = hp.Float('learning_rate', min_value=0.1, max_value=0.3)), 
			metrics=["accuracy"]
		)
		return model


@click.command()
@click.option("--learning-rate-start", default=0.1, type=float)
@click.option("--learning-rate-end", default=0.3, type=float)
@click.option("--vocab-length_start", default=80, type=int)
@click.option("--vocab-length_end", default=100, type=int)
@click.option("--seq-padding-style", default="post", type=str)
@click.option("--seq-truncating-style", default="post", type=str)
@click.option("--embedding-dim", default=50, type=int)
@click.option("--bs", default=64, type=int)
@click.option("--epochs", default=5, type=int)
@click.option("--max-length", default=50, type=int)
@click.option("--run-id", type=str)
def runBaseRunner(lr_start,lr_end, vocab_length_start, vocab_length_end, embedding_dim_start,embedding_dim_end,bs_start, bs_end, epochs_start, epochs_end,max_length_start, max_length_end), run_id):
	

	search_space = {
		'run_id':run_id,
		'learning_rate':tune.uniform(lr_start, lr_end),
		'vocab_length':tune.randint(vocab_length_start, vocab_length_end)
		'seq_padding_style':tune.randInt(0,3),
		'seq_truncating_style':tune.randInt(0,3),
		'embedding_dim'tune.randint(embedding_dim_start, embedding_dim_end),
		'bs':tune.randint(bs_start, bs_end),
		'epochs':tune.randint(epochs_start, epochs_end),
		'max_length':tune.randint(max_length_start, max_length_end)
		}
	}

	algo = BayesOptSearch(utility_kwargs={'kind':'ucb', 'kappa':2.5, 'xi':0.0})
	algo = ConcurrencyLimiter(algo, max_concurrent=4)

	num_samples = 5

	tuner = tune.Tuner(
			train,
			tune_config = tune.TuneConfig(
				metric='val_loss',
				mode='min'
				search_alg = algo,
				num_samples = num_samples
				),
			param_space=search_space
			)

	
	results = tuner.fit()
	print(results.get_best_result(metric='score', mode='min').config)


def train(config):


	styleMap = {
			0:'post',
			1:'pre'
			}

	paddingStyle = styleMap[config['seq_padding_style'].sample()]
	truncatingStyle = styleMap[config['seq_truncating_style'].sample()]

	signature = ''
	client = MlflowClient()
	with mlflow.start_run(run_id = config.run_id) as active_run:
		r = client.get_run(active_run.info.run_id)
		
		val_accuracy = -float('inf')
		train_accuracy = -float('inf')
		model_to_save = None

		filePath = r.data.tags["dataPath"]
		comments, targets = extractor(filePath)

		#Converting to numpy arrays
		comments = np.array(comments)
		targets = [i-1 for i in targets]
		targets = np.array(targets)

		splitter = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state = 0)

		for train_index, test_index in splitter.split(comments, targets):
			trainX = comments[train_index]
			trainY = targets[train_index]
			testX = comments[test_index]
			testY = targets[test_index]
			tokenizer = Tokenizer(num_words=config.vocab_length, oov_token="<OOV>")

			tokenizer.fit_on_texts(trainX) 
			
			Saving tokenizer because it will be needed for inference
			with open('tokenizer.pkl','wb') as wp:
				pickle.dump(tokenizer, wp, pickle.HIGHEST_PROTOCOL)
			client.log_artifact(r.info.run_id, 'tokenizer.pkl')
			os.remove('tokenizer.pkl')
			
			trainSequences = tokenizer.texts_to_sequences(trainX)
			paddedTrainSequences = pad_sequences(trainSequences, padding=paddingStyle, truncating=truncatingStyle, maxlen=config.max_length)
			testSequences = tokenizer.texts_to_sequences(testX)
			paddedTestSequences = pad_sequences(testSequences, padding=paddingStyle, truncating=truncatingStyle, maxlen=config.max_length)

			trainY = tf.keras.utils.to_categorical(trainY, 4)
			testY = tf.keras.utils.to_categorical(testY, 4)

			# Defining signature for the model
			signature = infer_signature(paddedTestSequences , testY)
		
			cur_train_accuracy=model.history.history['accuracy'][-1]
			cur_val_accuracy=model.evaluate(paddedTestSequences, testY, batch_size=bs)[1]
			if cur_val_accuracy > val_accuracy:
				val_accuracy = cur_val_accuracy,min_value=vocab_length[0],max_value=vocab_length[1]),step=1y
				train_accuracy = cur_train_accuracy
				model_to_save = model
			
		client.log_metric(run_id = active_run.info.run_id, key='trainAccuracy', value=train_accuracy)
		client.log_metric(run_id = active_run.info.run_id, key='valAccuracy', value=val_accuracy)
		client.set_terminated(active_run.info.run_id)
		mlflow.keras.log_model(keras_model = model_to_save, artifact_path='artifact_{}'.format(active_run.info.run_id), signature = signature)






def extractor(filePath):
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
	"""
	learning_rate_start=0.1 
	learning_rate_end=0.3

	vocab_length_start=20 
	vocab_length_end=25 

	seq_padding_style='post'
	seq_truncating_style='post'
	
	embedding_dim_start=20
	embedding_dim_Etart=20
	
	bs_start=16 
	bs_start=18 
	
	epochs_start=5 
	epochs_end=5 

	max_length_start=50
	max_length_end=60

	#run_id = []
	"""
	runBaseRunner()




#Creating an object because the hyperparameters have to be input by the user. So we need to maintain state
#model_blueprint = ModelBuilder(config.learning_rate, config.vocab_length, config.seq_padding_style, config.seq_truncating_style, config.embedding_dim, config.bs, config.epochs, config.max_length)
#tuner = keras_tuner.RandomSearch(
#		model_blueprint.build_model(),
#		objective = 'val_loss',
#		max_trials = 5)

#tuner.search(paddedTrainSequences, trainY, epochs=20, validation_data=(paddedTestSequences, testY))
#print(dir(tuner))
