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

mlflow.keras.autolog(log_models = False)

@click.command()
@click.option("--learning-rate", default=0.1, type=float)
@click.option("--vocab-length", default=100, type=int)
@click.option("--seq-padding-style", default="post", type=str)
@click.option("--seq-truncating-style", default="post", type=str)
@click.option("--embedding-dim", default=50, type=int)
@click.option("--bs", default=64, type=int)
@click.option("--epochs", default=5, type=int)
@click.option("--max-length", default=50, type=int)
@click.option("--run-id", type=str)
def runBaseRunner(learning_rate, vocab_length, seq_padding_style, seq_truncating_style, embedding_dim, bs, epochs, max_length, run_id):
	client = MlflowClient()
	with mlflow.start_run(run_id = run_id) as active_run:
		r = client.get_run(active_run.info.run_id)
		
		filePath = r.data.tags["dataPath"]
		comments, targets = extractor(filePath)

		#Converting to numpy arrays
		comments = np.array(comments)

		targets = [i-1 for i in targets]
		targets = np.array(targets)

		model = tf.keras.Sequential([
			tf.keras.layers.Embedding(vocab_length, embedding_dim, input_length=max_length),
			tf.keras.layers.GlobalAveragePooling1D(),
			tf.keras.layers.Dense(6, activation='relu'),
			tf.keras.layers.Dense(4, activation='softmax')
		])

		model.compile(
				loss='categorical_crossentropy', 
				optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate), 
				metrics=["accuracy"]
		)
		splitter = StratifiedShuffleSplit(n_splits = 3, test_size=0.2, random_state = 0)

		for train_index, test_index in splitter.split(comments, targets):
			trainX = comments[train_index]
			trainY = targets[train_index]
			testX = comments[test_index]
			testY = targets[test_index]
			tokenizer = Tokenizer(num_words=vocab_length, oov_token="<OOV>")
			tokenizer.fit_on_texts(trainX) 
			trainSequences = tokenizer.texts_to_sequences(trainX)
			paddedTrainSequences = pad_sequences(trainSequences, padding=seq_padding_style, truncating=seq_truncating_style, maxlen=max_length)
			testSequences = tokenizer.texts_to_sequences(testX)
			paddedTestSequences = pad_sequences(testSequences, padding=seq_padding_style, truncating=seq_truncating_style, maxlen=max_length)
			trainY = tf.keras.utils.to_categorical(trainY, 4)
			testY = tf.keras.utils.to_categorical(testY, 4)
			
			history = model.fit(paddedTrainSequences, trainY, batch_size=bs, epochs=epochs, verbose=1)	
			train_accuracy = model.history.history['accuracy'][-1]
			val_accuracy = model.evaluate(paddedTestSequences, testY, batch_size=bs)[1]
			
			client.log_metric(run_id = active_run.info.run_id, key='trainAccuracy', value=train_accuracy)
			client.log_metric(run_id = active_run.info.run_id, key='valAccuracy', value=val_accuracy)
			client.set_terminated(active_run.info.run_id)
			mlflow.keras.log_model(keras_model = model, artifact_path='artifact_{}'.format(active_run.info.run_id))


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
	runBaseRunner()

