
import os
import numpy as np
import tensorflow as tf
import mlflow
from mlflow.tracking.client import MlflowClient
import click
import mlflow.keras
import pickle
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from mlflow.models.signature import infer_signature


mlflow.set_tracking_uri('http://localhost:5000')
os.environ['MLFLOW_S3_ENDPOINT_URL'] =  'http://10.180.146.26:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'


mlflow.keras.autolog(log_models = False)

@click.command()
@click.option("--learning-rate", default=0.001, type=float)
@click.option("--bs", default=64, type=int)
@click.option("--epochs", default=5, type=int)
@click.option("--run-id", type=str)
def runBaseRunner(learning_rate, bs, epochs, run_id):

	client = MlflowClient()
	with mlflow.start_run(run_id = run_id) as active_run:
		r = client.get_run(active_run.info.run_id)
		
		val_accuracy = -float('inf')
		train_accuracy = -float('inf')
		model_to_save = None

		filePath = r.data.tags["dataPath"]

		DataGen = ImageDataGenerator(
                                   rescale = 1./255,
                                   rotation_range = 20,
                                   validation_split = 0.2
                                   )

		training_data = DataGen.flow_from_directory(
				directory = filePath,
                                target_size = (128,128),
                                batch_size = bs,
                                class_mode = 'binary'
                                #subset = 'train'
                                )
		test_data = DataGen.flow_from_directory(
				directory = filePath,
				target_size = (128,128),
				batch_size = bs,
				class_mode = 'binary'
				#subset = 'test'
                                )


		model = Sequential()
		model.add(Conv2D(input_shape = (128,128,3), activation = 'relu', filters = 32, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size = (2,2)))
		model.add(Conv2D(filters = 64, activation = 'relu', kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size = (2,2)))
		model.add(Conv2D(filters = 128, activation = 'relu', kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size = (2,2)))
		model.add(Conv2D(filters = 256, activation = 'relu', kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size = (2,2)))
		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(64, activation = 'relu'))
		model.add(Dropout(0.2))
		model.add(Dense(128, activation = 'relu'))
		model.add(Dropout(0.4))
		model.add(Dense(256, activation = 'relu'))
		model.add(Dropout(0.5))
		model.add(Dense(2, activation = 'softmax'))

		model.compile(
              		optimizer = Adam(lr = learning_rate),
              		loss = 'sparse_categorical_crossentropy',
              		metrics = ['accuracy']
		)


		callback = tf.keras.callbacks.EarlyStopping(
                                            monitor='val_loss', 
                                            mode = 'min', 
                                            patience=5
                                           )

		history = model.fit(
                              training_data,
                              validation_data = test_data,
                              epochs = epochs,
                              verbose = 1,
                              shuffle = True,
                              callbacks = [callback]
                             )
			
			
		train_accuracy = model.history.history['accuracy'][-1]
		val_accuracy = model.evaluate(test_data)
		model_to_save = model
			
		client.log_metric(run_id = active_run.info.run_id, key='trainAccuracy', value=train_accuracy)
		client.log_metric(run_id = active_run.info.run_id, key='valAccuracy', value=val_accuracy[1])
		client.set_terminated(active_run.info.run_id)
		mlflow.keras.log_model(keras_model = model_to_save, artifact_path='artifact_{}'.format(active_run.info.run_id))

		plt.plot(history.history['accuracy'])
		plt.plot(history.history['val_accuracy'])
		plt.title('Model Accuracy', fontsize = 20)
		plt.ylabel('Accuracy', fontsize = 15)
		plt.xlabel('Epoch', fontsize = 15)
		plt.legend(['train', 'test'], loc = 'upper left')
		plt.savefig('model_accuracy_graph.png')

		client.log_artifact(r.info.run_id, 'model_accuracy_graph.png')
		os.remove('model_accuracy_graph.png')


if __name__ == "__main__":
	runBaseRunner()





