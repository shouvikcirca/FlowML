import docker
import subprocess

def modelServing():
	experimentName = 'Alpha'
	runid = 'ea0697bed5c348d6b46e37674553325a'
	model_name = '{}_{}'.format(experimentName, runid)

	cmdString = 'mlflow models serve -m "models:/{}/Staging" -h 127.0.0.1 -p 5004 --env-manager=local'.format(model_name)
	p1 = subprocess.run(cmdString, shell = True, capture_output = True, text = True)
	print(p1.stdout)
	print('Maybe it works')


if __name__ == '__main__':
	modelServing()
