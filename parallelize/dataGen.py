import json
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p")
args = parser.parse_args()


num_hsets = int(args.p)

def randomRange(start, end):
	scale = end-start
	return start + np.random.random()*scale


if __name__ == '__main__':
	writeList = []
	for i in range(num_hsets):
		d = {
			"learning_rate":randomRange(0.1,0.9),
			"dataPath":'./',
			"vocab_length":int(randomRange(50,100)),
			"seq_padding_style":"post",
			"seq_truncating_style":"post",
			"embedding_dim":int(randomRange(50,100)),
			"bs":int(randomRange(32,64)),
			"epochs":int(randomRange(5,25)),
			"max_length":int(randomRange(20,50)),
		}
		writeList.append(d)

	with open("data.json",'w') as f:
		json.dump(writeList, f)


	data = json.load(open("data.json"))
	print("{} hyperparameter sets added".format(len(data)))
