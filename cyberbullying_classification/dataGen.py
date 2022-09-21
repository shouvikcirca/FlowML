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
	i = 1
	while i<=num_hsets:
		d = {
			"learning_rate":randomRange(0.1,0.9),
			"dataPath":'./',
			"max_length":50,
			"vocab_length":100,
			"seq_padding_style":"post",
			"seq_truncating_style":"post",
			"embedding_dim":50,
			"bs":int(randomRange(32,64)),
			"epochs":i*10
		}
		i+=1
		writeList.append(d)

	with open("data.json",'w') as f:
		json.dump(writeList, f)


	data = json.load(open("data.json"))
	print("{} hyperparameter sets added".format(len(data)))
