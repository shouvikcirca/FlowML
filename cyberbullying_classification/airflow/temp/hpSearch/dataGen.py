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
			"learning_rate_start":randomRange(0.1,0.5),
			"learning_rate_end":0.5+randomRange(0.1,0.9),
			#"dataPath":'./',
			#"max_length_start":50,
			#"max_length_end":60,
			"vocab_length_start":np.random.randint(20,50),
			"vocab_length_end":50 + np.random.randint(1,10),
			"seq_padding_style":"post",
			#"seq_truncating_style":"post",
			"embedding_dim_start":np.random.randint(50),
			"embedding_dim_end":50 + np.random.randint(1,10),
			"bs_start":np.random.randint(32,64),
			"bs_end":64+np.random.randint(1,20),
			"epochs_start":i,
			"epochs_end":np.random.randint(i+1,i+8)
		}
		i+=1
		writeList.append(d)

	with open("data.json",'w') as f:
		json.dump(writeList, f)


	data = json.load(open("data.json"))
	print("{} hyperparameter sets added".format(len(data)))
