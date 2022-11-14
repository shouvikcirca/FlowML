s = ''
with open('ObsceneCom.txt','r') as f:
	s = f.read().split('\n')
	s = s[:-1]
	s = [i.split('\t')[1] for i in s]
	with open('inferenceData.txt','w+') as f:
		for i in s:
			f.write(i+'\n')


