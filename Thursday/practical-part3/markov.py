from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

# Read the data
path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = io.open(path, encoding='utf-8').read().lower()
print('corpus length:', len(text))

transition_counts = dict()
for i in range(0,len(text)-1):
	currc = text[i]
	nextc = text[i+1]
	if currc not in transition_counts:
		transition_counts[currc] = dict()
	if nextc not in transition_counts[currc]:
		transition_counts[currc][nextc] = 0
	transition_counts[currc][nextc] += 1

print("Number of transitions from 'a' to 'b': " + str(transition_counts['a']['b']))

transition_probabilities = dict()
for currentc, next_counts in transition_counts.items():
	values = []
	probabilities = []
	sumall = 0
	for nextc, count in next_counts.items():
		values.append(nextc)
		probabilities.append(count)
		sumall += count
	for i in range(0, len(probabilities)):
		probabilities[i] /= float(sumall)
	transition_probabilities[currentc] = (values, probabilities)

for a,b in zip(transition_probabilities['a'][0], transition_probabilities['a'][1]):
	print(a,b)

# sample
current = 't'
for i in range(0, 1000):
	print(current, end='')
	values, probabilities = transition_probabilities[current]
	current = np.random.choice(values, p=probabilities)
