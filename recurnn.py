from nltk.tokenize import word_tokenize
from torch.autograd import Variable
import csv
import spacy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set hyperparameters
embed_size = 100
batch_size = 20
l_rate = 0.001
num_epochs = 10

# Read in text data
text_file = open('sentiment_dataset.csv', 'r')
reader = csv.reader(text_file)
data = []
full_text = ''
for line in reader:
	data.append(line)
	full_text += line[-1]
text_file.close()

# Create vocabulary
word_list = word_tokenize(full_text.lower())
vocab = np.unique(word_list)
vocab_size = len(vocab)
print(vocab_size)

# Create word to index mapping
w_to_i = {word: ind for ind, word in enumerate(vocab)}

# Load text parser
nlp = spacy.load('en')

class RecurNN(nn.Module):
	def __init__(self):
		super(RecurNN, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embed_size)
		self.first_linear = nn.Linear(2*embed_size, embed_size)
		self.tanh = nn.Tanh()
		self.final_linear = nn.Linear(embed_size, 1)
		self.sigmoid = nn.Sigmoid()

	def parse(sent):
		doc = nlp(sent)
		root = doc[0]
		for word in doc:
			if word.head == word:
				root = word
				break
		return root
		
	def compose(a, b):
		ab = torch.cat((a, b), 0)
		out = self.first_linear(ab)
		out = self.tanh(out)
		return out

	def compute_vec(root):
		lookup = torch.LongTensor([w_to_i[root.text]])
		embed_vec = self.embeddings(lookup).view((1, -1))
		if len(root.children) == 0:
			return embed_vec
		vec_list = []
		for child in root.children:
			vec_list.append(compute_vec(child))
			if len(vec_list) == 2:
				comp_vec = compose(vec_list[0], vec_list[1])
				vec_list.clear()
				vec_list.append(comp_vec)
		return compose(vec_list[-1], embed_vec)

	def forward(self, x):
		final_vec = compute_vec(parse(x))
		return self.sigmoid(self.final_linear(final_vec)).view((-1, 1))

model = RecurNN()

for epoch in range(len(num_iterations)):
	for line in data:
		print(line)
		print(word_tokenize(line.lower()).join(' '))


