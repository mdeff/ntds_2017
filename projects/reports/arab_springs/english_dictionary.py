###############################################################
########################## IMPORTS ############################
###############################################################
import numpy as np
import string
from nltk.tokenize import TweetTokenizer
import pickle


###############################################################
########################## FUNCTIONS ##########################
###############################################################
def tokenize(text):
	'''
	Tokenize string using nltk tweet tokenizer
	param text: string
	return: list of tokens
	'''
	tknzr = TweetTokenizer()
	return tknzr.tokenize(text)

def correct(sentence, contrac_dict={}):
	'''
	replace contractions in sentence and remove punctuation
	param sentence: string
	param contrac_dict: dictionary, english contraction
	return string, corrected sentece
	'''
	tokens = tokenize(sentence)
	new_tokens = []
	for token in tokens:
		if token in contrac_dict:
			new_tokens.append(contrac_dict[token])
		if len(token)>1:
			new_tokens.append(''.join(c for c in token if c not in string.punctuation))
	return ' '.join(new_tokens)


###############################################################
############################ MAIN #############################
###############################################################
def main():

	######## Upload dictionaries ######
	###################################
	#Define Paths
	#BASE = '../data/dictionaries/'
	BASE = ''

	## English Dicionary
	english_words = np.asarray([line.rstrip('\n').lower() for line in open(BASE+'english_words.txt')])
	idx = np.arange(int(len(english_words)/3))
	english_dictionary = dict(zip(english_words[3*idx+1], english_words[3*idx+2]))
	freq =  dict(zip(english_words[3*idx+1], english_words[3*idx+2]))

	## English contractions (#ignore)
	contractions = np.asarray([line.rstrip('\n').lower() for line in open(BASE+'contractions.txt')])
	idx = np.arange(int(len(contractions)/2))
	contractions_dict = dict(zip(contractions[2*idx], contractions[2*idx+1]))

	## Acronyms
	acronyms = np.asarray([line.rstrip('\n').lower() for line in open(BASE+'netlingo_acronyms.txt')])
	idx = np.arange(int(len(acronyms)/2))
	acronyms_dict = dict(zip(acronyms[2*idx], acronyms[2*idx+1]))

	#Remove multi explications
	for key in acronyms_dict:
		acronyms_dict[key] = acronyms_dict[key].split('/ ')[0]
  	
  	#correct descriptions
	for key in acronyms_dict:
		acronyms_dict[key] = correct(acronyms_dict[key])

	## Smileys
	smileys = np.asarray([line.rstrip('\n').lower() for line in open(BASE+'netlingo_smileys.txt')])
	idx = np.arange(int(len(smileys)/2))
	smileys_dict = dict(zip(smileys[2*idx], smileys[2*idx+1]))
	
	#Remove multi explications
	for key in smileys_dict:
		smileys_dict[key] = smileys_dict[key].split('- ')[0]

	## Final Dicionary
	freq_dict = { k:v for k, v in english_dictionary.items()}

	final_dict = { k:k for k, v in english_dictionary.items()}
	final_dict.update(acronyms_dict)
	final_dict.update(smileys_dict)

	# Save Dictionary
	pickle.dump([final_dict, freq_dict], open('../data/dictionaries.p', 'wb'))

if __name__ == "__main__":
	main()