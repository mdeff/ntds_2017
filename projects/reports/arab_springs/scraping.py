###############################################################
########################## IMPORTS ############################
###############################################################
import numpy as np
import string
import pickle
import requests
from bs4 import BeautifulSoup

###############################################################
########################## FUNCTIONS ##########################
###############################################################
def write_to_file(word_list, file):
	'''
	write a string list to a file line by line
	param word_list: list of string
	file: file name
	'''
	path = '../data/' + file
	f = open(path, 'w', encoding='utf-8') 
	for word in word_list:
		f.write(word + '\n')
	f.close()
	print('the file has been successfully created in :: ', path)

def scrap_netlingo(URL, filename):
	'''
	scrap smileys/emoticons explanations from netlingo and write it to txt file
	param URL: string, URL
	param filenam, string
	'''
	r = requests.get(URL)
	page_body = r.text
	soup = BeautifulSoup(page_body, 'html.parser')
	li = soup.find_all('li')
	symbols = []
	for p in li:
		s = p.find('span')
		if s != None:
			text = p.getText(separator=u'_______').replace('\t','').lower().split('_______')
			symbols.append(text[0].replace(' ',''))
			symbols.append(text[1])
	write_to_file(symbols, 'netlingo_{}.txt'.format(filename))


def scrap_wikitionary(URL):
	'''
	scrap english words list with frequencies from wikitionary
	param URL: string, URL
	return: dictionary {word: frequency}
	'''
	r = requests.get(URL)
	page_body = r.text
	soup = BeautifulSoup(page_body, 'html.parser')
	tr = soup.find_all('tr')
	words = []
	frequencies = []
	text_list = []
	for p in tr[1:]:
		s = p.find('td')
		if s != None:
			text = p.getText(separator=u'_______').replace('\t','').lower().split('_______')
			text = [t for t in text if t != '\n']
			text_list.append(text)
			words.append(text[1].replace(' ',''))
			frequencies.append(text[2])
	return dict(zip(words, frequencies)), text_list


###############################################################
############################ MAIN #############################
###############################################################
def main():
	# Scraping Acronyms from netlingo
	URL = 'http://www.netlingo.com/acronyms.php'
	acronyms = scrap_netlingo(URL, 'acronyms')

	# Scraping Smileys from netlingo
	URL = 'http://www.netlingo.com/smileys.php'
	acronyms = scrap_netlingo(URL, 'smileys')

	#Scrap frequency dictionary from Wikitionary
	english_dict = {}
	for i in range(4):
		URL = 'https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2006/04/{}-{}'.format(10000*i+1, 10000*(i+1))
		english_dict.update(scrap_wikitionary(URL)[0])
	write_to_file(english_dict, 'dictionary.txt')


if __name__ == "__main__":
	main()