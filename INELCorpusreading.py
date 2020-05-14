import nltk
from nltk.corpus import MTECorpusReader
from nltk.corpus.reader.isotei import INELTEIFileReader

"""
inel_dir = nltk.data.find('corpora/Daten/KamasTEI')
my_inelcorp = nltk.corpus.XMLCorpusReader(inel_dir, '.*\.xml')
print('Unparsed first word:', my_inelcorp.words('AA_1914_Brothers_flk.xml')[1])
"""
parsedinel_dir = nltk.data.find('corpora/Daten/KamasTEI/AA_1914_Brothers_flk.xml')
my_parsedinelcorp = INELTEIFileReader(parsedinel_dir)
print('Parsed words:', my_parsedinelcorp.words())
print('Parsed sentences:', my_parsedinelcorp.sents())
print('Parsed morphs:', my_parsedinelcorp.morphs())
print('Parsed english glosses:', my_parsedinelcorp.enggloss())
text_file = open("Output.txt", "w")
# words = my_parsedinelcorp.words()
# text_file.write('\n'.join(words))
# text_file.close()
print('Printed output to Output.txt')

# my_parsedinelcorp = INELTEIFileReader(parsedinel_dir)

# my_inelTEIcorp = nltk.corpus.MTECorpusReader(inel_dir, '.*\.xml')
# didn't really work: print('Parsed words:', my_inelTEIcorp.words())
# Parsed words: []
"""
inel_dir = nltk.data.find('corpora/Daten/KamasTEI')
my_inelcorp = nltk.corpus.XMLCorpusReader(inel_dir, '.*\.xml')
print('Unparsed first word:', my_inelcorp.words('AA_1914_Brothers_flk.xml')[1])
"""