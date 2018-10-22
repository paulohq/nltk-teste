import nltk

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')

from nltk.tag.stanford import StanfordPOSTagger as POS_Tag
from nltk.tag.stanford import StanfordNERTagger as NER_Tag
#from nltk.tokenize.stanford_segmenter import StanfordSegmenter

english_postagger = POS_Tag('stanford-postagger-full-2014-08-27/models/english-bidirectional-distsim.tagger',
                            'stanford-postagger-full-2014-08-27/stanford-postagger.jar')

print(english_postagger.tag('this is stanford postagger in nltk for python users'.split()))

#english_nertagger = NER_Tag('stanford-ner-2014-08-27/classifiers/english.all.3class.distsim.crf.ser.gz', 'stanford-ner-2014-08-27/stanford-ner.jar')
english_nertagger = NER_Tag('stanford-ner-2014-08-27/classifiers/english.muc.7class.distsim.crf.ser.gz', 'stanford-ner-2014-08-27/stanford-ner.jar')

#7 class: Time, Location, Organization, Person, Money, Percent, Date
print(english_nertagger.tag('Rami Eid is studying at Stony Brook University in Brazil monday april 2018 at cost 25 dollars and 50 percent per day '.split()))

#segmenter
#segmenter = StanfordSegmenter(path_to_jar="stanford-segmenter-2014-08-27/stanford-segmenter-3.4.1.jar", path_to_sihan_corpora_dict="./data", path_to_model="./data/pku.gz", path_to_dict="./data/dict-chris6.ser.gz")

#sentence = "this is one simple test"

#segmenter.segment(sentence)

#segmenter.segment_file("test.simp.utf8")