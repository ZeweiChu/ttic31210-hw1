cut -f1 senti.binary.dev.txt > senti.binary.dev.feat.txt
cut -f1 senti.binary.train.txt > senti.binary.train.feat.txt
cut -f1 senti.binary.test.txt > senti.binary.test.feat.txt

cd ../pos_tagger/stanford-postagger-2016-10-31/ 

java -cp "*" edu.stanford.nlp.tagger.maxent.MaxentTagger -model models/english-left3words-distsim.tagger -textFile ../../data/senti.binary.test.feat.txt -outputFormat tsv -outputFile ../../data/senti.binary.test.tag 
java -cp "*" edu.stanford.nlp.tagger.maxent.MaxentTagger -model models/english-left3words-distsim.tagger -textFile ../../data/senti.binary.train.feat.txt -outputFormat tsv -outputFile ../../data/senti.binary.train.tag 
java -cp "*" edu.stanford.nlp.tagger.maxent.MaxentTagger -model models/english-left3words-distsim.tagger -textFile ../../data/senti.binary.dev.feat.txt -outputFormat tsv -outputFile ../../data/senti.binary.dev.tag

cd ../../data
python convert_to_tag_text.py senti.binary.dev.tag senti.binary.dev.tag2
python convert_to_tag_text.py senti.binary.train.tag senti.binary.train.tag2
python convert_to_tag_text.py senti.binary.test.tag senti.binary.test.tag2
mv senti.binary.dev.tag2 senti.binary.dev.tag  
mv senti.binary.train.tag2 senti.binary.train.tag  
mv senti.binary.test.tag2 senti.binary.test.tag  
