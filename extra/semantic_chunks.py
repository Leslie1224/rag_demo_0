# 示例：用spaCy分割句子
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("This is the first sentence. This is the second. Third sentence here.")
sentences = [sent.text for sent in doc.sents]
# 输出：["This is the first sentence.", "This is the second.", "Third sentence here."]