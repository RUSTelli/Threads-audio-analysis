from transformers import BertTokenizerFast, DistilBertModel

tokenizer = BertTokenizerFast("osiria/distilbert-base-italian-cased")
model = DistilBertModel.from_pretrained("osiria/distilbert-base-italian-cased")




