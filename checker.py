import json

d = json.load(open("webeo.json", encoding="utf-8"))

vocab = []
with open("bert_models/uncased/vocab.txt", encoding="utf-8") as fp:
    for w in fp:
        vocab.append(w.strip())
acum_str = ""
for i in d["tgt"]:
    print(vocab[i],end=" ")

print()
