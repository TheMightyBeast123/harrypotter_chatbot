
import os
import kagglehub
from transformers import GPT2Tokenizer
path = kagglehub.dataset_download("moxxis/harry-potter-lstm")
file_path = os.path.join(path, 'Harry_Potter_all_books_preprocessed.txt')
inpath = file_path
outpath = "../data/tokenized_dataset.txt"

# Loading tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
f = open(inpath, 'r', encoding='utf-8')
text = f.read()
f.close()
text = text.replace('\n', ' ')
text = text.replace('"', '')
text = text.strip()
token_strings = []
tokens = tokenizer.encode(text)
for token in tokens:
    token_strings.append(str(token))
joined_text = ' '.join(token_strings)
f = open("output.txt", "w")
f.write(joined_text)
f.close()

