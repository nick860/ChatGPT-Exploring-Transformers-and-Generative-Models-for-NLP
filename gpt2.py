from mingpt.model import GPT
from mingpt.trainer import Trainer
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer

def generate(prompt, num_samples=10, steps=20, do_sample=True):
        

    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    if prompt == '': 
        # to create unconditional samples...
        # huggingface/transformers tokenizer special cases these strings
        prompt = '<|endoftext|>'
    encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
    x = encoded_input['input_ids']
    
    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)
    
    for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        print('-'*80)
        print(out)
        
set_seed(42)
use_mingpt = False # use minGPT or huggingface/transformers model?
model_type = 'gpt2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("preparing model...")
model = GPT2LMHeadModel.from_pretrained(model_type)
model.config.pad_token_id = model.config.eos_token_id # suppress a warning

model.to(device)
model.eval()

prefixes = ["My name", "Machine learning", "Buy me", "Your cat", "Artificial intelligence", 
            "I need", "Be aware", "Tomorrow will", "Next month", "Great course"]

for prefix in prefixes:
    print(f"prefix: {prefix}")
    generate(prompt=prefix, num_samples=5, steps=15)
