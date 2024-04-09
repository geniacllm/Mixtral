import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# read path from std input
path = sys.argv[1]

home_directory = os.getenv('HOME')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
tokenizer.save_pretrained(path)