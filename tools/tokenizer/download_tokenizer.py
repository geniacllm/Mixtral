from transformers import AutoModelForCausalLM, AutoTokenizer
import os

home_directory = os.getenv('HOME')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
tokenizer.save_pretrained(home_directory+"/moe-recipes/tokenizer_model_directory")