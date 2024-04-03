from transformers import AutoModelForCausalLM, AutoTokenizer
import os

home_directory = os.getenv('HOME')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
tokenizer.save_pretrained("/mnt/nfs-mnj-home-43/i23_eric/code-server/userdata/Mixtral/tokenizer_model_directory")