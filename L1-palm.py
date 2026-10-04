import os
import google.generativeai as genai

from google.api_core import client_options as client_options_lib
from utils import get_api_key


## ------------------------------------------------------ ##
genai.configure(api_key = get_api_key(),
                transport = "rest",
                client_options = client_options_lib.
                                    ClientOptions(api_endpoint = os.getenv("GOOGLE_API_BASE"), ))

## ------------------------------------------------------ ##
for m in genai.list_models():
    print(f"name: {m.name}")
    print(f"description: {m.description}")
    print(f"generation methods:{m.supported_generation_methods}\n")

## ------------------------------------------------------ ##
models = [m for m in genai.list_models() if 'generateText' in m.supported_generation_methods]
print(models)

## ------------------------------------------------------ ##
model_bison = models[0]
print(model_bison)

## ------------------------------------------------------ ##
model_flash = genai.GenerativeModel(model_name = 'gemini-1.5-flash')

## ------------------------------------------------------ ##
def generate_text(prompt, model = model_flash, temperature = 0.0):

    return model_flash.generate_content(prompt, generation_config = {'temperature' : temperature})

## ------------------------------------------------------ ##
prompt = "Show me how to iterate across a list in Python."

## ------------------------------------------------------ ##
completion = generate_text(prompt)

## ------------------------------------------------------ ##
print(completion.text)

## ------------------------------------------------------ ##
prompt = "write code to iterate across a list in Python"

## ------------------------------------------------------ ##
completion = generate_text(prompt)
print(completion.text)

## ------------------------------------------------------ ##
my_list = [10, 20, 30, 40, 50]

for item in my_list:
    print(item)

for i, item in enumerate(my_list):
    print(f"Index: {i}, Value: {item}")

my_list = [10, 20, 30, 40, 50]
i = 0

while i < len(my_list):
    print(my_list[i])
    i += 1

my_list = [10, 20, 30, 40, 50]

doubled_list = [item * 2 for item in my_list]
print(doubled_list)

even_numbers = [item for item in my_list if item % 2 == 0]
print(even_numbers)

my_list = [10, 20, 30, 40, 50]
my_iterator = iter(my_list)

try:
    while True:
        item = next(my_iterator)
        print(item)

except StopIteration:
    pass

## ------------------------------------------------------ ##
prompt = "Show me how to do dictionary comprehention"

completion = generate_text(prompt)

## ------------------------------------------------------ ##
print(completion.text)
