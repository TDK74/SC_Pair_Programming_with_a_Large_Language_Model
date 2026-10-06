import os
import random
import google.generativeai as genai

from google.api_core import client_options as client_options_lib
from utils import get_api_key


## ------------------------------------------------------ ##
genai.configure(api_key= get_api_key(),
                transport= "rest",
                client_options= client_options_lib.
                                    ClientOptions(api_endpoint= os.getenv("GOOGLE_API_BASE"), ))

## ------------------------------------------------------ ##
models = [m for m in genai.list_models() if 'generateText' in m.supported_generation_methods]
model_bison = models[0]
model_bison

## ------------------------------------------------------ ##
model_flash = genai.GenerativeModel(model_name= 'gemini-1.5-flash')

## ------------------------------------------------------ ##
def generate_text(prompt, model= model_flash, temperature= 0.0):
    return model_flash.generate_content(prompt, generation_config= {'temperature' : temperature})

## ------------------------------------------------------ ##
prompt_template = """{priming}

                    {question}

                    {decorator}

                    Your solution:
                    """

## ------------------------------------------------------ ##
priming_text = "You are an expert at writing clear, concise, Python code."

## ------------------------------------------------------ ##
question = "Create a doubly linked list"

## ------------------------------------------------------ ##
# decorator = "Work through it step by step, and show your work. One step per line."

decorator = "Insert comments for each line of code."

## ------------------------------------------------------ ##
prompt = prompt_template.format(priming= priming_text, question= question, decorator= decorator)

## ------------------------------------------------------ ##
print(prompt)

## ------------------------------------------------------ ##
completion = generate_text(prompt)

print(completion.text)

## ------------------------------------------------------ ##
question = """Create a very large list of random numbers in Python,
             and then write code to sort that list"""

## ------------------------------------------------------ ##
print(prompt)

## ------------------------------------------------------ ##
completion = generate_text(prompt)
print(completion.text)

## ------------------------------------------------------ ##
random_numbers = [random.randint(0, 100) for _ in range(100000)]
# print(random_numbers)
random_numbers.sort()
print(f"First 10 sorted numbers: {random_numbers[ : 10]}")
print(f"Last 10 sorted numbers: {random_numbers[-10 : ]}")

## ------------------------------------------------------ ##
random_numbers = [random.randint(0, 100) for _ in range(100000)]
random_numbers.sort()
print(f"First 10 sorted numbers: {random_numbers[ : 10]}")
print(f"Last 10 sorted numbers: {random_numbers[-10 : ]}")
