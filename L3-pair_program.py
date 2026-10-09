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
models = [m for m in genai.list_models() if 'generateText' in m.supported_generation_methods]
model_bison = models[0]
model_bison

## ------------------------------------------------------ ##
model_flash = genai.GenerativeModel(model_name = 'gemini-1.5-flash')

## ------------------------------------------------------ ##
def generate_text(prompt, model = model_flash, temperature = 0.0):
    return model_flash.generate_content(prompt, generation_config = {'temperature' : temperature})

## ------------------------------------------------------ ##
prompt_template = """ I don't think this code is the best way to do it in Python, can you help me?

                    {question}

                    Please explain, in detail, what you did to improve it.
                """

## ------------------------------------------------------ ##
question = """  def func_x(array):
                    for i in range(len(array)):
                        print(array[i])
        """

## ------------------------------------------------------ ##
completion = generate_text(prompt = prompt_template.format(question = question))

print(completion.text)

## ------------------------------------------------------ ##
prompt_template = """ I don't think this code is the best way to do it in Python, can you help me?

                    {question}

                    Please explore multiple ways of solving the problem, and explain each.
                """

## ------------------------------------------------------ ##
completion = generate_text(prompt = prompt_template.format(question = question))

print(completion.text)

## ------------------------------------------------------ ##
prompt_template = """ I don't think this code is the best way to do it in Python, can you help me?

                    {question}

                    Please explore multiple ways of solving the problem, and tell me which is \
                    the most Pythonic.
                """

## ------------------------------------------------------ ##
completion = generate_text(prompt = prompt_template.format(question = question))

print(completion.text)

## ------------------------------------------------------ ##
prompt_template = """ Can you please simplify this code for a linked list in Python?

                    {question}

                    Explain in detail what you did to modify it, and why.
                """

## ------------------------------------------------------ ##
question = """  class Node:
                    def __init__(self, dataval = None):
                        self.dataval = dataval
                        self.nextval = None

                class SLinkedList:
                    def __init__(self):
                        self.headval = None

                list1 = SLinkedList()
                list1.headval = Node("Mon")
                e2 = Node("Tue")
                e3 = Node("Wed")
                list1.headval.nextval = e2
                e2.nextval = e3
        """

## ------------------------------------------------------ ##
completion = generate_text(prompt = prompt_template.format(question = question)
                )
print(completion.text)

## ------------------------------------------------------ ##
prompt_template = """ Can you please create test cases in code for this Python code?

                    {question}

                    Explain in detail what these test cases are designed to achieve.
                """

## ------------------------------------------------------ ##
question = """  class Node:
                    def __init__(self, dataval = None):
                        self.dataval = dataval
                        self.nextval = None

                class SLinkedList:
                    def __init__(self):
                        self.head = None

                def create_linked_list(data):
                    head = Node(data[0])
                    for i in range(1, len(data)):
                        node = Node(data[i])
                        node.nextval = head
                        head = node
                    return head

                list1 = create_linked_list(["Mon", "Tue", "Wed"])
        """

## ------------------------------------------------------ ##
completion = generate_text(prompt = prompt_template.format(question = question))

print(completion.text)

## ------------------------------------------------------ ##
prompt_template = """ Can you please make this code more efficient?

                    {question}

                    Explain in detail what you changed and why.
                """

## ------------------------------------------------------ ##
question = """
            # Returns index of x in arr if present, else -1
            def binary_search(arr, low, high, x):
                # Check base case
                if high >= low:
                    mid = (high + low) // 2

                    if arr[mid] == x:
                        return mid

                    elif arr[mid] > x:
                        return binary_search(arr, low, mid - 1, x)

                    else:
                        return binary_search(arr, mid + 1, high, x)

                else:
                    return -1

            # Test array
            arr = [2, 3, 4, 10, 40]
            x = 10

            # Function call
            result = binary_search(arr, 0, len(arr) - 1, x)

            if result != -1:
                print("Element is present at index", str(result))
            else:
                print("Element is not present in array")

        """

## ------------------------------------------------------ ##
completion = generate_text(prompt = prompt_template.format(question = question))

print(completion.text)

## ------------------------------------------------------ ##
def binary_search_iterative(arr, x):
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = low + (high - low) // 2

        if arr[mid] == x:
            return mid

        elif arr[mid] < x:
            low = mid + 1

        else:
            high = mid - 1

    return -1


arr = [2, 3, 4, 10, 40]
x = 40

result = binary_search_iterative(arr, x)

if result != -1:
    print("Element is present at index", str(result))
else:
    print("Element is not present in array")

## ------------------------------------------------------ ##
prompt_template = """ Can you please help me to debug this code?

                    {question}

                    Explain in detail what you found and why it was a bug.
                """

## ------------------------------------------------------ ##
question = """  class Node:
                    def __init__(self, data):
                        self.data = data
                        self.next = None
                        self.prev = None

                class doubly_linked_list:
                    def __init__(self):
                        self.head = None

                # Adding data elements
                def push(self, NewVal):
                    NewNode = Node(NewVal)
                    NewNode.next = self.head

                    if self.head is not None:
                        self.head.prev = NewNode

                    self.head = NewNode

                # Print the Doubly Linked list in order
                def listprint(self, node):
                    print(node.data),
                    last = node
                    node = node.next

                dllist = doubly_linked_list()
                dllist.push(12)
                dllist.push(8)
                dllist.push(62)
                dllist.listprint(dllist.head)
        """

## ------------------------------------------------------ ##
completion = generate_text(prompt = prompt_template.format(question = question), temperature = 0.7)

print(completion.text)

## ------------------------------------------------------ ##
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


class doubly_linked_list:
    def __init__(self):
        self.head = None

    def push(self, NewVal):
        NewNode = Node(NewVal)
        NewNode.next = self.head

        if self.head is not None:
            self.head.prev = NewNode

        self.head = NewNode

    def listprint(self):
        node = self.head

        while node is not None:
            print(node.data, end = " ")
            node = node.next

        print()


dllist = doubly_linked_list()
dllist.push(12)
dllist.push(8)
dllist.push(62)
dllist.listprint()
