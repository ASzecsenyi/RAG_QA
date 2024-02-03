import time

from qa.GptQA import GptQA
from qa.MistralQA import MistralQA
from qa.LlamaQA import LlamaQA

start_time = time.time()

gpt_qa = GptQA(name="gptqa")
mi_qa = MistralQA(name="mistralqa")
lam_qa = LlamaQA(name="llamaqa")


question = "What is the full name of the main character?"

chunks = [
    "The main character is called Bob.",
    "He has a helicopter.",
    "His dad was called John Travolta.",
    "He took his father's last name.",
    "He is a pilot.",
]

# print('gpt', gpt_qa.predict(question, chunks))
print('mistral', mi_qa.predict(question, [""]))  # chunks))
# print('llama', lam_qa.predict(question, chunks))

print("--- %s seconds ---" % (time.time() - start_time))
