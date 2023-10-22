import time

from qa.GptQA import GptQA

start_time = time.time()

qa = GptQA(name="gptqa")

question = "What is the full name of the main character?"

chunks = [
    "The main character is called Bob.",
    "He has a helicopter.",
    "His dad was called John Travolta.",
    "He took his father's last name.",
    "He is a pilot.",
]

print(qa.predict(question, chunks))

print("--- %s seconds ---" % (time.time() - start_time))
