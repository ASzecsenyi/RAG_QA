from data.NewsQaDocument import NewsQaDocument

first_doc_id = './cnn/stories/289a45e715707cf650352f3eaa123f85d3653d4b.story'

document = NewsQaDocument(name="newsqa", story_id=first_doc_id)

print(document.questions)

all_documents = NewsQaDocument.all_documents()

print(len(all_documents))
