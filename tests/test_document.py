import pytest
from data import Document
import json


def test_document_init():
    questions = [{"question": "Whodunnit?", "ground_truths": "The butler."}]
    doc = Document("this is a document", "doc1", questions)
    assert doc.name == "doc1"
    assert doc.document == "this is a document"
    assert doc.questions == questions


def test_document_init_no_name():
    questions = [{"question": "Whodunnit?", "ground_truths": "The butler."}]
    doc = Document("this is an important document and it has a couple of words", None, questions)
    assert doc.name == "this_is_an_important_document"
    assert doc.document == "this is an important document and it has a couple of words"
    assert doc.questions == questions


def test_document_repr_str():
    questions = [{"question": "Whodunnit?", "ground_truths": "The butler."}]
    doc = Document("this is a document", "doc1", questions)
    expected_str = f"Document(name=doc1, document=this is a document, questions={questions})"
    assert str(doc) == expected_str
    assert repr(doc) == expected_str


def test_document_from_json(tmpdir):
    questions = [{"question": "Who did it?", "ground_truths": "The butler."}]
    data = {"name": "doc1", "document": "this is a document", "questions": questions}
    file_path = tmpdir.join("file.json")
    with open(file_path, "w") as f:
        json.dump(data, f)
    doc = Document.from_json(file_path)
    assert doc.name == "doc1"
    assert doc.document == "this is a document"
    assert doc.questions == questions


@pytest.mark.parametrize("file_path,name,document,questions", [
    (None, "doc1", "this is a doc", [{"question": "Who did it?", "ground_truths": "The butler."}])])
def test_document_save(tmpdir, file_path, name, document, questions):
    if file_path is None:
        file_path = tmpdir.join("file.json")
    doc = Document(document, name, questions)
    doc.save(file_path)
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data["name"] == name
    assert data["document"] == document
    assert data["questions"] == questions


def test_document_len():
    questions = [{"question": "Who did it?", "ground_truths": "The butler."}]
    doc = Document("this is a document", "doc1", questions)
    assert len(doc) == 18
