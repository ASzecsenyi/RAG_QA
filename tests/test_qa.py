import pytest
from qa.MistralQA import MistralQA  # replace with your actual module name
from unittest import mock


def test_mistralqa_init(mocker):
    mocker.patch.dict("os.environ", {"HUGGINGFACE_API_KEY": "test_key"})
    r = MistralQA('test', 'my_key')
    assert r.name == "test_y_key"

    r = MistralQA('test')
    assert r.name == "test_t_key"


@mock.patch('requests.post')
def test_mistralqa_predict(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = [{'generated_text': '[/INST] Test output'}]

    r = MistralQA('test', 'my_key')
    answer = r.predict('Who was the first president of the United States?', ['George Washington was.'])
    assert answer == 'Test output'
