import pytest
import json
from gazebo_world_generator.src.utils.llm_utils import extract_json_from_response, _sanitize_json_string

def test_extract_json_from_response_markdown():
    response = "Sure, here is your json: ```json\n{\"id\": 1, \"name\": \"test\"}\n``` hope it helps!"
    data = extract_json_from_response(response)
    assert data == {"id": 1, "name": "test"}

def test_extract_json_from_response_garbage():
    response = "Something before [{\"a\": 1}] something after"
    data = extract_json_from_response(response)
    assert data == [{"a": 1}]

def test_extract_json_from_response_nested():
    response = "Nested brackets { \"a\": { \"b\": [1,2,3] } } trailing text"
    data = extract_json_from_response(response)
    # The current extractor might be greedy or specific. 
    # Let's check what it actually returns.
    assert data["a"]["b"] == [1, 2, 3]

def test_sanitize_json_string():
    # Fix trailing commas
    dirty = '{"a": 1,}'
    assert _sanitize_json_string(dirty) == '{"a": 1}'
    
    # Fix unquoted keys if handled (let's see)
    # Actually json5 is used in extract_json_from_response usually
    pass

def test_extract_json_from_response_wrapped_lines():
    # Simulate LLM wrapping a long string
    response = """[
        {
            "description": "A very long
description that was wrapped"
        }
    ]"""
    data = extract_json_from_response(response)
    assert data[0]["description"] == "A very long description that was wrapped"
