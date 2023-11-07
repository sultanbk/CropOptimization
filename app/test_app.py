# test_app.py

from flask import Flask
from app import app  # assuming your Flask app is named "app" in your app.py file

def test_create_app():
    assert isinstance(app, Flask)