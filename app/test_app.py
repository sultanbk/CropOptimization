import pytest
from flask import Flask
from app import app

@pytest.fixture
def client():
    """
    Create a test client for the Flask app.
    """
    with app.test_client() as client:
        yield client

def test_create_app_instance():
    """
    Test if the app variable is an instance of Flask.
    """
    assert isinstance(app, Flask)

def test_app_configuration(client):
    """
    Test specific configuration settings of the Flask app.
    """
    assert app.debug is False  # Adjust based on your app's configuration
    assert app.config['TESTING'] is True

def test_home_route(client):
    """
    Test the home route of the Flask app.
    """
    response = client.get('/')
    assert response.status_code == 200
    assert b'Hello, World!' in response.data  # Adjust based on your app's response
