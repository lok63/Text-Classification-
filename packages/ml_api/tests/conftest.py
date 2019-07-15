import pytest

from api.app import create_app
from api.config import TestingConfig


"""
Used by PyTest to create setup functions 
Used to pass the fixtures to the test
Create test instances of our flask app
"""

@pytest.fixture
def app():
    app = create_app(config_object=TestingConfig)

    with app.app_context():
        yield app


@pytest.fixture
def flask_test_client(app):
    with app.test_client() as test_client:
        yield test_client
