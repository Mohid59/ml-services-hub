"""Shared pytest fixtures for ML Services Hub."""
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app  # noqa: E402


@pytest.fixture(scope="session")
def app():
    application = create_app()
    application.config.update(TESTING=True)
    return application


@pytest.fixture(scope="session")
def client(app):
    return app.test_client()


@pytest.fixture()
def jpeg_bytes():
    """A small in-memory JPEG for image-upload tests."""
    import numpy as np
    from PIL import Image

    im = Image.fromarray((np.random.rand(70, 70, 3) * 255).astype("uint8"))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    buf.seek(0)
    return buf
