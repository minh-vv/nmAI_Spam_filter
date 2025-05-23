# setup.py


import setuptools

from spam_detector_ai import (
    __api_url__, __author__, __author_email__, __author_website__, __description__, __package_name__, __url__,
    __version__)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name=__package_name__,
    version=__version__,
    author=__author__,
    author_email=__author_email__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=__url__,
    description=__description__,
    project_urls={
        "API Documentation": f"{__api_url__}/redoc/",
        "Author's Website": __author_website__,
        "Bug Tracker": f"{__url__}/issues",
        "Contact Page": f"{__author_website__}contact/",
    },
)
