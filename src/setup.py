from setuptools import setup, find_packages

setup(
    name="jupyter_tool",
    version="0.1.0",
    description="A Python package providing atomic tools for langchain-based AI agents to manipulate Jupyter notebooks. Built on nbclient/nbformat, it enables programmatic notebook creation, loading, and manipulation.",
    author="Christopher Brooks",
    author_email="cab938@gmail.com",
    url="https://github.com/cab938/jupyter_tool/",
    packages=find_packages(),
    install_requires="requirements.txt",
    python_requires=">=3.10",
)