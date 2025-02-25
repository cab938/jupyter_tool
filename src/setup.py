from setuptools import setup, find_packages

# Helper function to read requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as req_file:
        # Filter out comments and empty lines
        return [line.strip() for line in req_file if line.strip() and not line.startswith("#")]

setup(
    name="jupyter_tool",
    version="0.1.0",
    description="A Python package providing atomic tools for langchain-based AI agents to manipulate Jupyter notebooks. Built on nbclient/nbformat, it enables programmatic notebook creation, loading, and manipulation.",
    author="Christopher Brooks",
    author_email="cab938@gmail.com",
    url="https://github.com/cab938/jupyter_tool/",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.10",
)