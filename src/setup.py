from setuptools import setup, find_packages

# with open("requirements.txt") as f:
#     required = f.read().splitlines()
#     print("Required packages:", required)


deps=[
    "nbclient",
    "nbformat",
    "pydantic>=2.0.0",
    "requests",
    "langchain>=0.3.0",
    "boto3",
    "langchain-aws>=0.2.12",
    "langchain-community>=0.3.16",
    "langchain-core>=0.3.33",
    "ipykernel",
    "langgraph",
    "langfuse",
    "setuptools",
    "build"
]

setup(
    name="jupyter_tool",
    version="0.1.1",
    description="A Python package providing atomic tools for langchain-based AI agents to manipulate Jupyter notebooks. Built on nbclient/nbformat, it enables programmatic notebook creation, loading, and manipulation.",
    author="Christopher Brooks",
    author_email="cab938@gmail.com",
    url="https://github.com/cab938/jupyter_tool/",
    packages=find_packages(),
    install_requires=deps,
    python_requires=">=3.10",
)
