import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cyy_torch_vision",
    author="cyy",
    version="0.1",
    author_email="cyyever@outlook.com",
    long_description=long_description,
    long_description_content_type="vision/markdown",
    url="https://github.com/cyyever/torch_vision",
    packages=[
        "cyy_torch_vision",
        "cyy_torch_vision/dataset",
        "cyy_torch_vision/data_pipeline",
        "cyy_torch_vision/model",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
