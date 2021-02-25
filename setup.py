import setuptools

setuptools.setup(
    name="Project2", # Replace with your own username
    version="1.0",
    author="Anders Bjelland",
    author_email="anders.bjelland@hotmail.com",
    description="MCTS for playing Hex",
    long_description='missing',
    long_description_content_type="text/markdown",
    url="https://github.com/AndersBjelland/IT3105_Project2.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)