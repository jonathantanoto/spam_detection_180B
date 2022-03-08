import src.generate_dataset as generate
import src.process_build as build
import src.spam_or_not as test

import sys
import json


def main():
    """
    main executable with optional args
    :arg: data: downloads data and transform into dataframe
    :arg: build: preprocess data and build model
    :arg: test: runs a predicting script with user input
    """
    argv = sys.argv

    # Downloads data and transform into dataframe
    if "data" in argv:
        generate.generate()
    # Preprocess data and build model
    if "build" in argv:
        build.build()
    # Runs a predicting script with user input
    if "test" in argv:
        test.test()

if __name__ == "__main__":
    main()