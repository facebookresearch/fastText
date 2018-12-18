# Contributing to fastText
We want to make contributing to this project as easy and transparent as possible.

## Issues
We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.

### Reproducing issues
Please make sure that the issue you mention is not a result of one of the existing third-party libraries. For example, please do not post an issue if you encountered an error within a third-party Python library. We can only help you with errors which can be directly reproduced either with our C++ code or the corresponding Python bindings. If you do find an error, please post detailed steps to reproduce it. If we can't reproduce your error, we can't help you fix it.

## Pull Requests
Please post an Issue before submitting a pull request. This might save you some time as it is possible we can't support your contribution, albeit we try our best to accomodate your (planned) work and highly appreciate your time. Generally, it is best to have a pull request emerge from an issue rather than the other way around.

To create a pull request:

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Tests
First, you will need to make sure you have the required data. For that, please have a look at the fetch_test_data.sh script under tests. Next run the tests using the runtests.py script passing a path to the directory containing the datasets.

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## License
By contributing to fastText, you agree that your contributions will be licensed under its MIT license.
