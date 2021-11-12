# Contributing guidelines

## How to become a contributor and submit your own code

### Contribution guidelines and standards

Before sending your pull request for review, make sure your changes are consistent with the guidelines and follow the coding style

#### General guidelines and philosophy for contribution

* Include unit tests when you contribute new features, as they help to a) prove that your code works correctly, and b) guard against future breaking changes to lower the maintenance cost.
* Bug fixes also generally require unit tests, because the presence of bugs usually indicates insufficient test coverage.


#### Python coding style
Use `pylint` to check your Python changes. To install `pylint`:

```bash
pip install pylint
```

To check files with `pylint`:

```bash
pylint --rcfile=./.dev/pylintrc kogpt
```

Expected result:
```

--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
```
