# PolarBtest - Agent Guidelines

This repository contains PolarBtest, a hybrid backtesting engine combining vectorized preprocessing (Polars) with event-driven execution for trading strategies. Follow these guidelines when working with the codebase.

## Architecture Philosophy

- **Performance**: All heavy computation is vectorized using Polars
- **Flexibility**: Event-driven loop allows complex conditional logic
- **Scalability**: Multi-core execution for testing thousands of strategies
- **Simplicity**: Clean API

## Adding new features

### Plan the feature

Before proceeding with the implementation write out the plan, todo list as well as how all new classes, functions and promises between them should look like.

### Code style

Find comprehensive code and project structure explanation in `agents/CODE_STYLE.md`

- Avoid extra comments that a human wouldn't add or are inconsistent with the rest of the file
- Extra defensive checks or try/catch blocks that are abnormal for that area of the codebase (especially if called by trusted/validated codepaths)
- Casts to any to get around type issues
- Inline imports in Python (move to top of file with other imports)
- Any other style that is inconsistent with the file

### Found issues during development

If during development you encountered a potential problem or bug, and it doesn't affect the current feature, then note your findings in `POTENTIAL_ISSUES.md`.

If there is a misalignment in promises between classes/functions or there is incorrect information written in readme or documentation, then ask user which implementation is correct one and fix it.

If the issue you encountered is big, stop the implementation, write the outline of a problem and ask user how to proceed. If it is small, then fix it on the spot.

### Testing

During and after implementing a feature run the test to verify that everything works correctly. Add tests for the new feature, they should be comprehensive but, don't create redundancy if the feature could be tested with additional assertion in an already implemented test do that. Run test with:
```bash
pytest tests/
```

### Code review

After adding a feature analyze the code changes based on the following pillars:

- **Correctness**: Does the code achieve its stated purpose without bugs or logical errors?
- **Maintainability**: Is the code clean, well-structured, and easy to understand and modify in the future? Consider factors like code clarity, modularity, and adherence to established design patterns.
- **Readability**: Is the code well-commented (where necessary) and consistently formatted according to our project's coding style guidelines?
- **Efficiency**: Are there any obvious performance bottlenecks or resource inefficiencies introduced by the changes?
- **Security**: Are there any potential security vulnerabilities or insecure coding practices?
- **Edge Cases and Error Handling**: Does the code appropriately handle edge cases and potential errors?
- **Testability**: Is the new or modified code adequately covered by tests (even if preflight checks pass)? Suggest additional test cases that would improve coverage or robustness.

### Cleanup

After finishing implementing, testing and reviewing the feature, always check if readme, agents, documentation and examples need to be updated. Put the summary of your work in the archive folder.

## Module Structure

```
polarbtest/
├── core.py           # Portfolio, Strategy, Engine, BacktestContext
├── indicators.py     # Technical indicators as Polars expressions
├── metrics.py        # Performance metrics calculation
└── runner.py         # Parallel execution and optimization
```

## Additional Documentation

- See `IMPLEMENTATION.md` for detailed API reference, code patterns, and troubleshooting
- See `README.md` for installation and quick start guide
- See `agents/CODE_STYLE.md` for detailed coding style guide
- Check source code docstrings for detailed method documentation
