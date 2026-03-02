# PolarBT - Agent Guidelines

This repository contains PolarBT, a hybrid backtesting engine combining vectorized preprocessing (Polars) with event-driven execution for trading strategies. Follow these guidelines when working with the codebase.

## Architecture Philosophy

- **Performance**: All heavy computation is vectorized using Polars
- **Flexibility**: Event-driven loop allows complex conditional logic
- **Scalability**: Multi-core execution for testing thousands of strategies
- **Simplicity**: Clean API

## Adding new features

### Plan the feature

Before proceeding with the implementation write out the plan, todo list as well as how all new classes, functions and promises between them should look like.

### Code style

- use `uv` if you need interaction with venv or pip
- Avoid extra comments that a human wouldn't add or are inconsistent with the rest of the file
- Extra defensive checks or try/catch blocks that are abnormal for that area of the codebase (especially if called by trusted/validated codepaths)
- Casts to any to get around type issues
- Inline imports in Python (move to top of file with other imports)
- Any other style that is inconsistent with the file

### Problem solving

If there are different ways to solve a particular problem always prefer more forward thinking ones, and the ones that solve problems now instead of avoiding them. If it is a major decision prefer asking the user for feedback.

### Found issues during development

If during development you encountered a potential problem or bug, and it doesn't affect the current feature, then note your findings in `POTENTIAL_ISSUES.md`.

If there is a misalignment in promises between classes/functions or there is incorrect information written in readme or documentation, then ask user which implementation is correct one and fix it.

If the issue you encountered is big, stop the implementation, write the outline of a problem and ask user how to proceed. If it is small, then fix it on the spot.

### Testing

During and after implementing a feature run the test to verify that everything works correctly. Add tests for the new feature, they should be comprehensive but, don't create redundancy if the feature could be tested with additional assertion in an already implemented test do that. Run test with:
```bash
pytest tests/ -q
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

After finishing implementing, testing and reviewing the feature, always:
1. Run ruff and mypy, fix all issues, even not related to what you worked on
2. Put the summary of your work in the archive folder
3. Commit changes

## Module Structure

```
polarbt/
├── core.py           # Portfolio, Strategy, Engine, BacktestContext
├── orders.py         # Order, OrderType, OrderStatus
├── trades.py         # Trade, TradeTracker
├── indicators.py     # Technical indicators as Polars expressions
├── metrics.py        # Performance metrics calculation
├── runner.py         # Parallel execution and optimization
├── sizers.py         # Position sizing strategies
├── commissions.py    # Commission models
├── analysis.py       # Trade and performance analysis
├── data/
│   ├── cleaning.py   # Data cleaning utilities
│   ├── validation.py # Data validation
│   └── resampling.py # OHLCV resampling
├── plotting/
│   └── charts.py     # Charting and visualization
└── integrations/
    └── talib.py      # TA-Lib integration
```

## Additional Documentation

- See `DESCRIPTION.md` for library description, API reference, and current capabilities
- See `PLAN.md` for remaining work and development roadmap
- Check source code docstrings for detailed method documentation

# Code style

## Core Concepts

### 1. Automated Formatting

Let tools handle formatting debates. Configure once, enforce automatically.

### 2. Consistent Naming

Follow PEP 8 conventions with meaningful, descriptive names.

### 3. Documentation as Code

Docstrings should be maintained alongside the code they describe.

### 4. Type Annotations

Modern Python code should include type hints for all public APIs.

## Quick Start

```bash
# Install modern tooling
pip install ruff mypy

# Configure in pyproject.toml
[tool.ruff]
line-length = 120
target-version = "py312"  # Adjust based on your project's minimum Python version

[tool.mypy]
strict = true
```

## Fundamental Patterns

### Pattern 1: Modern Python Tooling

Use `ruff` as an all-in-one linter and formatter. It replaces flake8, isort, and black with a single fast tool.

```toml
# pyproject.toml
[tool.ruff]
line-length = 120
target-version = "py312"  # Adjust based on your project's minimum Python version

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
    "SIM",  # flake8-simplify
]
ignore = ["E501"]  # Line length handled by formatter

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

Run with:

```bash
ruff check --fix .  # Lint and auto-fix
ruff format .       # Format code
```

### Pattern 2: Type Checking Configuration

Configure strict type checking for production code.

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_incomplete_defs = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

Alternative: Use `pyright` for faster checking.

```toml
[tool.pyright]
pythonVersion = "3.12"
typeCheckingMode = "strict"
```

### Pattern 3: Naming Conventions

Follow PEP 8 with emphasis on clarity over brevity.

**Files and Modules:**

```python
# Good: Descriptive snake_case
user_repository.py
order_processing.py
http_client.py

# Avoid: Abbreviations
usr_repo.py
ord_proc.py
http_cli.py
```

**Classes and Functions:**

```python
# Classes: PascalCase
class UserRepository:
    pass

class HTTPClientFactory:  # Acronyms stay uppercase
    pass

# Functions and variables: snake_case
def get_user_by_email(email: str) -> User | None:
    retry_count = 3
    max_connections = 100
```

**Constants:**

```python
# Module-level constants: SCREAMING_SNAKE_CASE
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT_SECONDS = 30
API_BASE_URL = "https://api.example.com"
```

### Pattern 4: Import Organization

Group imports in a consistent order: standard library, third-party, local.

```python
# Standard library
import os
from collections.abc import Callable
from typing import Any

# Third-party packages
import httpx
from pydantic import BaseModel
from sqlalchemy import Column

# Local imports
from myproject.models import User
from myproject.services import UserService
```

Use absolute imports exclusively:

```python
# Preferred
from myproject.utils import retry_decorator

# Avoid relative imports
from ..utils import retry_decorator
```

## Advanced Patterns

### Pattern 5: Google-Style Docstrings

Write docstrings for all public classes, methods, and functions.

**Simple Function:**

```python
def get_user(user_id: str) -> User:
    """Retrieve a user by their unique identifier."""
    ...
```

**Complex Function:**

```python
def process_batch(
    items: list[Item],
    max_workers: int = 4,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """Process items concurrently using a worker pool.

    Processes each item in the batch using the configured number of
    workers. Progress can be monitored via the optional callback.

    Args:
        items: The items to process. Must not be empty.
        max_workers: Maximum concurrent workers. Defaults to 4.
        on_progress: Optional callback receiving (completed, total) counts.

    Returns:
        BatchResult containing succeeded items and any failures with
        their associated exceptions.

    Raises:
        ValueError: If items is empty.
        ProcessingError: If the batch cannot be processed.

    Example:
        >>> result = process_batch(items, max_workers=8)
        >>> print(f"Processed {len(result.succeeded)} items")
    """
    ...
```

**Class Docstring:**

```python
class UserService:
    """Service for managing user operations.

    Provides methods for creating, retrieving, updating, and
    deleting users with proper validation and error handling.

    Attributes:
        repository: The data access layer for user persistence.
        logger: Logger instance for operation tracking.

    Example:
        >>> service = UserService(repository, logger)
        >>> user = service.create_user(CreateUserInput(...))
    """

    def __init__(self, repository: UserRepository, logger: Logger) -> None:
        """Initialize the user service.

        Args:
            repository: Data access layer for users.
            logger: Logger for tracking operations.
        """
        self.repository = repository
        self.logger = logger
```

### Pattern 6: Line Length and Formatting

Set line length to 120 characters for modern displays while maintaining readability.

```python
# Good: Readable line breaks
def create_user(
    email: str,
    name: str,
    role: UserRole = UserRole.MEMBER,
    notify: bool = True,
) -> User:
    ...

# Good: Chain method calls clearly
result = (
    db.query(User)
    .filter(User.active == True)
    .order_by(User.created_at.desc())
    .limit(10)
    .all()
)

# Good: Format long strings
error_message = (
    f"Failed to process user {user_id}: "
    f"received status {response.status_code} "
    f"with body {response.text[:100]}"
)
```

### Pattern 7: Project Documentation

**README Structure:**

```markdown
# Project Name

Brief description of what the project does.

## Installation

\`\`\`bash
pip install myproject
\`\`\`

## Quick Start

\`\`\`python
from myproject import Client

client = Client(api_key="...")
result = client.process(data)
\`\`\`

## Configuration

Document environment variables and configuration options.

## Development

\`\`\`bash
pip install -e ".[dev]"
pytest
\`\`\`
```

**CHANGELOG Format (Keep a Changelog):**

```markdown
# Changelog

## [Unreleased]

### Added
- New feature X

### Changed
- Modified behavior of Y

### Fixed
- Bug in Z
```

Keep it short and simple, no elaboration, code examples etc.

## Best Practices Summary

1. **Use ruff** - Single tool for linting and formatting
2. **Enable strict mypy** - Catch type errors before runtime
3. **120 character lines** - Modern standard for readability
4. **Descriptive names** - Clarity over brevity
5. **Absolute imports** - More maintainable than relative
6. **Google-style docstrings** - Consistent, readable documentation
7. **Document public APIs** - Every public function needs a docstring
8. **Keep docs updated** - Treat documentation as code
9. **Automate in CI** - Run linters on every commit
10. **Target Python 3.10+** - For new projects, Python 3.12+ is recommended for modern language features

