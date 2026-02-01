# Potential Issues

This file tracks potential bugs and issues discovered during development that don't affect current features but should be addressed in the future.

## Type Safety Issues

### 1. Portfolio.update_prices Type Mismatch (polarbtest/core.py:750)

**Location:** `polarbtest/core.py:750`

**Issue:** Argument type mismatch in `update_prices` call
```python
ERROR [750:42] Argument of type "dict[str, Any | None]" cannot be assigned to parameter "prices" of type "Dict[str, float]"
```

**Details:**
- The `update_prices` method expects `Dict[str, float]`
- Receiving `dict[str, Any | None]` instead
- Type parameter is invariant, creating incompatibility

**Suggested Fix:**
- Consider changing the type signature to use `Mapping[str, float]` which is covariant
- Or add proper validation/filtering to ensure None values are handled before calling update_prices
- Or update the function signature to accept optional values: `Dict[str, float | None]`

### 2. DataFrame Slicing Type Error (polarbtest/runner.py:434-435)

**Location:** `polarbtest/runner.py:434-435`

**Issue:** Invalid slice usage on dictionary-like object
```python
ERROR [434:22] Argument of type "slice[int, int, None]" cannot be assigned to parameter "key" of type "str"
ERROR [435:21] Argument of type "slice[int, int, None]" cannot be assigned to parameter "key" of type "str"
```

**Details:**
- Attempting to use slice notation (e.g., `[0:100]`) on an object that expects string keys
- Likely trying to slice a Polars DataFrame or similar structure incorrectly

**Suggested Fix:**
- Review the slicing logic at these lines
- Use proper Polars DataFrame slicing methods (`.slice()`, `.head()`, `.tail()`)
- Or use integer-based indexing if working with sequences

### 3. None Comparison in Multi-Asset Example (examples/example_multi_asset.py:48)

**Location:** `examples/example_multi_asset.py:48`

**Issue:** Invalid operator usage with None type
```python
ERROR [48:12] Operator ">" not supported for "None"
```

**Details:**
- Attempting to use comparison operator on a value that could be None
- This will raise a runtime error if None is encountered

**Suggested Fix:**
- Add None check before comparison
- Use proper optional handling: `if value is not None and value > threshold:`
- Or provide default value: `(value or 0) > threshold`

## Impact Assessment

**Priority:** Medium
- These issues are flagged by MyPy strict mode
- May not cause runtime errors in current usage patterns
- Should be addressed before production use or publishing

**Next Steps:**
1. Run `mypy polarbtest` to get full context
2. Fix type annotations to match actual usage
3. Add tests to cover edge cases (None values, empty data, etc.)
4. Consider adding runtime validation where type safety is critical
