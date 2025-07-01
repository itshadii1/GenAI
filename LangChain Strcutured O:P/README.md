Use `TypedDict` if:

* You only need type hints (basic structure enforcement).
* You don’t need validation (e.g., checking numbers are positive).
* You trust the LLM to return correct data.

Use `Pydantic` if:

* You need data validation (e.g., sentiment must be `"positive"`, `"neutral"`, or `"negative"`).
* You need default values if the LLM misses fields.
* You want automatic type conversion (e.g., `"100"` → `100`).

Use `JSON Schema` if:

* You don’t want to import extra Python libraries (Pydantic).
* You need validation but don’t need Python objects.
* You want to define structure in a standard JSON format.


