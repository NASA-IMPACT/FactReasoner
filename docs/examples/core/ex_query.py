# This is a simple example

from mellea.backends import ModelOption

# Local imports
from src.fact_reasoner.core.query_builder import QueryBuilder

# Create a Mellea RITS backend
from mellea.backends.ollama import OllamaBackend
from mellea.stdlib.base import Context, Component

MODEL_NAME = os.get("MODEL_NAME", "llama3")
backend = OllamaBackend(model_id=MODEL_NAME)

# Create the query builder
qb = QueryBuilder(backend)

# Process a single atom (no knowledge)
# text = "The Apollo 14 mission to the Moon took place on January 31, 1971."
# text = "You'd have to yell if your friend is outside the same location"
text = "rootstock for honey crisp apples in wayne county, ny"

result = qb.run(text)
print(f"Query builder result: {result}")

# Print the query
print(f"Initial Text: {text}")
print(f"Query: {result}")

print("Done.")
