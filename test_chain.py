from dotenv import load_dotenv
load_dotenv()
from rag.chain import answer

result = answer("What are the different types of indexes in PostgreSQL?")
print("ANSWER:", result["answer"][:300])
print("SOURCES:", [s["name"] for s in result["sources"]])
print("REJECTED:", result["rejected"])

print("\n---\n")

result2 = answer("What is the capital of France?")
print("ANSWER:", result2["answer"])
print("REJECTED:", result2["rejected"])