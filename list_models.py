from groq import Groq
from src.config import settings

client = Groq(api_key=settings.groq_api_key)

print("Models your key can see:")
print("------------------------")
for model in client.models.list().data:
    print(model.id)