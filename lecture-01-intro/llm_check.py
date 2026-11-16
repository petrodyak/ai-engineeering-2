import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # бере OPENROUTER_API_KEY з файлу .env

open_router_model = "nvidia/nemotron-3-ultra-550b-a55b:free"

QUESTION = "Назви столицю України одним словом."
TARGETS = [
	# назва, base_url, ключ, модель
	# (
	# 	"OpenRouter",
	# 	"https://openrouter.ai/api/v1",
	# 	os.getenv("OPENROUTER_API_KEY"),
	# 	"openai/gpt-4o-mini",
	# ),
	(
            "OpenRouter",
            "https://openrouter.ai/api/v1",
            os.getenv("OPENROUTER_API_KEY"),
            open_router_model,
    ),
	("Ollama", "http://localhost:11434/v1", "ollama", "llama3.2:3b"),
	("Qwen2", "http://localhost:11434/v1", "qwen2", "qwen2.5:7b"),
]

for name, base_url, key, model in TARGETS:
	if not key:
		print(f"{name}: немає OPENROUTER_API_KEY у .env")
		continue

	client = OpenAI(base_url=base_url, api_key=key, timeout=60, max_retries=2)
	start = time.perf_counter()
	try:
		r = client.chat.completions.create(
			model=model,
			messages=[{"role": "user", "content": QUESTION}],
			temperature=0,
		)
	except Exception as e:
		print(f"{name}: помилка {type(e).__name__}: {e}")
		continue

	sec = time.perf_counter() - start
	u = r.usage
	answer = r.choices[0].message.content.strip()
	print(
		f"{name} | {model} | {sec:.1f} с | "
		f"токени: вхід {u.prompt_tokens}, вихід {u.completion_tokens} | {answer}"
	)