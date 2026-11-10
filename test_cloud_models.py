
from llm.model_manager import model_manager


print("\n===== FIVE LLM CONFIGURATION =====")

for model in model_manager.list_models():
    print(
        model["family"],
        "| Provider:", model["provider"],
        "| Model:", model["model_tag"],
        "| Configured/installed:", model["installed"],
    )


print("\n===== GENERATION TEST =====")

prompt = "Explain artificial intelligence in two sentences."

for family in ["llama", "gemini", "groq", "openrouter", "cloudflare"]:

    print(f"\nTesting {family}...")

    result = model_manager.generate(
        family,
        prompt,
        max_tokens=500,
    )

    print("Success:", result.success)
    print("Model:", result.model_name)
    print("Latency:", result.latency_seconds, "seconds")
    print("Response:", result.response_text[:250])
    print("Error:", result.error_message)
