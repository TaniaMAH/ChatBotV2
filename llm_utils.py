from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Puedes cambiar este modelo por otro compatible con Transformers
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# Carga del modelo solo una vez
print("🧠 Cargando modelo local desde Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

def generate_response(prompt: str, max_tokens: int = 1024, temperature: float = 0.1, top_p: float = 0.9) -> str:
    print("🤖 Generando respuesta LLM...")
    response = llm_pipeline(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    return response[0]['generated_text']