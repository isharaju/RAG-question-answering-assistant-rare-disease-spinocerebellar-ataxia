# save this as llm_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn

app = FastAPI()

# Load your GGUF model here (adjust path)
llm = Llama(model_path="../models/mistral-7b-instruct-v0.1.Q3_K_M.gguf")

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

@app.post("/v1/completions")
def completions(req: CompletionRequest):
    try:
        output = llm(
            req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stop=["\n\n"]
        )
        return {"choices": [{"text": output['choices'][0]['text']}]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Runs on http://localhost:1234 by default
    uvicorn.run(app, host="0.0.0.0", port=1234)
