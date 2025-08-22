from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import os

class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        if genai is None:
            raise RuntimeError(
                "SDK google-generativeai não encontrado. Instale com `pip install google-generativeai`."
            )
        self.model = model
        genai.configure(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.2, max_output_tokens: int = 1024) -> str:
        try:
            response = genai.GenerativeModel(self.model).generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens
                }
            )
            return response.text.strip()
        except Exception as e:
            return f"Erro ao chamar a API: {e}"

app = Flask(__name__)
gemini_client = GeminiClient("AIzaSyAsegb98WWAxWFGEWSaO9MS6xG4z-49hVg")

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt', '')
    resposta = gemini_client.generate(prompt)
    return jsonify({'resposta': resposta})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
