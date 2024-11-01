from flask import Flask, request, jsonify
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

app = Flask(__name__)

# Carregar o modelo e o tokenizador
model_name = "EleutherAI/gpt-neo-125M"
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Rota simples para verificar se a API está funcionando
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'API está funcionando!'})

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    
    # Verifique se o prompt não está vazio
    if not prompt:
        return jsonify({'error': 'Prompt não pode ser vazio!'}), 400

    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
