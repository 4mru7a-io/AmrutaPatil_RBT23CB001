
from flask import Flask, request, render_template_string
import requests

app = Flask(__name__)

# HTML and CSS for chatbot interface
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Local AI Chatbot</title>
    <style>
        body {
            font-family: Arial;
            background: #eef2f7;
            text-align: center;
            padding: 30px;
        }

        .container {
            background: white;
            max-width: 650px;
            margin: auto;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 0 10px #ccc;
        }

        h1 {
            color: #334155;
        }

        input {
            width: 75%;
            padding: 12px;
            border: 1px solid #aaa;
            border-radius: 6px;
        }

        button {
            padding: 12px 20px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }

        .response {
            margin-top: 25px;
            padding: 15px;
            background: #e0f2fe;
            border-radius: 8px;
            text-align: left;
            white-space: pre-wrap;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>Local AI Chatbot</h1>
        <p>Powered by Ollama and Llama 3.1 (8B)</p>

        <form method="POST">
            <input type="text" name="question"
                   placeholder="Ask me anything..."
                   required>

            <button type="submit">Send</button>
        </form>

        {% if answer %}
        <div class="response">
            <h3>AI Response:</h3>
            {{ answer }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def chatbot():

    answer = ""

    if request.method == "POST":

        question = request.form["question"]

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
    "model": "llama3.1:8b",
    "prompt": question,
    "stream": False,
    "keep_alive": "10m",
    "options": {
        "num_predict": 200
    }
},
                timeout=300
            )

            response.raise_for_status()

            answer = response.json()["response"]

        except requests.exceptions.RequestException as e:
            answer = "Error connecting to Ollama: " + str(e)

    return render_template_string(HTML, answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
