from flask import Flask, render_template, request, redirect, url_for
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory

app = Flask(__name__)

# Store chat history
chat_history = []


def get_api_key():
    """Get the API key from Gemini"""
    api = "Add_API_KEY_Here"
    return api


def initialize_llm(api_key):
    """Initialize the GoogleGenerativeAI LLM with safety settings."""
    return GoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    )


def get_gemini_response(llm, question):
    template = """
    You are an intelligent chatbot. Respond to the user query in a helpful and polite manner: {user_question}
    Question: {user_question}
    Answer:
    """
    prompt = PromptTemplate(input_variables=["user_question"], template=template)
    formatted_prompt = prompt.format(user_question=question)

    try:
        response = llm.invoke(formatted_prompt)
    except Exception as e:
        print(f"Error getting response from LLM: {e}")
        return "I'm sorry, there was an error processing your request."

    return response


# Initialize the LLM with API key
API_KEY = get_api_key()
llm = initialize_llm(API_KEY)


@app.route("/", methods=["GET", "POST"])
def home():
    """Home route to handle chat input and display history."""
    if request.method == "POST":
        user_question = request.form["user_question"]
        response = get_gemini_response(llm, user_question)

        # Add the user question and chatbot response to the chat history
        chat_history.append({"user": user_question, "bot": response})

        return redirect(url_for("home"))

    return render_template("index.html", chat_history=chat_history)


@app.route("/reset", methods=["POST"])
def reset():
    chat_history.clear()
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
