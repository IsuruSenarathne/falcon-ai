from dotenv import load_dotenv
from app import create_app
import langchain

# Enable LangChain debugging to see LLM logs and thinking
langchain.debug = True

load_dotenv()

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
