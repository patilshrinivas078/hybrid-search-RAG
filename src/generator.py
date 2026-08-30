import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langfuse import observe
from langfuse.langchain import CallbackHandler


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")


llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="openai/gpt-oss-20b")
print(f"[INFO] Groq LLM initialized")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that answers questions using ONLY "
        "the context provided below.\n"
        'If the answer is not contained in the context, say '
        '"I don\'t know based on the available documents" instead of guessing.\n'
        "Do not use any outside knowledge."
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
])

chain = prompt | llm | StrOutputParser()


@observe(name="generate")
def generate(query: str, context: list[str]) -> str:
    """Generate an answer using only the provided context.

    `context` must be a list of chunk strings, NOT a pre-joined string -
    this function does the joining itself. Passing an already-joined
    string here will silently break: "\\n\\n".join(a_string) iterates
    over individual characters, not chunks.
    """
    if not context:
        return "I don't know based on the available documents."

    context_text = "\n\n".join(context)

    # Langfuse's LangChain callback handler auto-captures the full
    # prompt, token usage, and latency for this chain since it's a real
    # LangChain runnable - richer than logging those fields by hand.
    # get_current_langchain_handler() links it as a child of whatever
    # @observe span is currently active (e.g. handle_query in app.py),
    # so it nests correctly instead of starting its own separate trace.

    answer = chain.invoke(
        {"context": context_text, "question": query},
        config={"callbacks": [CallbackHandler()]},
    )

    return answer