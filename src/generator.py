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
        """You are an insurance policy assistant.

            Answer the user's question using ONLY the provided policy context.
            Do not use outside knowledge or make assumptions.

            Follow these rules carefully:

            1. Use only facts explicitly supported by the context.
            2. Preserve exact policy identifiers, UINs, endorsement numbers, percentages,
            amounts, dates, limits, conditions, and exclusions exactly as written.
            3. Do not shorten, reconstruct, normalize, or guess identifiers or numeric values.
            4. Distinguish carefully between:
            - base policy coverage
            - add-on coverage
            - exclusions
            - conditions and limits
            5. Do not infer or derive policy rules that are not explicitly stated in the provided context.
                In particular, do not infer:
                - that a provision is the "only" option,
                - that something is or is not covered,
                - eligibility conditions,
                - exclusions,
                - limits,
                - timelines,
                - payment calculations,
                - or relationships between multiple provisions unless the context explicitly supports that conclusion.
            6. When answering a question with multiple parts, answer each part only from 
            evidence explicitly present in the context. If the context does not establish a part, 
            say that the available documents do not provide enough information.
            7. When several applicable provisions are present in the context, include all
            relevant provisions rather than selecting only one.
            8. If the context contains conflicting information, do not resolve the conflict
            using outside knowledge. State that the provided documents contain conflicting
            information. 
            9. If the context does not contain enough information to answer a part of the
            question, explicitly say that the available documents do not provide enough
            information for that part.
            10. Do not invent policy features, endorsements, add-ons, or conditions.

            Prefer a precise answer over a broad or speculative one.
        """
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
