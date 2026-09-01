import streamlit as st
import tempfile
import hashlib
import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


@st.cache_resource(show_spinner="Reading PDF and building the agent...")
def build_app(pdf_path: str, file_hash: str):
    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    llm = ChatGroq(model="llama-3.1-8b-instant")

    class RouteDecision(BaseModel):
        destination: Literal["pdf", "math", "general"] = Field(
            description="Which specialist should handle this question"
        )
        reasoning: str = Field(description="One short sentence explaining the choice")

    router_llm = llm.with_structured_output(RouteDecision)

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        route: str
        handoff_count: int

    def format_history(messages, limit=6):
        recent = messages[-limit:]
        lines = []
        for m in recent:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines)

    def supervisor(state: State):
        question = state["messages"][-1].content
        decision = router_llm.invoke([
            HumanMessage(content=(
                "Route the question to the right specialist. Examples:\n\n"
                "Q: What tools are recommended for load and stress testing?\n"
                "A: pdf\n\n"
                "Q: What does the document say about observability?\n"
                "A: pdf\n\n"
                "Q: Does the document mention cross-chain?\n"
                "A: pdf\n\n"
                "Q: Is that covered in what I uploaded?\n"
                "A: pdf\n\n"
                "Q: What's 12 plus 8?\n"
                "A: math\n\n"
                "Q: Now subtract 5 from that\n"
                "A: math\n\n"
                "Q: What's the capital of France?\n"
                "A: general\n\n"
                "Rules:\n"
                "- 'pdf': anything possibly answered by the uploaded document, INCLUDING "
                "any question that references 'the document', 'the doc', 'this file', "
                "'what I uploaded', or asks whether something is mentioned/covered/included\n"
                "- 'math': arithmetic, including follow-ups that reference a previous "
                "calculation (e.g. 'add 5 to that', 'now divide by 2')\n"
                "- 'general': everything else\n"
                "When in doubt between pdf and general, choose pdf.\n\n"
                f"Q: {question}\nA:"
            ))
        ])
        return {"route": decision.destination}

    def route_decision(state: State) -> Literal["pdf_agent", "math_agent", "general_agent"]:
        return {"pdf": "pdf_agent", "math": "math_agent", "general": "general_agent"}[state["route"]]

    def pdf_agent(state: State):
        history = state["messages"]
        latest_question = history[-1].content

        if len(history) > 1:
            transcript = format_history(history[:-1], limit=4)
            rewrite_prompt = (
                "Rewrite the follow-up question as a short, standƒalone search query "
                "for a document search engine. Strip out phrases like 'the document', "
                "'what I uploaded', 'this file' — just extract the actual topic being asked "
                "about. Output ONLY the rewritten query, nothing else.\n\n"
                f"Conversation so far:\n{transcript}\n\n"
                f"Follow-up question: {latest_question}\n\n"
                "Standalone query:"
            )
            rewritten = llm.invoke([HumanMessage(content=rewrite_prompt)])
            resolved_question = rewritten.content.strip()
        else:
            resolved_question = latest_question

        retrieved = retriever.invoke(resolved_question)
        context = "\n\n".join(d.page_content for d in retrieved)

        if not context.strip():
            return {"route": "general", "handoff_count": state.get("handoff_count", 0) + 1}

        prompt = f"Answer using only this context:\n{context}\n\nQuestion: {resolved_question}"
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}

    def math_agent(state: State):
        history = state["messages"]
        latest_question = history[-1].content

        if len(history) > 1:
            transcript = format_history(history[:-1], limit=4)
            prompt = (
                "Convert the follow-up into a standalone Python arithmetic expression "
                "only, nothing else — no words, no explanation.\n\n"
                f"Conversation so far:\n{transcript}\n\n"
                f"Follow-up: {latest_question}\n\nExpression:"
            )
        else:
            prompt = (
                f"Convert this to a Python arithmetic expression only, nothing else, "
                f"no explanation, no words — just the expression:\n{latest_question}"
            )

        response = llm.invoke([HumanMessage(content=prompt)])
        expr = response.content.strip()
        try:
            result = eval(expr, {"__builtins__": {}})
            answer = f"{expr} = {result}"
        except Exception:
            answer = f"Couldn't safely evaluate: {expr}"
        return {"messages": [AIMessage(content=answer)]}

    def general_agent(state: State):
        history = state["messages"]
        recent = history[-6:]
        response = llm.invoke(recent)
        return {"messages": [response]}

    def after_pdf(state: State) -> Literal["general_agent", "__end__"]:
        if state["route"] == "general" and state.get("handoff_count", 0) > 0:
            return "general_agent"
        return END

    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("pdf_agent", pdf_agent)
    graph.add_node("math_agent", math_agent)
    graph.add_node("general_agent", general_agent)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_decision)
    graph.add_conditional_edges("pdf_agent", after_pdf)
    graph.add_edge("math_agent", END)
    graph.add_edge("general_agent", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


st.set_page_config(page_title="PDF Q&A Agent", page_icon="\U0001F4C4")
st.title("PDF Q&A Agent")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is None:
    st.info("Upload a PDF to start asking questions.")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]

temp_dir = tempfile.gettempdir()
temp_pdf_path = os.path.join(temp_dir, f"pdf_qa_{file_hash}.pdf")
if not os.path.exists(temp_pdf_path):
    with open(temp_pdf_path, "wb") as f:
        f.write(file_bytes)

st.caption(f"Loaded: {uploaded_file.name}")

if st.session_state.get("current_pdf_hash") != file_hash:
    st.session_state.current_pdf_hash = file_hash
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.display_messages = []

app = build_app(temp_pdf_path, file_hash)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

with st.sidebar:
    st.header("Controls")
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.display_messages = []
        st.rerun()
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}`")

for role, content, route in st.session_state.display_messages:
    with st.chat_message(role):
        if route:
            st.caption(f"routed to: {route}")
        st.write(content)

if question := st.chat_input("Ask a question..."):
    st.session_state.display_messages.append(("user", question, None))
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = app.invoke(
                {"messages": [HumanMessage(content=question)], "route": "", "handoff_count": 0},
                config=config,
            )
            route = result.get("route", "?")
            answer = result["messages"][-1].content
        st.caption(f"routed to: {route}")
        st.write(answer)

    st.session_state.display_messages.append(("assistant", answer, route))