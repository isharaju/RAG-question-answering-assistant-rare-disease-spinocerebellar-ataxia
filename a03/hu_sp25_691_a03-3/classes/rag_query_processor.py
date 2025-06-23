from .llm_client import LLMClient
from .chromadb_retriever import ChromaDBRetriever
import logging
from collections import deque

# from pathlib import Path
# from typing import List
# import json

class RAGQueryProcessor:

    def __init__(self,
                 llm_client: LLMClient,
                 retriever: ChromaDBRetriever,
                 use_rag: bool = False):
        self.use_rag = use_rag
        self.llm_client = llm_client
        self.retriever = retriever if use_rag else None
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized RAGQueryProcessor: use_rag: {use_rag}")
        self.conversation_history = deque(maxlen=5)  # Keep last 5 exchanges


    # def query(self, query_text: str):
    #     """
    #     Processes the query with optional RAG.
    #     """
    #     self.logger.debug(f"Received query: {query_text}")

    #     context = ""
    #     if self.use_rag:
    #         self.logger.info("-"*80)
    #         self.logger.info("Using RAG pipeline...")
    #         retrieved_docs = self.retriever.query(query_text)
    #         if not retrieved_docs:
    #             logging.info("*** No relevant documents found.")
    #         else:
    #             result = retrieved_docs[0]
    #             context = result.get('context', '')
    #             logging.info(f"ID: {result.get('id', 'N/A')}")  # Handle missing ID
    #             logging.info(f"Score: {result.get('score', 'N/A')}")
    #             doc_text = result.get('text', '')
    #             preview_text = (doc_text[:150] + "...") if len(doc_text) > 150 else doc_text
    #             logging.info(f"Document: {preview_text}")
    #             logging.info(f"Context: {context}")
    #         self.logger.info("-" * 80)

    #     # Construct structured prompt


    #     final_prompt = f"""
    #     You are a medical assistant answering questions about Spinocerebellar Ataxia using the context provided below
    #     If the context is insufficient, say 'I don't know'. 

    #     Context:
    #     {context if context else "No relevant context found."}

    #     Question:
    #     {query_text}

    #     Answer:
    #     """

    #     self.logger.debug(f"Prompt to LLM: {final_prompt}")

    #     # response = self.llm_client.query(final_prompt)

    #     response = self.llm_client.query(
    #         final_prompt,
    #         stop=["\nContext:", "\nQuestion:", "\nAnswer:"]
    #     )

    #     self.logger.debug(f"LLM Response: {response}")

    #     return response

        # # RAG mode: Retrieve relevant context
        # context = ""
        # if self.use_rag:
        #     self.logger.info("-"*80)
        #     self.logger.info("Using RAG pipeline...")
        #     retrieved_docs = self.retriever.query(query_text)
        #     # context = "\n".join(retrieved_docs[0].get('contexts'))
        #     # print(f"retrieved_docs:\n", retrieved_docs)
        #
        #     if not retrieved_docs:
        #         logging.info("*** No relevant documents found.")
        #     else:
        #         result = retrieved_docs[0]
        #         context = result.get('context', '')
        #
        #         # logging.info(f"Result {idx + 1}:")
        #         logging.info(f"ID: {result.get('id', 'N/A')}")  # Handle missing ID
        #         logging.info(f"Score: {result.get('score', 'N/A')}")
        #         doc_text = result.get('text', '')
        #         preview_text = (doc_text[:150] + "...") if len(doc_text) > 150 else doc_text
        #         logging.info(f"Document: {preview_text}")
        #         logging.info(f"Context: {context}")
        #     self.logger.info("-" * 80)
        #
        # # Construct the final prompt
        # final_prompt = f"Context:\n{context}\n\nUser Query:\n{query_text}" if context else query_text
        # self.logger.debug(f"Prompt to LLM: {final_prompt}")
        # response = self.llm_client.query(final_prompt)
        # self.logger.debug(f"LLM Response: {response}")
        # return response


    # def query(self, query_text: str):
    #     """
    #     Processes the query with optional RAG and conversational history.
    #     """
    #     self.logger.debug(f"Received query: {query_text}")

    #     context = ""
    #     if self.use_rag:
    #         self.logger.info("-" * 80)
    #         self.logger.info("Using RAG pipeline...")
    #         retrieved_docs = self.retriever.query(query_text)
    #         if not retrieved_docs:
    #             logging.info("*** No relevant documents found.")
    #         else:
    #             result = retrieved_docs[0]
    #             context = result.get('context', '')
    #             logging.info(f"ID: {result.get('id', 'N/A')}")
    #             logging.info(f"Score: {result.get('score', 'N/A')}")
    #             doc_text = result.get('text', '')
    #             preview_text = (doc_text[:150] + "...") if len(doc_text) > 150 else doc_text
    #             logging.info(f"Document: {preview_text}")
    #             logging.info(f"Context: {context}")
    #         self.logger.info("-" * 80)

    #     # --------------------------------------
    #     # NEW: Add conversation history
    #     history_text = ""
    #     for i, (prev_q, prev_a) in enumerate(self.conversation_history, 1):
    #         history_text += f"[Q{i}] {prev_q}\n[A{i}] {prev_a}\n"
    #     # --------------------------------------

    #     # Final prompt with history and new query
    #     final_prompt = f"""
    # You are a medical assistant answering questions about Spinocerebellar Ataxia using the context and conversation history provided below.
    # If the context is insufficient, say "I don't know".

    # Context:
    # {context if context else "No relevant context found."}

    # Conversation History:
    # {history_text if history_text else "None"}

    # Current Question:
    # {query_text}

    # Answer:
    # """.strip()

    #     self.logger.debug(f"Prompt to LLM:\n{final_prompt}")

    #     # Query LLM
    #     response = self.llm_client.query(
    #         final_prompt,
    #         stop=["\nContext:", "\nQuestion:", "\nAnswer:", "\nConversation History:"]
    #     )

    #     # self.logger.debug(f"LLM Response: {response}")

    #     # --------------------------------------
    #     # NEW: Append current interaction to history
    #     self.conversation_history.append((query_text, response.strip()))
    #     # --------------------------------------

    #     return response
    # # Add the new query method to the class


    def query(self, query_text: str, top_k: int = 5):
        """
        Processes the query with optional RAG and conversational history.
        Supports top-k chunk retrieval.
        """
        self.logger.debug(f"Received query: {query_text}")

        context = ""
        if self.use_rag:
            self.logger.info("-" * 80)
            self.logger.info(f"Using RAG pipeline (top_k={top_k})...")
            retrieved_docs = self.retriever.query(query_text, top_k=top_k)
            if not retrieved_docs:
                self.logger.info("*** No relevant documents found.")
            else:
                for idx, result in enumerate(retrieved_docs):
                    chunk_text = result.get('context', '')
                    context += f"\n[Doc {idx+1}]\n{chunk_text}\n"
                    self.logger.info(f"ID: {result.get('id', 'N/A')}")
                    self.logger.info(f"Score: {result.get('score', 'N/A')}")
                    preview_text = (result.get('text', '')[:150] + "...") if result.get('text') else ''
                    self.logger.info(f"Document: {preview_text}")
            self.logger.info("-" * 80)

        # Build conversation history
        history_text = ""
        for i, (prev_q, prev_a) in enumerate(self.conversation_history, 1):
            history_text += f"[Q{i}] {prev_q}\n[A{i}] {prev_a}\n"

        # Build final prompt
        final_prompt = f"""
    You are a medical assistant answering questions about Spinocerebellar Ataxia using the context and conversation history provided below.
    If the context is insufficient, say "I don't know".

    Context:
    {context.strip() if context else "No relevant context found."}

    Conversation History:
    {history_text.strip() if history_text else "None"}

    Current Question:
    {query_text}

    Answer:
    """.strip()

        self.logger.debug(f"Prompt to LLM:\n{final_prompt}")

        # Query LLM
        response = self.llm_client.query(
            final_prompt,
            stop=["\nContext:", "\nQuestion:", "\nAnswer:", "\nConversation History:"]
        )

        # Update conversation history
        self.conversation_history.append((query_text, response.strip()))

        return response
