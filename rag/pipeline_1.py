from rag.retriever_1 import get_retriever

from rag.generator import get_llm


class Pipeline_1:
    def __init__(self):
        self.retriever = get_retriever()
        self.llm = get_llm()


    def _generate_answer(self,question , context):
        """Standard Generation Step."""
        generation_prompt = f"""
        You are a strict data extraction audit system.
        Your task is to answer the user's question using ONLY the explicitly stated facts in the provided context.

        CRITICAL CONSTRAINTS:
        1. ENTITY VERIFICATION: Before answering, verify if the exact unique entities, subjects, proper nouns, specific versions, or terminology mentioned in the question are explicitly present and active in the provided context.
        2. NO CROSS-APPLICATION: If the context discusses a different version, a related but distinct subject, or lacks the specific facts requested, you must not infer, cross-apply facts, or substitute information.
        3. STRICT REFUSAL: If the exact subject and answer cannot be explicitly determined from the provided context, you MUST respond exactly with this phrase and nothing else:
        "I cannot determine the answer from the provided context."

        Context:
        {context}

        Question:
        {question}

        Answer:
        
        """

        response = self.llm.invoke(generation_prompt)

        return response.content.strip()



    def ask(self,question):
        docs = self.retriever.invoke(question)

        context = [doc.page_content for doc in docs]

        answer = self._generate_answer(question,context)


        return {
            "answer": answer,
            "contexts":docs,
            "final_query_used":question,
            "retries":0
        }
