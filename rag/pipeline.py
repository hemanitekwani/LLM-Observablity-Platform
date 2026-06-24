from rag.retriever import get_retriever

from rag.generator import get_llm


class Pipeline:
    def __init__(self):
        self.retriever = get_retriever()
        self.llm = get_llm()
        self.max_retries = 2 

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
    
    def _grade_alignment(self, question, context, answer):
        """The Internal Audit Node (Hallucination Catcher)."""
        
        if "I cannot determine" in answer:
            return "YES"

        grader_prompt = f"""
        You are a ruthless, zero-tolerance hallucination auditor. 
        Your ONLY job is to verify if the 'Generated Answer' is strictly derived from the 'Context'.

        Context:
        {context}

        Question:
        {question}

        Generated Answer:
        {answer}

        CRITICAL INSTRUCTIONS:
        1. If the Generated Answer contains EVEN ONE word, number, or claim that is not explicitly written in the Context, you MUST reply NO.
        2. If the Generated Answer relies on outside knowledge, you MUST reply NO.
        3. Only if every single fact in the answer is proven by the context, reply YES.

        Reply ONLY with YES or NO:
        """

        response = self.llm.invoke(grader_prompt)
        return response.content.strip().upper()

    def _rewrite_query(self,question):
        """The Fallback Mechanism (Query Rewriter)."""
        rewrite_prompt = f"""
        The user asked a question, but our vector database failed to find relevant context.
        Analyze the original question and rewrite it into a highly specific keyword search query 
        to help the database find better chunks.

        Original Question: {question}
        Output ONLY the rewritten search query.
        """

        response = self.llm.invoke(rewrite_prompt)
        

        return response.content.strip()
    

    def ask(self , question):
        current_query_search = question
        final_docs = []


        for attempt in range(self.max_retries):
            docs = self.retriever.invoke(current_query_search)

            final_docs = docs

            context = "\n\n".join(doc.page_content for doc in docs)

            draft_answer = self._generate_answer(question , context)

            grade = self._grade_alignment(question , context, draft_answer)

            if "YES" in grade:
                return {
                    "answer":draft_answer,
                    "context":final_docs,
                    "final_query_used":current_query_search,
                    "retries":attempt
                }
            
            if attempt < self.max_retries - 1:
                current_query_search = self._rewrite_query(question)


        return {
            "answer":"I cannot determine the answer from the provided context.",
            "contexts":final_docs,
            "final_query_used":current_query_search,
            "retries":self.max_retries,


        }



