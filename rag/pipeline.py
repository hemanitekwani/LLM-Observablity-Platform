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

    def grade_context_relevance(self,question , context):
        """The CRAG Auditor Node (Grades Context BEFORE Generation)."""
        grader_prompt = f"""
        You are a strict relevance grader for a Retrieval-Augmented Generation system.
        Your job is to determine if the retrieved 'Context' contains sufficient, relevant facts to answer the 'Question'.

        Context:
        {context}

        Question:
        {question}

        INSTRUCTIONS:
        1. Read the question carefully to identify the core entities and requirements.
        2. Scan the context. Does it contain the actual information needed to answer the question?
        3. If the context is completely irrelevant, missing key facts, or talks about a different subject, reply NO.
        4. If the context contains the necessary facts (even partially), reply YES.
        5. If you are unsure, default to YES.

        Reply ONLY with YES or NO:
        """
        response = self.llm.invoke(grader_prompt)
        return response.content.strip().upper()

    
    # def _grade_alignment(self, question, context, answer):
    #     """The Internal Audit Node (Hallucination Catcher)."""
        
    #     grader_prompt = f"""
    #     You are a precise RAG auditor. Your goal is to verify if the 'Generated Answer' is supported by the 'Context'.

    #     Context:
    #     {context}

    #     Question:
    #     {question}

    #     Generated Answer:
    #     {answer}

    #     INSTRUCTIONS:
    #     1. FACTUAL GROUNDING: The answer must be based on the provided context. If the context does not contain the answer, the LLM should have said "I cannot determine...".
    #     2. TOLERANCE: Do NOT penalize for:
    #       - Conversational filler (e.g., "The answer is...", "According to the document...").
    #       - Formatting differences (e.g., list vs. paragraph).
    #       - Minor grammar changes.
    #     3. STRICT HALLUCINATION CHECK: Only reply NO if the answer provides specific data, names, numbers, or claims that are NOT present in the context.
    #     4. If you are unsure, default to YES.

    #     Reply ONLY with YES or NO:
    #     """

    #     response = self.llm.invoke(grader_prompt)
    #     return response.content.strip().upper()

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

            relevance_grade = self.grade_context_relevance(question , context)

            # draft_answer = self._generate_answer(question , context)

            # grade = self._grade_alignment(question , context, draft_answer)
            
            print(f"--- Attempt {attempt + 1} | Query: {current_query_search} | Grade: {relevance_grade} ---")

            if "YES" in relevance_grade:
                draft_answer = self._generate_answer(question , context)
                return {
                    "answer":draft_answer,
                    "contexts":[doc.page_content for doc in final_docs],
                    "final_query_used":current_query_search,
                    "retries":attempt,
                    "is_hallucination":False
                }
            
            if attempt < self.max_retries - 1:
                current_query_search = self._rewrite_query(question)


        return {
            "answer":"I cannot determine the answer from the provided context.",
            "contexts":[doc.page_content for doc in final_docs],
            "final_query_used":current_query_search,
            "retries":self.max_retries,
            "is_hallucination":True

        }



