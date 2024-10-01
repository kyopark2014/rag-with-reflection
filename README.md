# Reflection을 이용한 RAG의 성능 향상 

<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Frag-with-reflection&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
<img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">

여기에서는 LangGraph를 이용하여 기본 RAG를 구현하고, [reflection](https://github.com/kyopark2014/langgraph-agent/blob/main/reflection-agent.md)과 [query transformation](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/query-transformation.md)을 이용하여 RAG의 성능을 향상시키는 방법을 비교하여 설명합니다. Reflection과 transformation을 이용하였을 때의 activity diagram은 아래와 같습니다. 여기에서 "(a) RAG with reflection"은 RAG를 조회하여 얻은 문서들을 이용하여 답변을 구한 후에, reflection을 통하여 답변에서 개선하여야 할 목록 및 추가로 검색을 수행하여 얻은 문서들로 향상된 답변을 생성합니다. Reflection은 답변과 관련된 추가 질문을 하여 답변을 수정하므로 답변에 더 많은 정보를 제공할 수 있습니다. "(b) RAG with Transformation"은 사용자의 질문을 rewrite한 후에 추가적으로 검색할 질문들을 생성하여 RAG를 조회합니다. Transformation은 RAG를 조회하기 위해 질문을 명확히 하고 관련된 세부 질문을 미리 생성하여 검색하므로 사용자의 질문과 좀더 가까운 문서들을 검색하여 활용할 수 있습니다. 

Query Tansformation은 질문(query)이 짧은 경우에는 query transformation으로 얻어진 결과가 늘어난 지연 시간만큼 효과를 얻지 못할 수 있습니다. 또한 chatbot의 경우에 [이전 history를 이용해 질문을 rephrase](https://medium.com/thedeephub/rag-chatbot-powered-by-langchain-openai-google-generative-ai-and-hugging-face-apis-6a9b9d7d59db)하므로 query transformation을 통해 질문을 명확하게(rewrite) 하더라도 RAG의 결과가 충분히 좋아지지 않습니다. 하지만 reflection은 RAG를 통해 얻은 답변을 업데이트하므로 transformation보다 더 긴 지연시간을 필요로 합니다. 

![image](./chart/workflow.png)



## LangGraph를 이용한 RAG의 구현

### Basic RAG 

기본 RAG를 구현하기 위한  activity diagram 아래와 같습니다. Retrieve 노드에서 RAG로 사용자의 질문을 전달하여 관련된 문서(relevant document)들을 가져옵니다. RAG에 질문과 관련된 문서들이 없는 경우에는 관련도가 떨어지는 문서들이 선택될 수 있으므로 grading을 통해 문서를 선택합니다. 이후 generate 노드에서 사용자의 질문과 관련된 문서를 이용하여 결과를 얻습니다.

<img src="./chart/rag-basic.png" width="150">

LangGraph로 아래와 같이 workflow를 구성합니다. 여기에는 retrieve_node, parallel_grader, generate_node가 있습니다. retrieve_node는 RAG에 질의하고, parallel_grader는 가져온 관련된 문서를 검증하고, generate_node에서 답변을 생성합니다. 

```python
class State(TypedDict):
    query: str
    draft: str
    relevant_docs: List[str]
    filtered_docs: List[str]
    reflection : List[str]
    sub_queries : List[str]
    revision_number: int

def buildRagBasic():
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("parallel_grader", parallel_grader)
    workflow.add_node("generate_node", generate_node)

    # Set entry point
    workflow.set_entry_point("retrieve_node")
    
    workflow.add_edge("retrieve_node", "parallel_grader")
    workflow.add_edge("parallel_grader", "generate_node")
            
    return workflow.compile()
```

기본 RAG를 실행할 때에는 아래와 같이 입력으로 query를 이용하여 LangGraph로 생성한 앱을 수행합니다. 수행이 완료되면 State의 draft를 추출하여 답변으로 전달합니다.

```python
def run_rag_basic(connectionId, requestId, query):    
    app = buildRagBasic()
    
    # Run the workflow
    isTyping(connectionId, requestId)        
    inputs = {
        "query": query
    }    
    config = {
        "recursion_limit": 50
    }
    
    output = app.invoke(inputs, config)
    
    return output['draft']
```

RAG에서 관련된 문서를 가져오는 함수는 아래와 같습니다. State에서 query를 추출하여 [Bedrock Knowledge Base](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/rag-knowledge-base.md)를 이용해 구성한 RAG에 관련된 문서를 추출해서 전달하도록 요청합니다. 

```python
def retrieve_node(state: State):
    print("###### retrieve ######")
    query = state['query']
    relevant_docs = retrieve_from_knowledge_base(query)
    
    return {
        "relevant_docs": relevant_docs
    }
```

아래와 같이 RAG의 지식저장소로부터 얻어진 관련된 문서를 grading합니다. 여러번 grading을 수행하여야 하므로 [Process-based parallelism](https://docs.python.org/ko/3/library/multiprocessing.html)을 사용하였습니다. 

```python
def parallel_grader(state: State):
    print("###### parallel_grader ######")
    query = state['query']
    relevant_docs = state['relevant_docs']
    
    global selected_chat    
    filtered_docs = []    

    processes = []
    parent_connections = []
    
    for i, doc in enumerate(relevant_docs):
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        process = Process(target=grade_document_based_on_relevance, args=(child_conn, query, doc, multi_region_models, selected_chat))
        processes.append(process)

        selected_chat = selected_chat + 1
        if selected_chat == len(multi_region_models):
            selected_chat = 0
    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        doc = parent_conn.recv()

        if doc is not None:
            filtered_docs.append(doc)

    for process in processes:
        process.join()    
    
    return {
        "filtered_docs": filtered_docs
    }    
```

여기에서는 관련도를 판단하기 위하여 LLM을 이용하 grading을 "yes/no"로 판정하였습니다. 

```python
def get_retrieval_grader(chat):
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    structured_llm_grader = chat.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def grade_document_based_on_relevance(conn, question, doc, models, selected):     
    chat = get_multi_region_chat(models, selected)
    retrieval_grader = get_retrieval_grader(chat)
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    
    grade = score.binary_score    
    if grade == 'yes':
        print("---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:  # no
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        conn.send(None)
    
    conn.close()
```

 
### RAG with Reflection

기본 RAG에 reflection을 추가하여, RAG로 부터 생성된 답변을 강화합니다. 

<img src="./chart/rag-with-reflection.png" width="400">
   

아래는 reflection을 위한 workflow입니다. reflect_node에서는 이전 답변(draft)로 부터 개선점을 추출하고, 관련된 3개의 query를 생성합니다. parallel_retriever는 3개의 query를 병렬로 조회하여 관련된 문서(relevant documents)를 얻고, parallel_grader를 이용하여 grading한 후에 revise_node를 이용하여 향상된 답변을 얻습니다. 

```python
def buildRagWithReflection():
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("parallel_grader", parallel_grader)
    workflow.add_node("generate_node", generate_node)
    
    workflow.add_node("reflect_node", reflect_node)    
    workflow.add_node("parallel_retriever", parallel_retriever)    
    workflow.add_node("parallel_grader_subqueries", parallel_grader)
    workflow.add_node("revise_node", revise_node)

    # Set entry point
    workflow.set_entry_point("retrieve_node")
    
    workflow.add_edge("retrieve_node", "parallel_grader")
    workflow.add_edge("parallel_grader", "generate_node")
    
    workflow.add_edge("generate_node", "reflect_node")
    workflow.add_edge("reflect_node", "parallel_retriever")    
    workflow.add_edge("parallel_retriever", "parallel_grader_subqueries")    
    workflow.add_edge("parallel_grader_subqueries", "revise_node")
    
    workflow.add_conditional_edges(
        "revise_node", 
        continue_reflection, 
        {
            "end": END, 
            "continue": "reflect_node"}
    )
        
    return workflow.compile()
```

reflection은 초안(draft)로 부터 아래와 같이 [structured output](https://github.com/kyopark2014/langgraph-agent/blob/main/structured-output.md)을 이용하여 추출합니다. 추출된 결과에는 reflection과 관련하여 missing, advisable, superfluous을 얻어서 문자의 개선에 도움을 줄 수 있으며, sub_queries를 이용해 1-3개의 새로운 질문을 생성합니다.

```python
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    advisable: str = Field(description="Critique of what is helpful for better writing")
    superfluous: str = Field(description="Critique of what is superfluous")
class Research(BaseModel):
    """Provide reflection and then follow up with search queries to improve the question/answer."""

    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    sub_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )
class ReflectionKor(BaseModel):
    missing: str = Field(description="답변에 있어야하는데 빠진 내용이나 단점")
    advisable: str = Field(description="더 좋은 답변이 되기 위해 추가하여야 할 내용")
    superfluous: str = Field(description="답변의 길이나 스타일에 대한 비평")

class ResearchKor(BaseModel):
    """답변을 개선하기 위한 검색 쿼리를 제공합니다."""

    reflection: ReflectionKor = Field(description="답변에 대한 평가")
    sub_queries: list[str] = Field(
        description="답변과 관련된 3개 이내의 검색어"
    )

def reflect_node(state: State):
    print("###### reflect ######")
    query = state['query']
    draft = state['draft']
        
    reflection = []
    sub_queries = []
    for attempt in range(5):
        chat = get_chat()        
        if isKorean(draft):
            structured_llm = chat.with_structured_output(ResearchKor, include_raw=True)
            qa = f"질문: {query}\n\n답변: {draft}"    
        else:
            structured_llm = chat.with_structured_output(Research, include_raw=True)
            qa = f"Question: {query}\n\nAnswer: {draft}"
        
        info = structured_llm.invoke(qa)                
        if not info['parsed'] == None:
            parsed_info = info['parsed']
            reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
            sub_queries = parsed_info.sub_queries
                
            if isKorean(draft):
                translated_search = []
                for q in sub_queries:
                    chat = get_chat()
                    if isKorean(q):
                        search = traslation(chat, q, "Korean", "English")
                    else:
                        search = traslation(chat, q, "English", "Korean")
                    translated_search.append(search)
                        
                sub_queries += translated_search

            break
    return {
        "reflection": reflection,
        "sub_queries": sub_queries,
    }
```

parallel_retriever는 sub_queries만큼 병렬처리를 수행하여 속도를 개선합니다. 이때 retriever는 완젼관리형인 RAG 서비스인 knowledge base에서 관련된 문서를 가져옵니다. 

```python
def retriever(conn, query):
    relevant_docs = retrieve_from_knowledge_base(query)    
    print("---RETRIEVE: RELEVANT DOCUMENT---")
    
    conn.send(relevant_docs)    
    conn.close()
    
    return relevant_docs
    
def parallel_retriever(state: State):
    print("###### parallel_retriever ######")
    sub_queries = state['sub_queries']
    print('sub_queries: ', sub_queries)
    
    relevant_docs = []
    processes = []
    parent_connections = []
    
    for i, query in enumerate(sub_queries):
        print(f"retrieve sub_queries[{i}]: {query}")        
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        process = Process(target=retriever, args=(child_conn, query))
        processes.append(process)

    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        docs = parent_conn.recv()
        
        for doc in docs:
            relevant_docs.append(doc)

    for process in processes:
        process.join()    

    return {
        "relevant_docs": relevant_docs
    }
```

revise_node에서는 reflection으로 얻어진 reflection critique와 sub-quries를 이용해 조회한 관련된 문서들을 이용하여 아래와 같이 초안(draft)를 향상시킵니다. 

```python
def revise_node(state: State):   
    print("###### revise ######")
    draft = state['draft']
    reflection = state['reflection']
    
    if isKorean(draft):
        revise_template = (
            "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."                
            "draft을 critique과 information 사용하여 수정하십시오."
            "최종 결과는 한국어로 작성하고 <result> tag를 붙여주세요."
                            
            "<draft>"
            "{draft}"
            "</draft>"
                            
            "<critique>"
            "{reflection}"
            "</critique>"

            "<information>"
            "{content}"
            "</information>"
        )
    else:    
        revise_template = (
            "You are an excellent writing assistant." 
            "Revise this draft using the critique and additional information."
            "Provide the final answer with <result> tag."
                            
            "<draft>"
            "{draft}"
            "</draft>"
                        
            "<critique>"
            "{reflection}"
            "</critique>"

            "<information>"
            "{content}"
            "</information>"
        )
                    
    revise_prompt = ChatPromptTemplate([
        ('human', revise_template)
    ])    
    filtered_docs = state['filtered_docs']
              
    content = []   
    if len(filtered_docs):
        for d in filtered_docs:
            content.append(d.page_content)        

    chat = get_chat()
    reflect = revise_prompt | chat
           
    res = reflect.invoke(
        {
            "draft": draft,
            "reflection": reflection,
            "content": content
        }
    )
    output = res.content
        
    revised_draft = output[output.find('<result>')+8:len(output)-9]
            
    revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
            
    return {
        "draft": revised_draft,
        "revision_number": revision_number + 1
    }
```

이때, 초안은 MAX_REVISIONS만큼 반복하여 refection을 적용할 수 있습니다. 

```python
MAX_REVISIONS = 1

def continue_reflection(state: State, config):
    print("###### continue_reflection ######")
    max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
            
    if state["revision_number"] > max_revisions:
        return "end"
    return "continue"
```

### Query Transformation

RAG에 질문하기 전에 입력된 query를 변환하여 성능을 향상시키기 위해서는 아래와 같이 rewrite와 decompse과정이 필요합니다. rewrite_node는 RAG에서 좀더 좋은 결과를 얻도록 query의 내용을 자세하게 풀어 적습니다. 이후 decompse_node에서는 RAG에서 조회할때 사용할 sub-quries들을 생성합니다. 여기에서는 [query_transformations.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/query_transformations.ipynb)을 참조하여 query transformation을 위한 prompt를 생성합니다.


<img src="./chart/rag-with-transformation.png" width="400">

Transformation을 위한 workflow는 아래와 같습니다. RAG를 조회하기 전에 rewrite_node로 질문을 풀어서 쓰고, decompose_node로 상세한 질문들을 생성합니다. 

```python
def buildRagWithTransformation():
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("rewrite_node", rewrite_node)
    workflow.add_node("decompose_node", decompose_node)
    workflow.add_node("parallel_retriever", parallel_retriever)
    workflow.add_node("parallel_grader", parallel_grader)
    workflow.add_node("generate_node", generate_node)
    
    # Set entry point
    workflow.set_entry_point("rewrite_node")
    
    # Add edges
    workflow.add_edge("rewrite_node", "decompose_node")
    workflow.add_edge("decompose_node", "parallel_retriever")
    workflow.add_edge("parallel_retriever", "parallel_grader")
    workflow.add_edge("parallel_grader", "generate_node")    
    workflow.add_edge("generate_node", END)
```

rewrite_node에서는 질문을 검색에 맞게 상세하게 풀어줍니다.

```python
def rewrite_node(state: State):
    print("###### rewrite ######")
    query = state['query']
    
    query_rewrite_template = (
        "You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system."
        "Given the original query, rewrite it to be more specific," 
        "detailed, and likely to retrieve relevant information."
        "Put it in <result> tags."

        "Original query: {original_query}"
        "Rewritten query:"
    )
    
    rewrite_prompt = ChatPromptTemplate([
        ('human', query_rewrite_template)
    ])

    chat = get_chat()
    rewrite = rewrite_prompt | chat
           
    res = rewrite.invoke({"original_query": query})    
    revised_query = res.content
    
    revised_query = revised_query[revised_query.find('<result>')+8:len(revised_query)-9] # remove <result> tag                   
    
    return {
        "query": revised_query
    }
```


## 실행결과

### 기본 RAG

아래와 같이 메뉴에서 "RAG (Basic)"을 선택합니다.

<img src="https://github.com/user-attachments/assets/0bf0dfaf-b73e-441c-b6ae-123300b427da" width="400">

채팅창에 "Advanced RAG에 대해 설명해주세요"라고 입력하고 결과를 확인합니다. 이때, RAG에 포함된 문서에 따라 결과는 달라집니다. 여기에서는 RAG와 관련된 각종 PPT를 넣었을때의 결과입니다. 파일은 화면 하단의 파일 아이콘을 이용해 넣거나, [Amazon S3 Console](https://ap-northeast-2.console.aws.amazon.com/s3/home?region=us-west-2#)에서 직접 push해서 넣을 수 있습니다. 

![image](https://github.com/user-attachments/assets/db55d84c-ce7b-4362-a6cf-d0f312644df0)

### Reflection 

![noname](https://github.com/user-attachments/assets/0c0f7b04-f086-449a-949a-cc725aef7480)
