# Reflection을 이용한 RAG의 성능 향상 

<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Frag-with-reflection&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
<img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">


RAG의 결과가 충분한 context를 포함하고 있지 않은 경우에 [Query Tansformation](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/query-transformation.md)을 이용하면, 다양한 sub queries를 통해 풍부한 context를 가지는 결과를 얻을 수 있습니다. 그러나, 질문(query)이 짧은 경우에는 query transformation으로 얻어진 결과가 늘어난 지연 시간만큼 효과를 얻지 못할 수 있습니다. 또한 chatbot의 경우에 [이전 history를 이용해 질문을 rephrase](https://medium.com/thedeephub/rag-chatbot-powered-by-langchain-openai-google-generative-ai-and-hugging-face-apis-6a9b9d7d59db)하므로 query transformation을 통해 질문을 명확하게(rewrite) 하더라도 RAG의 결과가 충분히 좋아지지 않습니다. 

여기서에서는 Reflection을 이용한 RAG의 성능 강화에 대해 설명합니다. Reflection을 통해 RAG로 얻어진 결과에서 새로운 sub-queries들을 생성하여, 답변에 충분한 context를 제공할 수 있습닏. 따라서, Query transformation와 유사하게 질문과 관련된 문서를 조회할 수 있습니다. 

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



### Query Transformation

[query_transformations.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/query_transformations.ipynb)을 참조하여 query transformation을 위한 prompt를 생성합니다.


## 실행결과

### 기본 RAG

아래와 같이 메뉴에서 "RAG (Basic)"을 선택합니다.

![image](https://github.com/user-attachments/assets/0bf0dfaf-b73e-441c-b6ae-123300b427da)

채팅창에 "Advanced RAG에 대해 설명해주세요"라고 입력합니다.

![image](https://github.com/user-attachments/assets/db55d84c-ce7b-4362-a6cf-d0f312644df0)

### Reflection 

![noname](https://github.com/user-attachments/assets/0c0f7b04-f086-449a-949a-cc725aef7480)
