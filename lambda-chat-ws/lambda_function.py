import json
import boto3
import os
import time
import datetime
import PyPDF2
import csv
import re
import traceback
import base64
import operator

from botocore.config import Config
from io import BytesIO
from urllib import parse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_aws import ChatBedrock
from multiprocessing import Process, Pipe

# from langchain_community.tools.tavily_search import TavilySearchResults
from PIL import Image

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from pydantic.v1 import BaseModel, Field
from typing import Annotated, List, Tuple, TypedDict, Literal, Sequence, Union
from langchain_aws import AmazonKnowledgeBasesRetriever
from tavily import TavilyClient  
from langgraph.constants import Send

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
path = os.environ.get('path')
doc_prefix = s3_prefix+'/'
debugMessageMode = os.environ.get('debugMessageMode', 'false')
projectName = os.environ.get('projectName')
LLM_for_chat = json.loads(os.environ.get('LLM_for_chat'))
LLM_for_multimodal= json.loads(os.environ.get('LLM_for_multimodal'))
selected_chat = 0
selected_multimodal = 0
useEnhancedSearch = False

knowledge_base_name = os.environ.get('knowledge_base_name')
    
multi_region_models = [   # claude sonnet 3.0
    {   
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "claude3",
        "max_tokens": 4096,
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "claude3",
        "max_tokens": 4096,
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    },
    {
        "bedrock_region": "ca-central-1", # Canada
        "model_type": "claude3",
        "max_tokens": 4096,
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    },
    {
        "bedrock_region": "eu-west-2", # London
        "model_type": "claude3",
        "max_tokens": 4096,
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    },
    {
        "bedrock_region": "sa-east-1", # Sao Paulo
        "model_type": "claude3",
        "max_tokens": 4096,
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    }
]
multi_region = 'disable'

reference_docs = []

secretsmanager = boto3.client('secretsmanager')
   
# api key to use LangSmith
langsmith_api_key = ""
try:
    get_langsmith_api_secret = secretsmanager.get_secret_value(
        SecretId=f"langsmithapikey-{projectName}"
    )
    # print('get_langsmith_api_secret: ', get_langsmith_api_secret)
    secret = json.loads(get_langsmith_api_secret['SecretString'])
    #print('secret: ', secret)
    langsmith_api_key = secret['langsmith_api_key']
    langchain_project = secret['langchain_project']
except Exception as e:
    raise e

if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project
    
# api key to use Tavily Search
tavily_api_key = ""
try:
    get_tavily_api_secret = secretsmanager.get_secret_value(
        SecretId=f"tavilyapikey-{projectName}"
    )
    #print('get_tavily_api_secret: ', get_tavily_api_secret)
    secret = json.loads(get_tavily_api_secret['SecretString'])
    # print('secret: ', secret)
    tavily_api_key = json.loads(secret['tavily_api_key'])
    # print('tavily_api_key: ', tavily_api_key)
except Exception as e: 
    raise e

def check_tavily_secret(tavily_api_key):
    query = 'what is LangGraph'
    valid_keys = []
    for key in tavily_api_key:
        try:
            tavily_client = TavilyClient(api_key=key)
            response = tavily_client.search(query, max_results=1)
            # print('tavily response: ', response)
            
            if 'results' in response and len(response['results']):
                valid_keys.append(key)
        except Exception as e:
            print('Exception: ', e)
    # print('valid_keys: ', valid_keys)
    
    return valid_keys

tavily_api_key = check_tavily_secret(tavily_api_key)
# print('tavily_api_key: ', tavily_api_key)
print('The number of valid tavily api keys: ', len(tavily_api_key))

selected_tavily = -1
if len(tavily_api_key):
    os.environ["TAVILY_API_KEY"] = tavily_api_key[0]
    selected_tavily = 0
      
def tavily_search(query, k):
    global selected_tavily
    docs = []
        
    if selected_tavily != -1:
        selected_tavily = selected_tavily + 1
        if selected_tavily == len(tavily_api_key):
            selected_tavily = 0

        try:
            tavily_client = TavilyClient(api_key=tavily_api_key[selected_tavily])
            response = tavily_client.search(query, max_results=k)
            # print('tavily response: ', response)
            
            if "url" in r:
                url = r.get("url")
                
            for r in response["results"]:
                name = r.get("title")
                if name is None:
                    name = 'WWW'
            
                docs.append(
                    Document(
                        page_content=r.get("content"),
                        metadata={
                            'name': name,
                            'url': url,
                            'from': 'tavily'
                        },
                    )
                )   
        except Exception as e:
            print('Exception: ', e)
    return docs

# result = tavily_search('what is LangChain', 2)
# print('search result: ', result)

# websocket
connection_url = os.environ.get('connection_url')
client = boto3.client('apigatewaymanagementapi', endpoint_url=connection_url)
print('connection_url: ', connection_url)

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"

map_chain = dict() 

# Multi-LLM
def get_chat():
    global selected_chat
    
    if multi_region == 'enable':
        length_of_models = len(multi_region_models)
        profile = multi_region_models[selected_chat]
    else:
        length_of_models = len(LLM_for_chat)
        profile = LLM_for_chat[selected_chat]
        
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
    print(f'LLM: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
    
    selected_chat = selected_chat + 1
    if selected_chat == length_of_models:
        selected_chat = 0
    
    return chat

def get_multi_region_chat(models, selected):
    profile = models[selected]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
    print(f'selected_chat: {selected}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
    
    return chat

def get_multimodal():
    global selected_multimodal
    print('LLM_for_chat: ', LLM_for_chat)
    print('selected_multimodal: ', selected_multimodal)
        
    profile = LLM_for_multimodal[selected_multimodal]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
    print(f'LLM: {selected_multimodal}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    multimodal = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
    
    selected_multimodal = selected_multimodal + 1
    if selected_multimodal == len(LLM_for_multimodal):
        selected_multimodal = 0
    
    return multimodal
    
# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read().decode('utf-8')
        
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    ) 

    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
    
    return texts

# load csv documents from s3
def load_csv_document(s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    print('lins: ', len(lines))
        
    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]  
    #columns_to_metadata = ["type","Source"]
    print('columns: ', columns)
    
    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        # print('row: ', row)
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = Document(
            page_content=content,
            metadata={
                'name': s3_file_name,
                'row': n+1,
            }
            #metadata=to_metadata
        )
        docs.append(doc)
        n = n+1
    print('docs[0]: ', docs[0])

    return docs

def get_summary(chat, docs):    
    text = ""
    for doc in docs:
        text = text + doc
    
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장을 요약해서 500자 이내로 설명하세오."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Write a concise summary within 500 characters."
        )
    
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        summary = result.content
        print('result of summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary
    
def load_chatHistory(userId, allowTime, chat_memory):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    # print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text' and text and msg:
            memory_chain.chat_memory.add_user_message(text)
            memory_chain.chat_memory.add_ai_message(msg) 
                
def getAllowTime():
    d = datetime.datetime.now() - datetime.timedelta(days = 2)
    timeStr = str(d)[0:19]
    print('allow time: ',timeStr)

    return timeStr

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        print('Korean: ', word_kor)
        return True
    else:
        print('Not Korean: ', word_kor)
        return False

def general_conversation(connectionId, requestId, chat, query):
    if isKorean(query)==True :
        system = (
            "다음의 Human과 Assistant의 친근한 이전 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )
    else: 
        system = (
            "Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor."
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        isTyping(connectionId, requestId)  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(connectionId, requestId, stream.content)    
        
        usage = stream.response_metadata['usage']
        print('prompt_tokens: ', usage['prompt_tokens'])
        print('completion_tokens: ', usage['completion_tokens'])
        print('total_tokens: ', usage['total_tokens'])
        msg = stream.content

    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")
    
    return msg

####################### LangGraph #######################
# Functions
#########################################################

knowledge_base_id = None
def retrieve_from_knowledge_base(query):
    global knowledge_base_id
    if not knowledge_base_id:        
        client = boto3.client('bedrock-agent')         
        response = client.list_knowledge_bases(
            maxResults=10
        )
        print('response: ', response)
                
        if "knowledgeBaseSummaries" in response:
            summaries = response["knowledgeBaseSummaries"]
            for summary in summaries:
                if summary["name"] == knowledge_base_name:
                    knowledge_base_id = summary["knowledgeBaseId"]
                    print('knowledge_base_id: ', knowledge_base_id)
                    break
    
    relevant_docs = []
    if knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 2}},
        )
        
        relevant_docs = retriever.invoke(query)
        print(relevant_docs)
    
    docs = []
    for i, document in enumerate(relevant_docs):
        #print(f"{i}: {document.page_content}")
        print_doc(i, document)
        if document.page_content:
            excerpt = document.page_content
        
        score = document.metadata["score"]
        # print('score:', score)
        doc_prefix = "knowledge-base"
        
        link = ""
        if "s3Location" in document.metadata["location"]:
            link = document.metadata["location"]["s3Location"]["uri"] if document.metadata["location"]["s3Location"]["uri"] is not None else ""
            
            # print('link:', link)    
            pos = link.find(f"/{doc_prefix}")
            name = link[pos+len(doc_prefix)+1:]
            encoded_name = parse.quote(name)
            # print('name:', name)
            link = f"{path}{doc_prefix}{encoded_name}"
            
        elif "webLocation" in document.metadata["location"]:
            link = document.metadata["location"]["webLocation"]["url"] if document.metadata["location"]["webLocation"]["url"] is not None else ""
            name = "Web Crawler"

        # print('link:', link)                    

        docs.append(
            Document(
                page_content=excerpt,
                metadata={
                    'name': name,
                    'url': link,
                    'from': 'RAG'
                },
            )
        )
    return docs

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
    # print(f"score: {score}")
    
    grade = score.binary_score    
    if grade == 'yes':
        print("---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:  # no
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        conn.send(None)
    
    conn.close()
                                    
def grade_documents_using_parallel_processing(question, documents):
    global selected_chat
    
    filtered_docs = []    

    processes = []
    parent_connections = []
    
    for i, doc in enumerate(documents):
        #print(f"grading doc[{i}]: {doc.page_content}")        
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        process = Process(target=grade_document_based_on_relevance, args=(child_conn, question, doc, multi_region_models, selected_chat))
        processes.append(process)

        selected_chat = selected_chat + 1
        if selected_chat == len(multi_region_models):
            selected_chat = 0
    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        relevant_doc = parent_conn.recv()

        if relevant_doc is not None:
            filtered_docs.append(relevant_doc)

    for process in processes:
        process.join()
    
    #print('filtered_docs: ', filtered_docs)
    return filtered_docs
    
def print_doc(i, doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content
            
    print(f"{i}: {text}, metadata:{doc.metadata}")
    
def grade_documents(question, documents):
    print("###### grade_documents ######")
    
    filtered_docs = []
    if multi_region == 'enable':  # parallel processing
        print("start grading...")
        filtered_docs = grade_documents_using_parallel_processing(question, documents)

    else:
        # Score each doc    
        chat = get_chat()
        retrieval_grader = get_retrieval_grader(chat)
        for i, doc in enumerate(documents):
            print_doc(i, doc)
            
            score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
            # print("score: ", score)
            
            grade = score.binary_score
            # print("grade: ", grade)
            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                # We do not include the document in filtered_docs
                # We set a flag to indicate that we want to run web search
                continue
    
    global reference_docs 
    reference_docs += filtered_docs    
    # print('langth of reference_docs: ', len(reference_docs))
    
    # print('len(docments): ', len(filtered_docs))    
    return filtered_docs

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

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

def get_answer_grader():
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )
        
    chat = get_chat()
    structured_llm_grade_answer = chat.with_structured_output(GradeAnswer)
        
    system = (
        "You are a grader assessing whether an answer addresses / resolves a question."
        "Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."
    )
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )
    answer_grader = answer_prompt | structured_llm_grade_answer
    return answer_grader
    
def get_hallucination_grader():    
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )
        
    system = (
        "You are a grader assessing whether an LLM generation is grounded in supported by a set of retrieved facts."
        "Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in supported by the set of facts."
    )    
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
        
    chat = get_chat()
    structured_llm_grade_hallucination = chat.with_structured_output(GradeHallucinations)
        
    hallucination_grader = hallucination_prompt | structured_llm_grade_hallucination
    return hallucination_grader
    
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def get_reg_chain(langMode):
    if langMode:
        system = (
        """다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.

        <context>
        {context}
        </context>""")
    else: 
        system = (
        """Here is pieces of context, contained in <context> tags. Provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        <context>
        {context}
        </context>""")
        
    human = "{question}"
        
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
                    
    chat = get_chat()
    rag_chain = prompt | chat
    return rag_chain

def traslation(chat, text, input_language, output_language):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        
        msg = result.content
        # print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

####################### LangGraph #######################
# RAG functions
#########################################################
MAX_REVISIONS = 1
class State(TypedDict):
    query: str
    draft: str
    docs: List[str]
    reflection : List[str]
    sub_queries : List[str]
    revision_number: int

def retrieve_node(state: State):
    print("###### retrieve ######")
    query = state['query']
    relevant_docs = retrieve_from_knowledge_base(query)
    
    # print(f'q: {query}, RAG: {relevant_docs}')
    print(f'--> query: {query}, length: {len(relevant_docs)}')
    
    filtered_docs = []
    if len(relevant_docs):
        filtered_docs = grade_documents(query, relevant_docs)
    print('length of filtered_docs: ', len(filtered_docs))
    
    return {
        "docs": filtered_docs
    }

class RetrieveState(TypedDict):
    sub_query: str

def retrieve_query(state: RetrieveState):    
    print("###### retrieve_query ######")
    print('state (retrieve_query): ', state)
    
    sub_query = state['sub_query']
    relevant_docs = retrieve_from_knowledge_base(sub_query)
    
    # print(f'sub_query: {sub_query}, RAG: {relevant_docs}')
    print(f'--> sub_query: {sub_query}, length: {len(relevant_docs)}')
    for i, doc in enumerate(relevant_docs):
        if len(doc.page_content)>=100:
            print(f"{i}, doc: {doc.page_content[:100]}")
        else:
            print(f"{i}, doc: {doc.page_content}")
    
    filtered_docs = []
    if len(relevant_docs):
        filtered_docs = grade_documents(sub_query, relevant_docs)
    print('length of filtered_docs: ', len(filtered_docs))

    # web search
    #for q in sub_queries:
    #    docs = tavily_search(q, 4)
    #    print(f'q: {q}, WEB: {docs}')
        
    #    if len(docs):
    #        filtered_docs += grade_documents(q, docs)
    
    return {
        "docs": filtered_docs
    }

def generate_node(state: State):
    print("###### generate ######")
    query = state["query"]
    docs = state["docs"]
    print('query: ', query)
    print('docs: ', docs)
        
    # RAG generation
    rag_chain = get_reg_chain(isKorean(query))
        
    answer = rag_chain.invoke({"context": docs, "question": query})
    print('answer: ', answer.content)
            
    return {"query": query, "draft": answer.content}
        
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

    reflection: ReflectionKor = Field(description="작성된 답변에 대한 평가")
    sub_queries: list[str] = Field(
        description="답변과 관련된 3개 이내의 검색어"
    )
    
def reflect_node(state: State):
    print("###### reflect ######")
    #print('state: ', state)    
    query = state['query']
    print('query: ', query)
    draft = state['draft']
    print('draft: ', draft)
        
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
        print(f'attempt: {attempt}, info: {info}')
                
        if not info['parsed'] == None:
            parsed_info = info['parsed']
            # print('reflection: ', parsed_info.reflection)
            reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
            sub_queries = parsed_info.sub_queries
                
            print('reflection: ', parsed_info.reflection)            
        
            """
            if isKorean(draft):
                translated_search = []
                for q in sub_queries:
                    chat = get_chat()
                    if isKorean(q):
                        search = traslation(chat, q, "Korean", "English")
                    else:
                        search = traslation(chat, q, "English", "Korean")
                    translated_search.append(search)
                        
                print('translated_search: ', translated_search)
                sub_queries += translated_search
            """
            break
    print('sub_queries: ', sub_queries)
        
    return {
        "reflection": reflection,
        "sub_queries": sub_queries,
    }

def revise_node(state: State):   
    print("###### revise ######")
        
    draft = state['draft']
    reflection = state['reflection']
    sub_queries = state['sub_queries']
    print('draft: ', draft)
    print('reflection: ', reflection)
    print('sub_queries: ', sub_queries)
        
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
            # "Provide the final answer using Korean with <result> tag."
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
    
    filtered_docs = state['docs']
    print('filtered_docs: ', filtered_docs)
              
    content = []   
    if len(filtered_docs):
        for d in filtered_docs:
            content.append(d.page_content)        
    print('content: ', content)

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
    # print('output: ', output)
        
    revised_draft = output[output.find('<result>')+8:len(output)-9]
    # print('revised_draft: ', revised_draft) 
            
    if revised_draft.find('#')!=-1 and revised_draft.find('#')!=0:
        revised_draft = revised_draft[revised_draft.find('#'):]

    print('--> draft: ', draft)
    print('--> reflection: ', reflection)
    print('--> revised_draft: ', revised_draft)
        
    revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
            
    return {
        "draft": revised_draft,
        "revision_number": revision_number + 1
    }

def rewrite_node(state: State):
    print("###### rewrite ######")
    query = state['query']
    
    return {
        "query": query
    }
    
def decompose_node(state: State):
    print("###### decompose ######")
    query = state['query']
    
    return {
        "sub_queries": [query]
    }    
    
####################### LangGraph #######################
# RAG with Reflection
#########################################################

def continue_to_retrieve_query(state: State):
    print('###### continue_to_retrieve_query ######')
    print('state (continue_to_retrieve_query): ', state)
    
    revise_request = []
    for idx, sub_query in enumerate(state["sub_queries"]):
        print(f"draft[{idx}]: {sub_query}")
        
        if sub_query:
            revise_request.append(Send("retrieve_query", {
                "sub_query": sub_query
            }))
    
    print('revise_request: ', revise_request)
    
    return revise_request

def should_to_reflection(state: State, config):
    print("###### should_to_reflection ######")
    max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
    print("max_revisions: ", max_revisions)
            
    if state["revision_number"] > max_revisions:
        return "end"
    return "continue"

def buildRagWithReflection():
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("generate_node", generate_node)
    workflow.add_node("reflect_node", reflect_node)
    workflow.add_node("retrieve_query", retrieve_query)    
    workflow.add_node("revise_node", revise_node)

    # Set entry point
    workflow.set_entry_point("retrieve_node")
    
    workflow.add_edge("retrieve_node", "generate_node")
    workflow.add_edge("generate_node", "reflect_node")
    
    workflow.add_conditional_edges(
        "reflect_node", 
        continue_to_retrieve_query, 
        ["retrieve_query"]
    )
        
    # Add edges
    workflow.add_edge("retrieve_query", "revise_node")
    
    workflow.add_conditional_edges(
        "revise_node", 
        should_to_reflection, 
        {
            "end": END, 
            "continue": "reflect_node"}
    )
        
    return workflow.compile()
    
def run_rag_with_reflection(connectionId, requestId, query):    
    app = buildRagWithReflection()
    
    # Run the workflow
    isTyping(connectionId, requestId)        
    inputs = {
        "query": query
    }    
    config = {
        "recursion_limit": 50
    }
    
    output = app.invoke(inputs, config)
    print('output (run_rag_with_reflection): ', output)
    
    return output['final_answer']

####################### LangGraph #######################
# RAG with query trasnformation
#########################################################

def continue_to_retrieve_node(state: State):
    print('###### continue_to_retrieve_query ######')
    print('state (continue_to_retrieve_query): ', state)
    
    revise_request = []
    for idx, sub_query in enumerate(state["sub_queries"]):
        print(f"draft[{idx}]: {sub_query}")
        
        if sub_query:
            revise_request.append(Send("retrieve_node", {
                "sub_query": sub_query
            }))
    
    print('revise_request: ', revise_request)
    
    return revise_request

def buildRagWithTransformation():
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("rewrite_node", rewrite_node)
    workflow.add_node("decompose_node", decompose_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("generate_node", generate_node)
    
    # Set entry point
    workflow.set_entry_point("rewrite_node")
    
    workflow.add_edge("rewrite_node", "decompose_node")
    
    workflow.add_conditional_edges(
        "decompose_node", 
        continue_to_retrieve_node, 
        ["retrieve_node"]
    )

    # Add edges
    workflow.add_edge("retrieve_node", "generate_node")
    workflow.add_edge("generate_node", END)
            
    return workflow.compile()

def run_rag_with_transformation(connectionId, requestId, query):    
    app = buildRagWithTransformation()
    
    # Run the workflow
    isTyping(connectionId, requestId)        
    inputs = {
        "query": query
    }    
    config = {
        "recursion_limit": 50
    }
    
    output = app.invoke(inputs, config)
    print('output (run_rag_with_reflection): ', output)
    
    return output['final_answer']

#########################################################                    
def revise_question(connectionId, requestId, chat, query):    
    global history_length, token_counter_history    
    history_length = token_counter_history = 0
        
    if isKorean(query)==True :      
        system = (
            ""
        )  
        human = """이전 대화를 참조하여, 다음의 <question>의 뜻을 명확히 하는 새로운 질문을 한국어로 생성하세요. 새로운 질문은 원래 질문의 중요한 단어를 반드시 포함합니다. 결과는 <result> tag를 붙여주세요.
        
        <question>            
        {question}
        </question>"""
        
    else: 
        system = (
            ""
        )
        human = """Rephrase the follow up <question> to be a standalone question. Put it in <result> tags.
        <question>            
        {question}
        </question>"""
            
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "history": history,
                "question": query,
            }
        )
        generated_question = result.content
        
        revised_question = generated_question[generated_question.find('<result>')+8:len(generated_question)-9] # remove <result> tag                   
        print('revised_question: ', revised_question)
        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")

    if debugMessageMode == 'true':  
        chat_history = ""
        for dialogue_turn in history:
            #print('type: ', dialogue_turn.type)
            #print('content: ', dialogue_turn.content)
            
            dialog = f"{dialogue_turn.type}: {dialogue_turn.content}\n"            
            chat_history = chat_history + dialog
                
        history_length = len(chat_history)
        print('chat_history length: ', history_length)
        
        token_counter_history = 0
        if chat_history:
            token_counter_history = chat.get_num_tokens(chat_history)
            print('token_size of history: ', token_counter_history)
            
        sendDebugMessage(connectionId, requestId, f"새로운 질문: {revised_question}\n * 대화이력({str(history_length)}자, {token_counter_history} Tokens)을 활용하였습니다.")
            
    return revised_question    
    # return revised_question.replace("\n"," ")

def isTyping(connectionId, requestId):    
    msg_proceeding = {
        'request_id': requestId,
        'msg': 'Proceeding...',
        'status': 'istyping'
    }
    #print('result: ', json.dumps(result))
    sendMessage(connectionId, msg_proceeding)
            
def readStreamMsg(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            #print('event: ', event)
            msg = msg + event
            
            result = {
                'request_id': requestId,
                'msg': msg,
                'status': 'proceeding'
            }
            #print('result: ', json.dumps(result))
            sendMessage(connectionId, result)
    # print('msg: ', msg)
    return msg
    
def sendMessage(id, body):
    # print('sendMessage size: ', len(body))
    try:
        client.post_to_connection(
            ConnectionId=id, 
            Data=json.dumps(body)
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('err_msg: ', err_msg)
        # raise Exception ("Not able to send a message")

def sendResultMessage(connectionId, requestId, msg):    
    result = {
        'request_id': requestId,
        'msg': msg,
        'status': 'completed'
    }
    #print('debug: ', json.dumps(debugMsg))
    sendMessage(connectionId, result)
    
def sendDebugMessage(connectionId, requestId, msg):
    debugMsg = {
        'request_id': requestId,
        'msg': msg,
        'status': 'debug'
    }
    #print('debug: ', json.dumps(debugMsg))
    sendMessage(connectionId, debugMsg)    
        
def sendErrorMessage(connectionId, requestId, msg):
    errorMsg = {
        'request_id': requestId,
        'msg': msg,
        'status': 'error'
    }
    print('error: ', json.dumps(errorMsg))
    sendMessage(connectionId, errorMsg)    

def load_chat_history(userId, allowTime):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    # print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text':
            memory_chain.chat_memory.add_user_message(text)
            memory_chain.chat_memory.add_ai_message(msg)     

def translate_text(chat, text):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    if isKorean(text)==False :
        input_language = "English"
        output_language = "Korean"
    else:
        input_language = "Korean"
        output_language = "English"
                        
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        msg = result.content
        print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def use_multimodal(img_base64, query):    
    multimodal = get_multimodal()
    
    if query == "":
        query = "그림에 대해 상세히 설명해줘."
    
    messages = [
        SystemMessage(content="답변은 500자 이내의 한국어로 설명해주세요."),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = multimodal.invoke(messages)
        
        summary = result.content
        print('result of code summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def extract_text(chat, img_base64):    
    query = "텍스트를 추출해서 utf8로 변환하세요. <result> tag를 붙여주세요."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = chat.invoke(messages)
        
        extracted_text = result.content
        print('result of text extraction from an image: ', extracted_text)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return extracted_text

def getResponse(connectionId, jsonBody):
    print('jsonBody: ', jsonBody)
    
    userId  = jsonBody['user_id']
    print('userId: ', userId)
    requestId  = jsonBody['request_id']
    print('requestId: ', requestId)
    requestTime  = jsonBody['request_time']
    print('requestTime: ', requestTime)
    type  = jsonBody['type']
    print('type: ', type)
    body = jsonBody['body']
    print('body: ', body)
    convType = jsonBody['convType']
    print('convType: ', convType)
    
    global multi_region    
    if "multi_region" in jsonBody:
        multi_region = jsonBody['multi_region']
    print('multi_region: ', multi_region)
    
    global rag_state    
    if "rag" in jsonBody:
        rag_state = jsonBody['rag']
    else:
        rag_state = 'disable'
    print('rag_state: ', rag_state)
        
    print('initiate....')
    global reference_docs
    reference_docs = []

    global map_chain, memory_chain
    
    # Multi-LLM
    if multi_region == 'enable':
        profile = multi_region_models[selected_chat]
    else:
        profile = LLM_for_chat[selected_chat]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    # print(f'selected_chat: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}')
    # print('profile: ', profile)
    
    chat = get_chat()    
    
    # create memory
    if userId in map_chain:  
        print('memory exist. reuse it!')
        memory_chain = map_chain[userId]
    else: 
        print('memory does not exist. create new one!')        
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
        map_chain[userId] = memory_chain

        allowTime = getAllowTime()
        load_chat_history(userId, allowTime)
    
    start = int(time.time())    

    msg = ""
    reference = ""
    
    if type == 'text' and body[:11] == 'list models':
        bedrock_client = boto3.client(
            service_name='bedrock',
            region_name=bedrock_region,
        )
        modelInfo = bedrock_client.list_foundation_models()    
        print('models: ', modelInfo)

        msg = f"The list of models: \n"
        lists = modelInfo['modelSummaries']
        
        for model in lists:
            msg += f"{model['modelId']}\n"
        
        msg += f"current model: {modelId}"
        print('model lists: ', msg)    
    else:             
        if type == 'text':
            text = body
            print('query: ', text)

            querySize = len(text)
            textCount = len(text.split())
            print(f"query size: {querySize}, words: {textCount}")

            if text == 'clearMemory':
                memory_chain.clear()
                map_chain[userId] = memory_chain
                    
                print('initiate the chat memory!')
                msg  = "The chat memory was intialized in this session."
            else:            
                if convType == 'normal':      # normal
                    msg = general_conversation(connectionId, requestId, chat, text)                  

                elif convType == 'rag-with-reflection':  # rag-with-reflection
                    msg = run_rag_with_reflection(connectionId, requestId, text)

                elif convType == 'rag-with-transformation':  # rag-with-transformation
                    msg = run_rag_with_transformation(connectionId, requestId, text)

                memory_chain.chat_memory.add_user_message(text)
                memory_chain.chat_memory.add_ai_message(msg)
                
        elif type == 'document':
            isTyping(connectionId, requestId)
            
            object = body
            file_type = object[object.rfind('.')+1:len(object)]            
            print('file_type: ', file_type)
            
            if file_type == 'csv':
                if not convType == "bedrock-agent":
                    docs = load_csv_document(object)
                    contexts = []
                    for doc in docs:
                        contexts.append(doc.page_content)
                    print('contexts: ', contexts)
                
                    msg = get_summary(chat, contexts)

            elif file_type == 'pdf' or file_type == 'txt' or file_type == 'md' or file_type == 'pptx' or file_type == 'docx':
                texts = load_document(file_type, object)

                docs = []
                for i in range(len(texts)):
                    docs.append(
                        Document(
                            page_content=texts[i],
                            metadata={
                                'name': object,
                                # 'page':i+1,
                                'url': path+doc_prefix+parse.quote(object)
                            }
                        )
                    )
                print('docs[0]: ', docs[0])    
                print('docs size: ', len(docs))

                contexts = []
                for doc in docs:
                    contexts.append(doc.page_content)
                print('contexts: ', contexts)

                msg = get_summary(chat, contexts)
                                
            elif file_type == 'png' or file_type == 'jpeg' or file_type == 'jpg':
                print('multimodal: ', object)
                
                s3_client = boto3.client('s3') 
                    
                image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix+'/'+object)
                # print('image_obj: ', image_obj)
                
                image_content = image_obj['Body'].read()
                img = Image.open(BytesIO(image_content))
                
                width, height = img.size 
                print(f"width: {width}, height: {height}, size: {width*height}")
                
                isResized = False
                while(width*height > 5242880):                    
                    width = int(width/2)
                    height = int(height/2)
                    isResized = True
                    print(f"width: {width}, height: {height}, size: {width*height}")
                
                if isResized:
                    img = img.resize((width, height))
                
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                command = ""        
                if 'command' in jsonBody:
                    command  = jsonBody['command']
                    print('command: ', command)
                
                # verify the image
                msg = use_multimodal(img_base64, command)       
                
                # extract text from the image
                text = extract_text(chat, img_base64)
                extracted_text = text[text.find('<result>')+8:len(text)-9] # remove <result> tag
                print('extracted_text: ', extracted_text)
                if len(extracted_text)>10:
                    msg = msg + f"\n\n[추출된 Text]\n{extracted_text}\n"
                
                memory_chain.chat_memory.add_user_message(f"{object}에서 텍스트를 추출하세요.")
                memory_chain.chat_memory.add_ai_message(extracted_text)
            
            else:
                msg = "uploaded file: "+object
        
        sendResultMessage(connectionId, requestId, msg+reference)
        # print('msg+reference: ', msg+reference)    
                
        elapsed_time = int(time.time()) - start
        print("total run time(sec): ", elapsed_time)
        
        print('msg: ', msg)

        item = {
            'user_id': {'S':userId},
            'request_id': {'S':requestId},
            'request_time': {'S':requestTime},
            'type': {'S':type},
            'body': {'S':body},
            'msg': {'S':msg+reference}
        }
        client = boto3.client('dynamodb')
        try:
            resp =  client.put_item(TableName=callLogTableName, Item=item)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)
            # raise Exception ("Not able to write into dynamodb")         
        #print('resp, ', resp)

    return msg, reference

def lambda_handler(event, context):
    # print('event: ', event)
    
    msg = ""
    if event['requestContext']: 
        connectionId = event['requestContext']['connectionId']        
        routeKey = event['requestContext']['routeKey']
        
        if routeKey == '$connect':
            print('connected!')
        elif routeKey == '$disconnect':
            print('disconnected!')
        else:
            body = event.get("body", "")
            #print("data[0:8]: ", body[0:8])
            if body[0:8] == "__ping__":
                # print("keep alive!")                
                sendMessage(connectionId, "__pong__")
            else:
                print('connectionId: ', connectionId)
                print('routeKey: ', routeKey)
        
                jsonBody = json.loads(body)
                print('request body: ', json.dumps(jsonBody))

                requestId  = jsonBody['request_id']
                try:
                    msg, reference = getResponse(connectionId, jsonBody)

                    print('msg+reference: ', msg+reference)
                                        
                except Exception:
                    err_msg = traceback.format_exc()
                    print('err_msg: ', err_msg)

                    sendErrorMessage(connectionId, requestId, err_msg)    
                    raise Exception ("Not able to send a message")

    return {
        'statusCode': 200
    }
