import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
# from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.schema import Document
from langchain import LLMChain

docs_path="annual_reports"
api_keys=["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","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","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","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","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","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","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","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","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","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","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","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"]
api_keys=api_keys[0:6]

#defining the LLMs
llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0)
llm1=ChatGroq(model="mixtral-8x7b-32768",temperature=0)

#Importing the embedding model
model_name="BAAI/bge-small-en"
model_kwargs={"device":"cpu"}
encode_kwargs={"normalize_embeddings":True}
hf_embeddings=HuggingFaceBgeEmbeddings(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)

#Connecting with the neo4j graph
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

#Loading and splitting the annual report
loader = DirectoryLoader(docs_path, glob="**/Torrent-pharma.pdf", loader_cls=PyPDFLoader)
docs=loader.load()
text_splitter = CharacterTextSplitter(
    separator='\n\n',
    chunk_size=325,
    chunk_overlap=50,
)
chunks=text_splitter.split_documents(docs)

#Defining the prompt for content enchancement
prompt='''
Create a concise summary of the following text, including only specific details about {company}. Each point should be standalone and provide clear context.

Output should follow this format:

General context: -context-

Point 1
Point 2
Point 3
Guidelines:

Exclude any general statements or information not directly related to {company}.
Each sentence should contain no more than 10 words.
Use simple, clear language with no missing details or ambiguities.
Avoid adding any information not explicitly mentioned in the text.
The output should be suitable for processing by a graph transformer.
Replace vague terms ('the company,' 'the entity') with '{company}'.


Text:{chunk}
'''
default_prompt=PromptTemplate(template=prompt)

#Defining the graph transformer
entities = [
    "Acquisition", "Board Member", "Business Segment", "Clinical Trial", "Company",
    "Competitor", "Country", "Department", "Disease Area", "Disease", "Dividend",
    "Drug", "Executive", "Facility", "Financial Statement", "Fiscal Year",
    "Geographic Region", "Investment", "Market", "Patent", "Product",
    "Regulatory Body", "Revenue", "Subsidiary", "Supplier", "Technology","Monetary Value"
]

relationships = [
    "ACHIEVED_GROWTH_IN", "TREATS", "DEVELOPED", "EMPLOYS", "FACES_RISK",
    "GENERATES_REVENUE", "HAS_MARKET", "HAS_SUBSIDIARY", "IN_TRIAL",
    "INVESTED_IN", "LEADS", "LICENSES_TO", "MANUFACTURES", "MARKETS",
    "MERGED_WITH", "OWNS", "OWNS_PATENT", "PARTNERS_WITH","LAUNCHED"
    "REGULATED_BY", "REPORTS_TO", "SUPPLIES_TO", "USES_TECHNOLOGY","COMPETES_WITH","HAS_ROLE",
    "CHALLENGES_FACED","REVENUE_GROWTH","LAUNCH"
]
docs_transformer=LLMGraphTransformer(
    llm=llm,
    allowed_nodes=entities,
    allowed_relationships=relationships,
    node_properties=["name","description"],
    strict_mode=False
)

#Code for api key swtiching
def switch_api_key():
    global curapikey
    global docs_transformer
    global llm_chain
    curapikey=(curapikey+1)%len(api_keys)
    os.environ["GROQ_API_KEY"]=api_keys[curapikey]
    llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0)

    llm_chain=LLMChain(llm=llm,prompt=default_prompt)
    
    docs_transformer=LLMGraphTransformer(
    llm=llm,
    allowed_nodes=entities,
    allowed_relationships=relationships,
    node_properties=["name","description"],
    strict_mode=False
    )


#Defining the function for processing the documents
i=0
chunk_count=0
def chunkprocess(chunk):
    graph_docs1=[]
    global chunk_count
    filename=os.path.basename(chunk.metadata['source'])
    chunk_id=f"{filename}.{chunk.metadata['page']}"
    print("Processing -",chunk_id)
    
    company_name=os.path.basename(chunk.metadata['source'])[:-4]
    
    
    
    max_retries=3
    attempt=0
    
    success_chain=False
    
    while not success_chain:
        try:
            refined_text=llm_chain({"chunk":chunk.page_content,"company":company_name})['text']
            success_chain=True
        except Exception as e:
            print("Summarisation error: ",e)
            switch_api_key()

    chunk_embedding=hf_embeddings.embed_query(refined_text)
    properties={
        "filename":filename,
        "chunk_id":chunk_id,
        "text":refined_text,
        "embedding":chunk_embedding
    }
    
    # graph.query("""
    #             MERGE(d:Document{id:$filename})
    #             MERGE(c:Chunk{id:$chunk_id})
    #             SET c.text=$text
    #             MERGE(d)<-[:PART_OF]-(c)
    #             with c
    #             CALL db.create.setNodeVectorProperty(c,'textEmbedding',$embedding)
    #             """,
    #             params=properties)

    success=False
    while not success and attempt<max_retries:
        try:
            doc_input = Document(page_content=refined_text, metadata={"id": filename})
            graph_docs = docs_transformer.convert_to_graph_documents([doc_input])

            print("done")
            success=True
            
        except Exception as e:
            print("Graph generation error: ",e)
            switch_api_key()
            attempt+=1
            if attempt<max_retries:
            
                print("retrying...")
            else:
                print("Max retry limit reached.")

    if success:
        for graph_doc in graph_docs:
            chunknode=Node(
                id=chunk_id,
                type="Chunk"
            )
            
            for node in graph_doc.nodes:
                graph_doc.relationships.append(
                    Relationship(
                        source=chunknode,
                        target=node,
                        type="HAS_ENTITY"
                    )
                )
        graph.query("""
                MERGE(d:Document{id:$filename})
                MERGE(c:Chunk{id:$chunk_id})
                SET c.text=$text
                MERGE(d)<-[:PART_OF]-(c)
                with c
                CALL db.create.setNodeVectorProperty(c,'textEmbedding',$embedding)
                """,
                params=properties)
        # graph.add_graph_documents(graph_docs)
        graph_docs1.append(graph_docs)
        print("Chunk",chunk_count," processed")
    chunk_count+=1
    return graph_docs1

#Implementing multithreading for faster performance
from concurrent.futures import ThreadPoolExecutor,as_completed
import tqdm
MAX_WORKERS = 10
graph_doc2=[]
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures={executor.submit(chunkprocess,chunk):chunk for chunk in chunks}

    for future in tqdm(as_completed(futures),total=len(futures),desc="Processing chunks"):
        graph_doc2.append(future.result())

