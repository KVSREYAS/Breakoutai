from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph


llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0)



# Define a structured prompt template
prompt = PromptTemplate(
    input_variables=["input_text"],
    template=(
        "You are a helpful assistant. Process the input text and generate a structured output in JSON format. "
        "Input: {input_text} \n"
        "Output: {{'key1': value1, 'key2': value2}}"
    )
)


# Define a structured prompt for entity extraction
entity_extraction_prompt = PromptTemplate(
    input_variables=["user_query"],
    template=(
        "You are an intelligent assistant that extracts the main entities from a user's query. "
        "Entities are specific names of things such as people, organizations, products, diseases, or locations "
        "explicitly mentioned in the query. "
        "Do not infer or assume any entities that are not stated in the query. "
        "Avoid extracting generic terms like 'company' unless it is part of a proper noun (e.g., 'Company X').\n"
        "\nProvide the output in the following format:\n"
        "\nEntities: <extracted_entity>\n"
        "\nIf no entity is explicitly mentioned, respond with 'Entities: None'.\n"
        "\nUser Query: {user_query}\n"
        "\nOutput:"
    )
)
#entity extraction
entity_extraction_chain=LLMChain(llm=llm,prompt=entity_extraction_prompt)

def entity_extraction(query):
    result=entity_extraction_chain.run(user_query=query)
    result1 = result[10:]
    entities = [entity.strip() for entity in result1.split(',')]
    return entities


import re
def remove_lucene_chars(text):
    # Regular expression to remove Lucene-specific characters
    return re.sub(r'[+\-!*()"{}[\]^]', '', text)

def generate_full_text_query(input:str)->str:
    full_text_query=""
    words=[el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query+=f"{word}~2 AND"
    full_text_query+=f"{words[-1]}~2"
    return full_text_query.strip()


#Connecting to neo4j database
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

#Checking whether context is sufficient or not
checkprompt='''
You are given a graph context with relationships between entities. Use this context to answer the question.

Make logical inferences based on the relationships.
Simplify the context to include only essential relationships required to answer the question under "New_context" in the SAME FORMAT as the given one as a string.
Do not signal sufficient as yes until you can reach a definitive answer for the question
Enclose the property name in double quotes

The response must strictly follow this format and give output in a json format:
<curly braces>
Reasoning:,
Sufficient: [Yes/No],
Nodes: [Node1, Node2, ...],
New_context: Node1-relation->Node2
<curly braces>

Question : {question}

Context: {context}
'''

finalprompt=PromptTemplate(template=checkprompt)
llm_chain_2=LLMChain(llm=llm,prompt=finalprompt)

#Query for extracting info
query='''CALL db.index.fulltext.queryNodes('entity', $query, {limit: 2})
YIELD node, score
CALL {
    WITH node
    MATCH (node) -[r:!HAS_ENTITY]->(neighbor)
    RETURN node.id + ' [' + apoc.text.join([label IN labels(neighbor) WHERE label <> 'NonChunk'], ', ') + ']' 
           + '-' + type(r) + '->' + neighbor.id + ' [' + apoc.text.join([label IN labels(neighbor) WHERE label <> 'NonChunk'], ', ') + ']' AS output
    UNION ALL
    WITH node
    MATCH (node) <-[r:!HAS_ENTITY]-(neighbor)
    RETURN neighbor.id + ' [' + apoc.text.join([label IN labels(neighbor) WHERE label <> 'NonChunk'], ', ') + ']' 
           + '-' + type(r) + '->' + node.id + ' [' + apoc.text.join([label IN labels(neighbor) WHERE label <> 'NonChunk'], ', ') + ']' AS output
}
RETURN output
'''

#Defining the retriever process
def structured_retriever(question):
    result=""
    entities=entity_extraction(question)
    for entity in entities:
        response=graph.query(
            query,
                {"query":question},
        )
        result+="\n".join([el['output'] for el in response])
    return result

#Prompt for generating the final output
prompt='''You are a knowledgeable assistant that can process graph-based data. Given the following description of a graph, answer the query that follows. The graph is structured with nodes representing entities, and edges representing relationships between those entities. Each node and edge may have attributes that provide more information.
        Do not give a very detailed answer. keep the answer short and concise. Also mention the context u used to arrive at the conclusion in the same format provided in the context

graph description : {context}

query :{query}'''

finalprompt=PromptTemplate(template=prompt)
llm_chain=LLMChain(llm=llm,prompt=finalprompt)

#Culminating all the processes into a single function
def final_chain(question):
    results=entity_extraction(question)

    context=""
    for r in results:
        context+=structured_retriever(r)
    attempts=3
    count=1
    while True:
        print("pass ",count)
        print(context)
        out=json.loads(llm_chain_2({"question":question,"context":context})['text'])
        print(out)
        # print(out)
        try:
            if out['Sufficient']=='Yes' or attempts==0:
                break
            else:
                attempts-=1
                # print(out['Nodes'])
                for i in out['Nodes']:
                    context+=structured_retriever(i)
        except Exception as e:
            break
        print("-----------------")
        count+=1

    # print(context)
    print('\n\n')
    print("Final output")
    print("-------------------------")
    output=llm_chain({"context":context,"query":question})
    return output['text']
    



