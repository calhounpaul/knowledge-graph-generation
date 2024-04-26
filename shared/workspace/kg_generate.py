import os, sys, json, shutil, getpass, atexit, time, hashlib

from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, load_index_from_storage, load_graph_from_storage
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
#from llama_index.llms.ollama import Ollama
from llama_index.llms.huggingface import HuggingFaceLLM, HuggingFaceInferenceAPI
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
#from llama_index.embeddings.ollama import OllamaEmbedding
import torch

this_dir = container_workspace_dir = os.path.dirname(os.path.abspath(__file__))

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

RESET_PERSIST_DIR = False
RESET_VECTOR_STORE = False
RESET_GRAPH_STORE = False

def fix_gutenberg(text):
    chunks = text.split("PROJECT GUTENBERG")
    largest_chunk = ""
    for chunk in chunks:
        if len(chunk) > len(largest_chunk):
            largest_chunk = chunk
    largest_chunk = "\n".join([line.strip() for line in largest_chunk.split("\n")])
    largest_chunk = largest_chunk.replace("\n\n", "<<DOUBLELINEBREAK>>")
    largest_chunk = largest_chunk.replace("\n", " ")
    largest_chunk = largest_chunk.replace("<<DOUBLELINEBREAK>>", "\n\n")
    return "\n".join(largest_chunk.split("\n")[1:-1]).strip()

if not os.path.exists("data"):
    os.makedirs("data")
with open("urls_list.txt") as f:
    urls = f.readlines()
    for url in urls:
        url = url.strip()
        filename = url.split("/")[-1]
        if not os.path.exists(f"data/{filename}"):
            os.system(f"wget {url} -P data")
            with open(f"data/{filename}") as f:
                text = f.read()
                text = fix_gutenberg(text)
            with open(f"data/{filename}", "w") as f:
                f.write(text)

documents = SimpleDirectoryReader("data").load_data()

"""Settings.embed_model = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)"""

Settings.embed_model = HuggingFaceEmbedding(
    model_name="mixedbread-ai/mxbai-embed-large-v1", #"BAAI/bge-m3",
    embed_batch_size=32,
    normalize=True,
)

#Settings.llm = Ollama(model="llama3:8b-instruct-q4_1", request_timeout=999.0)


Settings.chunk_size = 768
Settings.chunk_overlap = 256

chunker = SemanticSplitterNodeParser(
    embed_model=Settings.embed_model,
    breakpoint_percentile_threshold=95,
    buffer_size=1,
    #sentence_splitter=SentenceSplitter.from_defaults(),
)

recreate_vector_store = False

if RESET_VECTOR_STORE or not os.path.exists("tmp/persist_dir/default__vector_store.json"):
    if os.path.exists("tmp/persist_dir/default__vector_store.json"):
        os.remove("tmp/persist_dir/default__vector_store.json")
    recreate_vector_store = True

recreate_graph_store = False
if RESET_GRAPH_STORE or not os.path.exists("tmp/persist_dir/graph_store.json"):
    if os.path.exists("tmp/persist_dir/graph_store.json"):
        os.remove("tmp/persist_dir/graph_store.json")
    recreate_graph_store = True

if RESET_PERSIST_DIR or not os.path.exists("tmp/persist_dir"):
    if os.path.exists("tmp/persist_dir"):
        shutil.rmtree("tmp/persist_dir")
    recreate_vector_store = True

if recreate_vector_store:
    #parser = SentenceSplitter()
    #nodes = parser.get_nodes_from_documents(documents)

    nodes = chunker.build_semantic_nodes_from_documents(documents)

    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore(),
        vector_store=SimpleVectorStore(),
        index_store=SimpleIndexStore(),
    )

    storage_context.docstore.add_documents(nodes)

    index = VectorStoreIndex(nodes, storage_context=storage_context)

    index.storage_context.persist(persist_dir="tmp/persist_dir")

else:
    storage_context = StorageContext.from_defaults(persist_dir="tmp/persist_dir")

#index = load_index_from_storage(storage_context)

#retriever = index.as_retriever()
#nodes = retriever.retrieve("news of Beth's death")

#for node in nodes:
#    print(node.text)

Settings.llm = HuggingFaceLLM(
    context_window=3800,
    max_new_tokens=256,
    generate_kwargs={ "do_sample": True}, #"temperature": 0.1,
    #tokenizer_name="microsoft/Phi-3-mini-128k-instruct",
    #model_name="microsoft/Phi-3-mini-128k-instruct",
    tokenizer_name="microsoft/Phi-3-mini-4k-instruct",
    model_name="microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    #stopping_ids=[50278, 50279, 50277, 1, 0],
    stopping_ids=[32000, 32007, 2, 1, 0],
    tokenizer_kwargs={"max_length": 3800},
    model_kwargs={"torch_dtype": "auto","trust_remote_code": True,},
)

if recreate_graph_store:
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=10,
        storage_context=storage_context,
        include_embeddings=True,
    )

    kg_index.storage_context.persist(persist_dir="tmp/persist_dir")

    try:
        print("index_id:",kg_index.index_id)
    except:
        try:
            print("index_id:",kg_index.id)
        except:
            pass
    #print("ROOT_ID:",kg_index.root_id)
else:
    #kg_index = load_graph_from_storage(storage_context)
    kg_index = load_index_from_storage(storage_context,index_id="45b43b0b-f7dc-4d95-85bc-ea1bdb1ca6a2")

query_engine = kg_index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
)

with open("tmp/test_qs.txt") as f:
    questions = f.readlines()
    questions = [q.strip() for q in questions[:42]]
    for question in questions:
        print(f"Question: {question}")
        streaming_response = query_engine.query(
            question
        )
        print("Response:",end=" ",flush=True)
        streaming_response.print_response_stream()
        print("\n----------\n")
