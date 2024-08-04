import chromadb
from chromadb.utils import embedding_functions
import faiss
from sentence_transformers import SentenceTransformer
import time
import json
import os

# ChromaDB setup
def setup_chromadb(persist_directory="./chroma_db"):
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
    
    collection_name = "agentd_collection"
    if collection_name in [col.name for col in chroma_client.list_collections()]:
        collection = chroma_client.get_collection(name=collection_name)
    else:
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    
    return collection

# Query ChromaDB
def query_database(query_texts, n_results=1):
    results = collection.query(
        query_texts=query_texts,
        n_results=n_results
    )
    return results

# Semantic Cache implementation
def init_cache():
    index = faiss.IndexFlatL2(768)
    if index.is_trained:
        print("Index trained")
    
    encoder = SentenceTransformer("all-mpnet-base-v2")
    
    return index, encoder

def retrieve_cache(json_file):
    try:
        with open(json_file, "r") as file:
            cache = json.load(file)
    except FileNotFoundError:
        cache = {"questions": [], "embeddings": [], "answers": [], "response_text": []}
    
    return cache

def store_cache(json_file, cache):
    with open(json_file, "w") as file:
        json.dump(cache, file)

class semantic_cache:
    def __init__(self, json_file="cache_file.json", threshold=0.35, max_responses=100, eviction_policy=None):
        self.index, self.encoder = init_cache()
        self.euclidean_threshold = threshold
        self.json_file = json_file
        self.cache = retrieve_cache(self.json_file)
        self.max_responses = max_responses
        self.eviction_policy = eviction_policy

    def evict(self):
        if self.eviction_policy and len(self.cache["questions"]) > self.max_responses:
            for _ in range((len(self.cache["questions"]) - self.max_responses)):
                if self.eviction_policy == "FIFO":
                    self.cache["questions"].pop(0)
                    self.cache["embeddings"].pop(0)
                    self.cache["answers"].pop(0)
                    self.cache["response_text"].pop(0)

    def ask(self, question: str) -> str:
        start_time = time.time()
        try:
            embedding = self.encoder.encode([question])
            
            self.index.nprobe = 8
            D, I = self.index.search(embedding, 1)
            
            if D[0] >= 0:
                if I[0][0] >= 0 and D[0][0] <= self.euclidean_threshold:
                    row_id = int(I[0][0])
                    
                    print("Answer recovered from Cache.")
                    print(f"{D[0][0]:.3f} smaller than {self.euclidean_threshold}")
                    print(f"Found cache in row: {row_id} with score {D[0][0]:.3f}")
                    print(f"response_text: " + self.cache["response_text"][row_id])
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Time taken: {elapsed_time:.3f} seconds")
                    return self.cache["response_text"][row_id]
            
            answer = query_database([question], 1)
            response_text = answer["documents"][0][0] if answer["documents"] else "No answer found in the database."
            
            self.cache["questions"].append(question)
            self.cache["embeddings"].append(embedding[0].tolist())
            self.cache["answers"].append(answer)
            self.cache["response_text"].append(response_text)
            
            print("Answer recovered from ChromaDB.")
            print(f"response_text: {response_text}")
            
            self.index.add(embedding)
            
            self.evict()
            
            store_cache(self.json_file, self.cache)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time:.3f} seconds")
            
            return response_text
        except Exception as e:
            raise RuntimeError(f"Error during 'ask' method: {e}")

# Main setup and examples
if __name__ == "__main__":
    # Setup ChromaDB
    current_dir = os.getcwd()
    chroma_db_path = os.path.join(current_dir, "chroma_db")
    collection = setup_chromadb(chroma_db_path)
    print(f"ChromaDB set up in: {chroma_db_path}")
    print(f"Collection 'agentd_collection' is ready to use.")

    # Initialize semantic cache
    cache_file = os.path.join(current_dir, "semantic_cache.json")
    cache = semantic_cache(cache_file, threshold=0.35, max_responses=100, eviction_policy="FIFO")
    print(f"Semantic cache initialized with file: {cache_file}")

    print("\nRAG system with semantic cache is ready to use.")

    # Example 1: Adding documents to the collection
    print("\nExample 1: Adding documents to the collection")
    collection.add(
        documents=[
            "Paris is the capital and most populous city of France.",
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
            "The Louvre is the world's largest art museum and a historic monument in Paris.",
            "London is the capital and largest city of England and the United Kingdom.",
            "The British Museum in London houses a vast collection of world art and artifacts."
        ],
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )
    print("Documents added to the collection.")

    # Example 2: Querying the system
    print("\nExample 2: Querying the system")
    questions = [
        "What is the capital of France?",
        "Tell me about the Eiffel Tower.",
        "What's the largest museum in Paris?",
        "What's the capital of the UK?",
        "What can I find in the British Museum?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        result = cache.ask(question)
        print(f"Answer: {result}")

    # Example 3: Demonstrating cache hit
    print("\nExample 3: Demonstrating cache hit")
    similar_questions = [
        "Can you tell me the capital city of France?",
        "What do you know about the Eiffel Tower in Paris?",
        "Which is the biggest art museum in Paris?",
        "What is the capital city of the United Kingdom?",
        "What kind of artifacts are in the British Museum?"
    ]

    for question in similar_questions:
        print(f"\nQuestion: {question}")
        result = cache.ask(question)
        print(f"Answer: {result}")

    # Example 4: Querying with a new topic
    print("\nExample 4: Querying with a new topic")
    new_question = "What is the population of Tokyo?"
    print(f"Question: {new_question}")
    result = cache.ask(new_question)
    print(f"Answer: {result}")

    print("\nRAG system demonstration completed.")