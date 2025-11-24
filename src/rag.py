# rag.py
import chromadb
from sentence_transformers import SentenceTransformer
import os
import json # For serializing/deserializing chat history
from typing import List, Dict, Any

# Define a wrapper class for the Sentence Transformer embedding function to satisfy ChromaDB's requirements
class SentenceTransformerChromaEmbeddingFunction:
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        """
        Initializes the SentenceTransformer model.
        Downloads the model if not already cached.
        """
        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            self._model_name = model_name
            print(f"SentenceTransformer model '{model_name}' loaded.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model '{model_name}': {e}")
            raise

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts using the SentenceTransformer model.
        The parameter name 'input' is required by ChromaDB's EmbeddingFunction interface.
        """
        embeddings = self.model.encode(input).tolist()
        return embeddings

    def name(self) -> str:
        """
        Returns the name of the embedding function, as expected by ChromaDB.
        """
        return self._model_name

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self.model.encode(documents).tolist()

    def embed_query(self, input) -> List[float] | List[List[float]]:
        if isinstance(input, str):
            return self.model.encode([input]).tolist()[0]
        return self.model.encode(input).tolist()

class DocumentRag:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initializes the ChromaDB client and collections.
        Uses PersistentClient for local, file-based storage.
        Manages two collections: one for document chunks, one for chat metadata.
        """
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.persist_directory = persist_directory
        except Exception as e:
            fallback_path = persist_directory + "_fallback"
            os.makedirs(fallback_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=fallback_path)
            self.persist_directory = fallback_path
        self.embedding_function = SentenceTransformerChromaEmbeddingFunction()

        # Collection for document chunks and their embeddings
        self.document_chunks_collection = self.client.get_or_create_collection(
            name="document_chunks",
            embedding_function=self.embedding_function
        )
        print(f"ChromaDB collection 'document_chunks' initialized.")

        # Collection for chat session metadata (including chat history)
        # This collection will store chat_id as its document ID and other chat details as metadata
        # No embedding function needed for this collection as it's just for metadata storage
        self.chat_metadata_collection = self.client.get_or_create_collection(
            name="chat_metadata"
        )
        print(f"ChromaDB collection 'chat_metadata' initialized.")


    def add_documents(self, texts: List[str], chat_id: str, metadatas: List[Dict] = None) -> None:
        """
        Adds text documents to the 'document_chunks' ChromaDB collection,
        associating them with a chat_id.

        Args:
            texts (List[str]): A list of text strings to add.
            chat_id (str): The ID of the chat session these documents belong to.
            metadatas (List[Dict], optional): A list of metadata dictionaries,
                                               one for each text. Defaults to None.
        """
        if not texts:
            print("No texts provided to add to ChromaDB.")
            return

        # ChromaDB requires unique IDs for each document within a collection.
        # We'll prefix them with chat_id to ensure uniqueness across chats.
        ids = [f"{chat_id}_chunk_{i}" for i in range(len(texts))]

        # Ensure metadatas list matches the texts list length and add chat_id
        if metadatas is None:
            metadatas = []
            for i in range(len(texts)):
                metadatas.append({"source": "uploaded_document", "chunk_index": i, "chat_id": chat_id})
        else:
            for meta in metadatas:
                meta["chat_id"] = chat_id
            if len(metadatas) != len(texts):
                raise ValueError("Length of metadatas must match length of texts.")

        try:
            self.document_chunks_collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(texts)} documents to 'document_chunks' for chat_id: {chat_id}.")
        except Exception as e:
            print(f"Error adding documents to 'document_chunks' ChromaDB: {e}")

    def query_documents(self, query_text: str, chat_id: str, n_results: int = 1) -> List[str]:
        """
        Queries the 'document_chunks' ChromaDB collection for relevant documents,
        filtered by chat_id.

        Args:
            query_text (str): The query string.
            chat_id (str): The ID of the chat session to filter documents by.
            n_results (int): The number of most relevant results to return.

        Returns:
            List[str]: A list of relevant document content strings.
        """
        if not query_text:
            print("Query text is empty. Returning no results.")
            return []
        
        # Filter by chat_id using the 'where' clause
        where_clause = {"chat_id": chat_id}

        try:
            results = self.document_chunks_collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_clause # Apply the filter here
            )
            # results['documents'][0] contains the list of document contents
            if results and results['documents'] and len(results['documents']) > 0:
                return results['documents'][0]
            else:
                return []
        except Exception as e:
            print(f"Error querying 'document_chunks' ChromaDB: {e}")
            return []

    def save_chat_metadata(self, chat_id: str, chat_data: Dict[str, Any]) -> None:
        """
        Saves or updates chat session metadata in the 'chat_metadata' ChromaDB collection.
        chat_data should include: name, timestamp, document_name, full_document_text,
        summary, ask_history (which will be JSON serialized), processed_document_chunks_count.
        """
        try:
            # ChromaDB stores documents as strings, so we need to serialize the dict
            # The 'documents' parameter is used for the main content, 'metadatas' for filtering
            # For chat_metadata, we'll store the entire dict as a string in 'documents'
            # and use the 'id' for the chat_id.
            
            # Make a copy to avoid modifying the original dict and handle ask_history serialization
            data_to_save = chat_data.copy()
            if 'ask_history' in data_to_save:
                data_to_save['ask_history'] = json.dumps(data_to_save['ask_history'])
            
            # ChromaDB's add/update expects 'documents' to be a list of strings
            # and 'ids' to be a list of strings.
            # We'll use the chat_id as the document ID directly.
            self.chat_metadata_collection.upsert(
                documents=[json.dumps(data_to_save)], # Store the entire dict as a JSON string
                ids=[chat_id]
            )
            print(f"Chat metadata for '{chat_id}' saved/updated in 'chat_metadata' ChromaDB collection.")
        except Exception as e:
            print(f"Error saving chat metadata to ChromaDB: {e}")

    def update_chat_metadata(self, chat_id: str, updates: Dict[str, Any]) -> bool:
        """
        Partially updates fields in chat metadata for the given chat_id.
        Loads existing metadata, merges updates, and upserts back to ChromaDB.
        Returns True on success, False otherwise.
        """
        try:
            existing = self.load_chat_metadata(chat_id) or {"chat_id": chat_id}
            # Merge updates (shallow merge is sufficient for our fields)
            for k, v in updates.items():
                existing[k] = v

            data_to_save = existing.copy()
            if 'ask_history' in data_to_save and isinstance(data_to_save['ask_history'], list):
                data_to_save['ask_history'] = json.dumps(data_to_save['ask_history'])

            self.chat_metadata_collection.upsert(
                documents=[json.dumps(data_to_save)],
                ids=[chat_id]
            )
            print(f"Updated chat metadata for '{chat_id}' with fields: {list(updates.keys())}")
            return True
        except Exception as e:
            print(f"Error updating chat metadata for '{chat_id}': {e}")
            return False

    def load_chat_metadata(self, chat_id: str) -> Dict[str, Any] | None:
        """
        Loads chat session metadata from the 'chat_metadata' ChromaDB collection.
        """
        try:
            results = self.chat_metadata_collection.get(ids=[chat_id])
            if results and results['documents']:
                # Deserialize the JSON string back into a dictionary
                loaded_data = json.loads(results['documents'][0])
                if 'ask_history' in loaded_data:
                    loaded_data['ask_history'] = json.loads(loaded_data['ask_history'])
                return loaded_data
            return None
        except Exception as e:
            print(f"Error loading chat metadata from ChromaDB for '{chat_id}': {e}")
            return None

    def fetch_all_chat_metadata(self) -> List[Dict[str, Any]]:
        """
        Fetches all chat session metadata from the 'chat_metadata' ChromaDB collection.
        """
        try:
            existing = self.chat_metadata_collection.get()
            ids = existing.get('ids', []) if isinstance(existing, dict) else []
            if not ids:
                return []
            results = self.chat_metadata_collection.get(ids=ids)
            chats_list = []
            if results and results['documents']:
                for doc_str in results['documents']:
                    try:
                        chat_data = json.loads(doc_str)
                        if 'ask_history' in chat_data:
                            chat_data['ask_history'] = json.loads(chat_data['ask_history'])
                        chats_list.append(chat_data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding chat metadata JSON: {e} - Data: {doc_str[:100]}...") # Log partial data
            
            # Sort by timestamp (assuming timestamp is present and sortable)
            # Firestore timestamps are objects, but if we store ISO string, it's fine.
            # For simplicity, sorting by name or just using the order from ChromaDB is also an option.
            # If timestamp is a string (ISO format), direct comparison works.
            chats_list.sort(key=lambda x: x.get('timestamp', '0000-00-00T00:00:00'), reverse=True)
            return chats_list
        except Exception as e:
            print(f"Error fetching all chat metadata from ChromaDB: {e}")
            return []

    def delete_documents_by_chat_id(self, chat_id: str) -> None:
        """
        Deletes all documents (chunks and metadata) associated with a specific chat_id.
        """
        try:
            # Delete chunks from document_chunks_collection
            self.document_chunks_collection.delete(where={"chat_id": chat_id})
            print(f"Deleted document chunks for chat_id: {chat_id}.")

            # Delete metadata from chat_metadata_collection
            self.chat_metadata_collection.delete(ids=[chat_id])
            print(f"Deleted chat metadata for chat_id: {chat_id}.")
        except Exception as e:
            print(f"Error deleting data for chat_id {chat_id} from ChromaDB: {e}")

    def clear_all_data(self):
        """
        Deletes all data from both ChromaDB collections.
        """
        try:
            self.client.delete_collection(name="document_chunks")
            self.document_chunks_collection = self.client.get_or_create_collection(
                name="document_chunks",
                embedding_function=self.embedding_function
            )
            print("ChromaDB 'document_chunks' collection cleared and re-initialized.")

            self.client.delete_collection(name="chat_metadata")
            self.chat_metadata_collection = self.client.get_or_create_collection(
                name="chat_metadata"
            )
            print("ChromaDB 'chat_metadata' collection cleared and re-initialized.")
            
            # Optionally, clean up the physical directory
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                print(f"Removed ChromaDB persistence directory: {self.persist_directory}")

        except Exception as e:
            print(f"Error clearing all ChromaDB data: {e}")

# Example usage for testing (this block will only run if rag.py is executed directly)
if __name__ == '__main__':
    # Clean up any previous test data
    if os.path.exists("./test_chroma_db"):
        import shutil
        shutil.rmtree("./test_chroma_db")

    rag = DocumentRag(persist_directory="./test_chroma_db")
    rag.clear_all_data() # Ensure a clean start for testing

    # Test adding documents for different chat IDs
    chat_id_1 = "chat_123"
    dummy_texts_1 = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is a rapidly developing field.",
    ]
    rag.add_documents(dummy_texts_1, chat_id=chat_id_1)

    chat_id_2 = "chat_456"
    dummy_texts_2 = [
        "Machine learning is a subset of AI that focuses on algorithms.",
        "Natural language processing deals with the interaction between computers and human language.",
        "Vector databases are optimized for storing and querying high-dimensional vectors."
    ]
    rag.add_documents(dummy_texts_2, chat_id=chat_id_2)

    print(f"\n--- Testing Querying for chat_id: {chat_id_1} ---")
    query_1 = "What is AI?"
    relevant_chunks_1 = rag.query_documents(query_1, chat_id=chat_id_1, n_results=1)
    print("Relevant chunks found:")
    for chunk in relevant_chunks_1:
        print(f"- {chunk}")

    print(f"\n--- Testing Querying for chat_id: {chat_id_2} ---")
    query_2 = "What are vector databases used for?"
    relevant_chunks_2 = rag.query_documents(query_2, chat_id=chat_id_2, n_results=1)
    print("Relevant chunks found:")
    for chunk in relevant_chunks_2:
        print(f"- {chunk}")
    
    print("\n--- Testing Querying with empty string (should return empty) ---")
    empty_query_results = rag.query_documents("", chat_id=chat_id_1)
    print(f"Results for empty query: {empty_query_results}")

    # Test saving and loading chat metadata
    chat_data_1 = {
        "chat_id": chat_id_1,
        "name": "AI Basics Chat",
        "timestamp": datetime.now().isoformat(),
        "document_name": "AI_Doc.txt",
        "full_document_text": "AI is...",
        "summary": "Summary of AI.",
        "ask_history": [{"question": "What is AI?", "answer": "It's artificial intelligence."}],
        "processed_document_chunks_count": 2
    }
    rag.save_chat_metadata(chat_id_1, chat_data_1)

    chat_data_2 = {
        "chat_id": chat_id_2,
        "name": "ML Concepts Chat",
        "timestamp": datetime.now().isoformat(),
        "document_name": "ML_Doc.pdf",
        "full_document_text": "ML is...",
        "summary": "Summary of ML.",
        "ask_history": [{"question": "What is ML?", "answer": "It's machine learning."}],
        "processed_document_chunks_count": 3
    }
    rag.save_chat_metadata(chat_id_2, chat_data_2)

    print(f"\n--- Testing Loading Chat Metadata for {chat_id_1} ---")
    loaded_chat_1 = rag.load_chat_metadata(chat_id_1)
    print(f"Loaded Chat 1: {loaded_chat_1}")
    assert loaded_chat_1['name'] == "AI Basics Chat"
    assert loaded_chat_1['ask_history'][0]['question'] == "What is AI?"

    print(f"\n--- Testing Fetching All Chat Metadata ---")
    all_chats = rag.fetch_all_chat_metadata()
    print(f"All Chats: {all_chats}")
    assert len(all_chats) == 2

    print(f"\n--- Testing Deleting documents for chat_id: {chat_id_1} ---")
    rag.delete_documents_by_chat_id(chat_id_1)
    # Verify deletion by querying again for chunks
    relevant_chunks_after_delete = rag.query_documents(query_1, chat_id=chat_id_1, n_results=1)
    print(f"Relevant chunks for {chat_id_1} after deletion: {relevant_chunks_after_delete}")
    assert len(relevant_chunks_after_delete) == 0
    # Verify deletion of metadata
    loaded_chat_1_after_delete = rag.load_chat_metadata(chat_id_1)
    print(f"Loaded Chat 1 after deletion: {loaded_chat_1_after_delete}")
    assert loaded_chat_1_after_delete is None

    print(f"\n--- Testing Querying for chat_id: {chat_id_2} (should still work) ---")
    relevant_chunks_2_after_delete = rag.query_documents(query_2, chat_id=chat_id_2, n_results=1)
    print("Relevant chunks found:")
    for chunk in relevant_chunks_2_after_delete:
        print(f"- {chunk}")

    rag.clear_all_data() # Clean up at the end of all tests