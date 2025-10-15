import os
import sys
import json
from dotenv import load_dotenv
from multi_doc_chat.utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from multi_doc_chat.logger.custom_logger import CustomLogger as log
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException

class ApiKeyManager:
    REQUIRED_KEYS = ["GOOGLE_GENAI_API_KEY", "GROQ_API_KEY"]
    
    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("API_KEYS")
        
        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS must be a JSON object")
                self.api_keys = parsed
                log.info("API keys loaded from environment variable.")
            except json.JSONDecodeError as e:
                log.warning(f"Failed to parse API_KEYS: {e}")
        
        
        for key in self.REQUIRED_KEYS:
            if key not in self.api_keys:
                env_value = os.getenv(key)
                if env_value:
                    self.api_keys[key] = env_value
                    log.info(f"API key for {key} loaded from environment variable.")
                    
        missing_keys = [key for key in self.REQUIRED_KEYS if key not in self.api_keys]
        if missing_keys:
            log.error(f"Missing required API keys: {', '.join(missing_keys)}")
            
    def get_key(self, key_name: str) -> str:
        if key_name in self.api_keys:
            return self.api_keys[key_name]
        raise KeyError(f"API key '{key_name}' not found")
    
class ModelLoader:
    def __init__(self):
        if os.getenv("ENV", "development").lower() != "production":
            load_dotenv()
            log.info("Loaded environment variables from .env file in development mode.")
        else:
            log.info("Running in production mode")
            
        self.api_manager = ApiKeyManager()
        self.config = load_config()
        log.info(f"YAML configuration loaded successfully.", config=list(self.config.keys()))
        
    def get_embedding_model(self):
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info(f"Initializing embedding model: {model_name}")
            # Try primary (Google) embeddings first
            try:
                return GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    api_key=self.api_manager.get_key("GOOGLE_GENAI_API_KEY")
                )
            except Exception as primary_err:
                # Primary provider failed — attempt a local SentenceTransformer fallback.
                log.warning(f"Primary embedding provider failed: {primary_err}. Attempting SentenceTransformer fallback.")

                try:
                    # Import lazily to avoid heavy startup cost when not needed
                    from sentence_transformers import SentenceTransformer
                except Exception as imp_err:
                    log.error(f"SentenceTransformer not available: {imp_err}. Cannot fall back to local embeddings.")
                    raise

                class _SentenceTransformerEmbeddings:
                    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
                        self.model = SentenceTransformer(model_name)

                    def embed_documents(self, texts):
                        # Expecting a list of strings -> list of vectors
                        return [list(v) for v in self.model.encode(list(texts), show_progress_bar=False)]

                    def embed_query(self, text: str):
                        # Return single vector for query
                        vec = self.model.encode([text], show_progress_bar=False)[0]
                        return list(vec)

                    # LangChain/FAISS sometimes expects the embeddings object to be callable
                    # as a function that takes a single text and returns its vector. Provide
                    # __call__ to be compatible with that usage.
                    def __call__(self, text: str):
                        return self.embed_query(text)

                # Instantiate adapter with a compact, fast model
                adapter = _SentenceTransformerEmbeddings()
                log.info("Using SentenceTransformer fallback for embeddings (all-MiniLM-L6-v2).")
                return adapter
            
        except Exception as e:
            log.error(f"Failed to initialize embedding model: {e}")
            raise DocumentPortalException("Failed to initialize embedding model", sys)

    # Backwards-compatible alias used by other modules
    def load_embeddings(self):
        """Alias for get_embedding_model() retained for backward compatibility."""
        return self.get_embedding_model()
        
    def load_llm_model(self):
        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER", "google")
        
        if provider_key not in llm_block:
            log.error(f"LLM provider '{provider_key}' not found in configuration.")
            raise DocumentPortalException(f"LLM provider '{provider_key}' not found in configuration.", sys)
        
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_tokens", 1024)
        
        log.info(f"Loading LLM model: Provider={provider}, Model={model_name}")
        
        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
                api_key=self.api_manager.get_key("GOOGLE_GENAI_API_KEY")
            )
            
        elif provider == "groq":
            return ChatGroq(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=self.api_manager.get_key("GROQ_API_KEY")
            )
            
        else:
            log.error(f"Unsupported LLM provider: {provider}")
            raise DocumentPortalException(f"Unsupported LLM provider: {provider}", sys)

    # Backwards-compatible alias used by other modules
    def load_llm(self):
        """Alias for load_llm_model() retained for backward compatibility."""
        return self.load_llm_model()
        
if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")