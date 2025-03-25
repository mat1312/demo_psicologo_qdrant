async def delete_collection_if_exists(client):
    """Elimina la collezione se esiste già."""
    try:
        collections = await client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if COLLECTION_NAME in collection_names:
            logger.info(f"Eliminazione della collezione esistente '{COLLECTION_NAME}'...")
            await client.delete_collection(collection_name=COLLECTION_NAME)
            logger.info(f"Collezione '{COLLECTION_NAME}' eliminata con successo")
    except Exception as e:
        logger.error(f"Errore nell'eliminazione della collezione: {e}")
        raise"""
Script per indicizzare i documenti della Croce Rossa Italiana nel vector database Qdrant Cloud.
Questo script processa tutti i file dalla cartella 'output' e li indicizza nel Qdrant Cloud.
Versione ottimizzata con operazioni asincrone.

Requisiti:
    pip install langchain langchain-openai langchain-qdrant python-dotenv
"""

import os
import glob
import argparse
import asyncio
import random
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    DirectoryLoader,
    UnstructuredMarkdownLoader
)
# Utilizza il pacchetto più recente dedicato a Qdrant
from langchain_community.vectorstores import Qdrant  # Torniamo alla versione stabile
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import logging
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import Distance, VectorParams

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carica variabili d'ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")  # URL del cluster Qdrant Cloud
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # API key di Qdrant Cloud
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")  # Nome della collezione

# Configurazione percorsi
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # Directory di output
DATA_DIR = os.path.join(BASE_DIR, "data") # Directory di dati

# Assicurati che le directory esistano
os.makedirs(DATA_DIR, exist_ok=True)

# Configurazione chunking
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
# Configurazione per operazioni parallele
MAX_WORKERS = 16  # Parallelismo per operazioni generali
BATCH_SIZE = 800  # Ridotto per evitare rate limit
MAX_CONCURRENT_API_CALLS = 3  # Ridotto per rispettare i rate limit di OpenAI
RETRY_BASE_DELAY = 3  # Ritardo base per i retry in secondi
MAX_RETRIES = 10  # Numero massimo di tentativi prima di fallire


async def load_file(loader_class, file_path, **kwargs):
    """Carica un singolo file usando il loader specificato in modo asincrono."""
    try:
        # Usa ThreadPoolExecutor per operazioni di I/O bloccanti
        with ThreadPoolExecutor() as executor:
            loader = loader_class(file_path, **kwargs)
            docs = await asyncio.get_event_loop().run_in_executor(
                executor, loader.load
            )
            logger.info(f"Caricato: {file_path}")
            return docs
    except Exception as e:
        logger.error(f"Errore nel caricamento di {file_path}: {e}")
        return []


async def load_documents(input_dir: str) -> List[Document]:
    """Carica tutti i file .md dalla directory specificata."""
    if not os.path.exists(input_dir):
        logger.error(f"La directory {input_dir} non esiste.")
        return []
    
    tasks = []
    
    # Carica solo file markdown
    md_pattern = os.path.join(input_dir, "**/*.md")
    md_files = glob.glob(md_pattern, recursive=True)
    
    # Accodamento di tutti i file MD
    for md_path in md_files:
        logger.info(f"Accodamento MD: {md_path}")
        tasks.append(load_file(UnstructuredMarkdownLoader, md_path))
    
    # Attendi il completamento di tutti i task
    results = await asyncio.gather(*tasks)
    
    # Appiattisci i risultati
    documents = []
    for docs in results:
        documents.extend(docs)
    
    logger.info(f"Caricati {len(documents)} documenti in totale da {len(md_files)} file MD")
    return documents


async def split_documents_async(documents: List[Document]) -> List[Document]:
    """Divide i documenti in chunks più piccoli in modo asincrono."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Usa ThreadPoolExecutor poiché RecursiveCharacterTextSplitter non è async-native
    with ThreadPoolExecutor() as executor:
        chunks = await asyncio.get_event_loop().run_in_executor(
            executor, text_splitter.split_documents, documents
        )
    
    logger.info(f"Documenti suddivisi in {len(chunks)} chunks")
    return chunks


async def create_qdrant_collection(client, embedding_size=1536):
    """Crea una collezione su Qdrant se non esiste già."""
    try:
        # Verifica se la collezione esiste
        collections = await client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            logger.info(f"Creazione nuova collezione '{COLLECTION_NAME}'...")
            # Usiamo la dimensione passata come parametro
            logger.info(f"Dimensione impostata per il vettore di embedding: {embedding_size}")
            
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=embedding_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Collezione '{COLLECTION_NAME}' creata con successo con dimensione {embedding_size}")
        else:
            logger.info(f"La collezione '{COLLECTION_NAME}' esiste già")
    except Exception as e:
        logger.error(f"Errore nella creazione/verifica della collezione: {e}")
        raise


async def process_batch_embedding(batch, embeddings, semaphore, batch_idx, total_batches):
    """Processa un singolo batch di embedding con controllo della concorrenza e gestione avanzata degli errori."""
    async with semaphore:
        logger.info(f"Elaborando batch {batch_idx+1}/{total_batches} di {len(batch)} chunks")
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        
        # Implementazione robusta con backoff esponenziale
        retries = 0
        while retries < MAX_RETRIES:
            try:
                with ThreadPoolExecutor() as executor:
                    # Esegui l'embedding in modo asincrono
                    embedded_texts = await asyncio.get_event_loop().run_in_executor(
                        executor, 
                        lambda: embeddings.embed_documents(texts)
                    )
                
                logger.info(f"Batch {batch_idx+1}/{total_batches} completato con successo")
                return embedded_texts, texts, metadatas
                
            except Exception as e:
                retries += 1
                # Calcola backoff esponenziale con jitter
                delay = RETRY_BASE_DELAY * (2 ** retries) + (random.random() * 2)
                if "rate_limit" in str(e).lower():
                    logger.warning(f"Rate limit raggiunto per il batch {batch_idx+1}. Attesa di {delay:.2f}s. Tentativo {retries}/{MAX_RETRIES}")
                else:
                    logger.warning(f"Errore per il batch {batch_idx+1}: {str(e)}. Attesa di {delay:.2f}s. Tentativo {retries}/{MAX_RETRIES}")
                
                await asyncio.sleep(delay)
        
        # Se arriviamo qui, tutti i tentativi sono falliti
        error_msg = f"Tutti i {MAX_RETRIES} tentativi falliti per il batch {batch_idx+1}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


async def create_vector_store(chunks: List[Document]):
    """Crea il vector store utilizzando Qdrant in modo asincrono con gestione robusta dei rate limit."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY non trovata. Imposta la variabile d'ambiente.")
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL o QDRANT_API_KEY non trovati. Imposta le variabili d'ambiente.")
    
    start_time = time.time()
    
    # Importa random per il jitter nel backoff
    import random
    random.seed()
    
    # Usa OpenAI per gli embeddings con configurazione ottimizzata per rate limit
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # Usiamo il modello small per stabilità
        chunk_size=BATCH_SIZE,
        request_timeout=120,  # Timeout più lungo per gestire backoff
        max_retries=MAX_RETRIES
    )
    
    # Test per verificare la dimensione dell'embedding
    test_text = "Test embedding dimensionality"
    test_embedding = embeddings.embed_query(test_text)
    embed_dim = len(test_embedding)
    logger.info(f"Dimensione dell'embedding di test: {embed_dim}")
    
    # Creazione client Qdrant asincrono
    qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Elimina la collezione se esiste già
    await delete_collection_if_exists(qdrant_client)
    
    # Creazione della collezione con la dimensione corretta
    await create_qdrant_collection(qdrant_client, embedding_size=embed_dim)
    
    # Prepara chunks per l'inserimento nel database
    logger.info(f"Inizializzazione upserting in Qdrant con {len(chunks)} chunks totali...")
    
    # Crea il vector store usando la classe stabile
    sync_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vector_store = Qdrant(
        client=sync_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
        content_payload_key="page_content",
        metadata_payload_key="metadata"
    )
    
    # Anziché pre-calcolare tutti gli embedding, aggiungiamo direttamente i documenti a Qdrant
    # in batch più piccoli. Questo approccio è più lento ma più robusto contro i rate limit
    batch_size = BATCH_SIZE
    total_batches = (len(chunks) - 1) // batch_size + 1
    logger.info(f"Elaborazione in {total_batches} batch di dimensione {batch_size}")
    
    # Crea elenco di batch di documenti
    doc_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    total_processed = 0
    
    # Creiamo un singolo executor per tutta la funzione per evitare problemi di shutdown
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Elaborazione sequenziale con backoff adattivo
        for batch_idx, doc_batch in enumerate(doc_batches):
            logger.info(f"Elaborazione batch {batch_idx + 1}/{len(doc_batches)} ({len(doc_batch)} chunks)")
            
            # Implementazione robusta con backoff esponenziale
            retries = 0
            success = False
            
            while not success and retries < MAX_RETRIES:
                try:
                    # Aggiungi documenti al vector store
                    await asyncio.get_event_loop().run_in_executor(
                        executor, 
                        lambda: vector_store.add_documents(doc_batch)
                    )
                    
                    total_processed += len(doc_batch)
                    logger.info(f"Batch {batch_idx+1}/{len(doc_batches)} completato con successo")
                    success = True
                    
                    # Log progressione
                    if (batch_idx + 1) % 5 == 0 or batch_idx == len(doc_batches) - 1:
                        logger.info(f"Progressione: {total_processed}/{len(chunks)} chunks elaborati ({total_processed/len(chunks)*100:.1f}%)")
                    
                except Exception as e:
                    retries += 1
                    # Calcola backoff esponenziale con jitter
                    delay = RETRY_BASE_DELAY * (2 ** retries) + (random.random() * 2)
                    
                    if "rate_limit" in str(e).lower():
                        logger.warning(f"Rate limit raggiunto per il batch {batch_idx+1}. Attesa di {delay:.2f}s. Tentativo {retries}/{MAX_RETRIES}")
                    else:
                        logger.warning(f"Errore per il batch {batch_idx+1}: {str(e)}. Attesa di {delay:.2f}s. Tentativo {retries}/{MAX_RETRIES}")
                    
                    await asyncio.sleep(delay)
            
            # Se il batch ha fallito tutti i tentativi, logghiamo l'errore ma continuiamo con il prossimo batch
            if not success:
                logger.error(f"Tutti i {MAX_RETRIES} tentativi falliti per il batch {batch_idx+1}. Proseguo con il prossimo batch.")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Vector store creato in Qdrant Cloud in {elapsed_time:.2f} secondi")
    logger.info(f"Totale chunks elaborati con successo: {total_processed}/{len(chunks)}")
    
    # Chiudi client
    await qdrant_client.close()
    
    return vector_store


async def main_async():
    global COLLECTION_NAME
    
    # Configurazione parser per argomenti da riga di comando
    parser = argparse.ArgumentParser(description='Indicizzazione documenti per l\'assistente CRI su Qdrant Cloud')
    parser.add_argument('--input-dir', type=str, default=OUTPUT_DIR,
                        help=f'Directory contenente i documenti da indicizzare (default: {OUTPUT_DIR})')
    parser.add_argument('--collection-name', type=str, default=COLLECTION_NAME,
                        help=f'Nome della collezione Qdrant (default: {COLLECTION_NAME})')
    args = parser.parse_args()
    
    input_dir = args.input_dir
    COLLECTION_NAME = args.collection_name
    
    logger.info(f"Inizializzazione processo di indicizzazione documenti da {input_dir} su Qdrant Cloud")
    logger.info(f"Collezione target: {COLLECTION_NAME}")
    
    start_time = time.time()
    
    try:
        # Carica i documenti in modo asincrono
        documents = await load_documents(input_dir)
        if not documents:
            logger.warning(f"Nessun documento trovato nella directory {input_dir}")
            return
        
        # Dividi i documenti in chunks in modo asincrono
        chunks = await split_documents_async(documents)
        
        # Crea e salva il vector store in modo asincrono
        await create_vector_store(chunks)
        
        total_time = time.time() - start_time
        logger.info(f"Processo di indicizzazione completato con successo in {total_time:.2f} secondi")
    
    except Exception as e:
        logger.error(f"ERRORE FATALE: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info(f"Processo di indicizzazione FALLITO dopo {time.time() - start_time:.2f} secondi")
        
        # In caso di errore fatale, rialza l'eccezione per far terminare il programma con errore
        raise


def main():
    """Funzione main che avvia il loop di eventi asincrono."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()