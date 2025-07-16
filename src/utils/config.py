"""Configuraciones globales del proyecto"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documentos"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
LOGS_DIR = PROJECT_ROOT / "logs"

# Crear directorios si no existen
for dir_path in [DATA_DIR, DOCUMENTS_DIR, VECTORSTORE_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Configuración de BGE-M3
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")
BGE_DEVICE = os.getenv("BGE_DEVICE", "auto")  # auto, cpu, cuda
BGE_BATCH_SIZE = int(os.getenv("BGE_BATCH_SIZE", "8"))

# Configuración de Vector Store
VECTORSTORE_COLLECTION_NAME = os.getenv("VECTORSTORE_COLLECTION", "usc_curriculum")
VECTORSTORE_PERSIST_DIR = str(VECTORSTORE_DIR)

# Configuración de chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Configuración de búsqueda
DEFAULT_SEARCH_RESULTS = int(os.getenv("DEFAULT_SEARCH_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

print(f"📁 Proyecto configurado en: {PROJECT_ROOT}")
print(f"📂 Documentos en: {DOCUMENTS_DIR}")
print(f"🗄️ Vector store en: {VECTORSTORE_DIR}")