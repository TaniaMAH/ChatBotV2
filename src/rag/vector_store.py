"""
Vector Store local con ChromaDB para USC Curriculum RAG
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from typing import List, Dict, Optional, Any
import json
import uuid
from pathlib import Path

class LocalVectorStore:
    def __init__(self, 
                 collection_name: str = "usc_curriculum", 
                 persist_directory: str = "./data/vectorstore"):
        """
        Inicializa ChromaDB local para almacenar embeddings
        """
        
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        
        print(f"🔄 Inicializando ChromaDB...")
        print(f"   📂 Directorio: {persist_directory}")
        print(f"   📝 Colección: {collection_name}")
        
        # Crear directorio si no existe
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Configurar ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Obtener o crear colección (CORREGIDO)
        try:
            # Intentar obtener colección existente
            self.collection = self.client.get_collection(collection_name)
            existing_count = self.collection.count()
            print(f"✅ Colección existente cargada")
            print(f"   📊 Documentos existentes: {existing_count}")
            
        except Exception as e:
            # Si no existe, crear nueva colección
            print(f"📝 Colección no existe, creando nueva...")
            try:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "USC Curriculum RAG Collection"}
                )
                print(f"✅ Nueva colección creada: {collection_name}")
                
            except Exception as create_error:
                print(f"❌ Error creando colección: {create_error}")
                # Intentar eliminar y recrear
                try:
                    self.client.delete_collection(collection_name)
                    print(f"🗑️  Colección anterior eliminada")
                except:
                    pass
                
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "USC Curriculum RAG Collection"}
                )
                print(f"✅ Colección recreada: {collection_name}")
        
        self.stats = {
            'documents_added': 0,
            'queries_processed': 0,
            'last_update': None
        }
    
    def add_documents(self, 
                     texts: List[str], 
                     metadatas: List[Dict[str, Any]], 
                     embeddings: np.ndarray,
                     ids: Optional[List[str]] = None) -> bool:
        """Agrega documentos al vector store"""
        
        if len(texts) != len(metadatas) or len(texts) != len(embeddings):
            raise ValueError("❌ Longitudes de texts, metadatas y embeddings deben coincidir")
        
        print(f"🔄 Agregando {len(texts)} documentos al vector store...")
        
        try:
            # Generar IDs únicos si no se proporcionan
            if ids is None:
                ids = [f"doc_{uuid.uuid4().hex[:8]}" for _ in texts]
            
            # Convertir embeddings a lista de listas
            embeddings_list = embeddings.tolist()
            
            # Limpiar metadatos para ChromaDB
            cleaned_metadatas = []
            for metadata in metadatas:
                cleaned_metadata = {}
                for key, value in metadata.items():
                    # ChromaDB solo acepta str, int, float, bool
                    if isinstance(value, (str, int, float, bool)):
                        cleaned_metadata[key] = value
                    else:
                        cleaned_metadata[key] = str(value)
                cleaned_metadatas.append(cleaned_metadata)
            
            # Agregar a ChromaDB
            self.collection.add(
                documents=texts,
                metadatas=cleaned_metadatas,
                embeddings=embeddings_list,
                ids=ids
            )
            
            self.stats['documents_added'] += len(texts)
            print(f"✅ Documentos agregados exitosamente")
            print(f"   📊 Total en colección: {self.collection.count()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error agregando documentos: {e}")
            return False
    
    def search(self, 
              query_embedding: np.ndarray, 
              n_results: int = 5,
              where: Optional[Dict] = None) -> Dict[str, List]:
        """Busca documentos similares"""
        
        if self.collection.count() == 0:
            print("⚠️  Vector store vacío")
            return {
                'documents': [],
                'metadatas': [],
                'distances': [],
                'ids': []
            }
        
        try:
            # Convertir embedding a lista
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Realizar búsqueda
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            self.stats['queries_processed'] += 1
            
            # Formatear resultados
            formatted_results = {
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if results['distances'] else [],
                'ids': results['ids'][0] if results['ids'] else []
            }
            
            print(f"🔍 Búsqueda completada: {len(formatted_results['documents'])} resultados")
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return {
                'documents': [],
                'metadatas': [],
                'distances': [],
                'ids': []
            }
    
    def clear_collection(self) -> bool:
        """Limpia todos los documentos de la colección"""
        
        try:
            # Eliminar colección existente
            self.client.delete_collection(self.collection_name)
            
            # Crear nueva colección vacía
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "USC Curriculum RAG Collection"}
            )
            
            print(f"🗑️  Colección {self.collection_name} limpiada")
            return True
            
        except Exception as e:
            print(f"❌ Error limpiando colección: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del vector store"""
        
        try:
            total_docs = self.collection.count()
            
            # Obtener algunos documentos para extraer metadatos
            doc_types = []
            programs = []
            
            if total_docs > 0:
                # Obtener una muestra de documentos
                sample_docs = self.collection.get(
                    limit=min(100, total_docs),
                    include=["metadatas"]
                )
                
                for metadata in sample_docs['metadatas']:
                    if 'type' in metadata:
                        doc_types.append(metadata['type'])
                    if 'program' in metadata:
                        programs.append(metadata['program'])
                
                # Obtener valores únicos
                doc_types = list(set(doc_types))
                programs = list(set(programs))
            
            stats = {
                'total_documents': total_docs,
                'collection_name': self.collection_name,
                'persist_directory': str(self.persist_directory),
                'document_types': doc_types,
                'programs': programs,
                'queries_processed': self.stats['queries_processed'],
                'documents_added': self.stats['documents_added']
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error obteniendo estadísticas: {e}")
            return {'total_documents': 0, 'error': str(e)}

# Test básico del vector store
if __name__ == "__main__":
    print("🧪 PROBANDO VECTOR STORE LOCAL (Versión Corregida)")
    print("=" * 50)
    
    # Crear embeddings dummy para test
    test_texts = [
        "Ingeniería de Sistemas cuesta $5.298.134 COP",
        "Bioingeniería tiene perfil en tecnologías médicas"
    ]
    
    test_metadatas = [
        {"type": "fee", "program": "Systems Engineering", "amount": 5298134},
        {"type": "profile", "program": "Bioengineering"}
    ]
    
    # Embeddings dummy
    test_embeddings = np.random.random((len(test_texts), 1024)).astype(np.float32)
    
    # Test vector store
    vs = LocalVectorStore(collection_name="test_collection_fixed")
    
    # Limpiar para test limpio
    vs.clear_collection()
    
    # Agregar documentos
    success = vs.add_documents(test_texts, test_metadatas, test_embeddings)
    print(f"📝 Documentos agregados: {'✅' if success else '❌'}")
    
    # Test búsqueda
    query_embedding = np.random.random(1024).astype(np.float32)
    results = vs.search(query_embedding, n_results=2)
    
    print(f"\n🔍 Resultados de búsqueda:")
    print(f"   Documentos encontrados: {len(results['documents'])}")
    
    # Estadísticas
    print(f"\n📊 Estadísticas:")
    stats = vs.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Vector store funcionando correctamente!")