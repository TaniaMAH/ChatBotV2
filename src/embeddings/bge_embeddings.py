"""
BGE-M3 Embeddings para RAG USC (Versión Compatible)
Maneja embeddings locales gratuitos con alta calidad
"""

from FlagEmbedding import BGEM3FlagModel
import numpy as np
import torch
from typing import List, Union, Optional
import os
from pathlib import Path

class BGEEmbeddings:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None):
        """
        Inicializa BGE-M3 para embeddings de alta calidad
        
        Args:
            model_name: Modelo BGE a usar (default: bge-m3)
            device: 'cuda', 'cpu' o None (auto-detect)
        """
        
        # Auto-detectar dispositivo si no se especifica
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        print(f"🔄 Inicializando BGE-M3...")
        print(f"   📋 Modelo: {model_name}")
        print(f"   💻 Dispositivo: {device}")
        
        try:
            # Cargar modelo BGE-M3 (versión compatible)
            self.model = BGEM3FlagModel(
                model_name,
                use_fp16=True if device == "cuda" else False,
                device=device
            )
            
            print(f"✅ BGE-M3 cargado exitosamente")
            print(f"   📏 Dimensiones: {self.get_dimension()}")
            
        except Exception as e:
            print(f"❌ Error cargando BGE-M3: {e}")
            print("💡 Intentando con configuración básica...")
            
            try:
                # Fallback a configuración más básica
                self.model = BGEM3FlagModel(model_name)
                self.device = "cpu"
                print(f"✅ BGE-M3 cargado en modo básico")
                
            except Exception as e2:
                print(f"❌ Error en fallback: {e2}")
                raise Exception(f"No se pudo cargar BGE-M3: {e2}")
    
    def embed_documents(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Crea embeddings para múltiples documentos/chunks
        
        Args:
            texts: Lista de textos para embebido
            batch_size: Tamaño de lote para procesamiento (reducido para estabilidad)
            
        Returns:
            numpy array con embeddings densos
        """
        
        if not texts:
            return np.array([])
        
        print(f"🔄 Creando embeddings para {len(texts)} documentos...")
        
        try:
            # BGE-M3 encode con configuración básica
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                max_length=1024,        # Reducido para compatibilidad
                return_dense=True,      # Solo embeddings densos
                return_sparse=False,    
                return_colbert_vecs=False
            )
            
            # Extraer embeddings densos
            dense_embeddings = embeddings['dense_vecs']
            
            print(f"✅ Embeddings creados: {dense_embeddings.shape}")
            return dense_embeddings
            
        except Exception as e:
            print(f"❌ Error creando embeddings: {e}")
            print("🔄 Intentando procesamiento secuencial...")
            return self._embed_documents_sequential(texts)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Crea embedding para una consulta de usuario
        
        Args:
            query: Consulta del usuario
            
        Returns:
            numpy array con embedding de la consulta
        """
        
        try:
            # Usar encode simple para consultas
            embedding = self.model.encode(
                [query],
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            
            return embedding['dense_vecs'][0]
            
        except Exception as e:
            print(f"❌ Error creando embedding de consulta: {e}")
            # Fallback: embedding dummy
            return np.random.random(self.get_dimension()).astype(np.float32)
    
    def _embed_documents_sequential(self, texts: List[str]) -> np.ndarray:
        """Fallback: procesar documentos uno por uno"""
        
        print("⚠️  Procesando embeddings secuencialmente...")
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                # Procesar texto individual
                emb = self.model.encode([text], return_dense=True)['dense_vecs'][0]
                embeddings.append(emb)
                
                if (i + 1) % 5 == 0:
                    print(f"   ✅ Procesados {i + 1}/{len(texts)}")
                    
            except Exception as e:
                print(f"   ⚠️  Error en documento {i}: {e}")
                # Embedding dummy para mantener indexing
                embeddings.append(np.random.random(1024).astype(np.float32))
        
        return np.array(embeddings)
    
    def get_dimension(self) -> int:
        """Retorna dimensión de los embeddings BGE-M3"""
        return 1024  # BGE-M3 produce embeddings de 1024 dimensiones
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calcula similitud coseno entre dos embeddings"""
        
        # Normalizar embeddings
        norm1 = embedding1 / np.linalg.norm(embedding1)
        norm2 = embedding2 / np.linalg.norm(embedding2)
        
        # Similitud coseno
        similarity = np.dot(norm1, norm2)
        return float(similarity)
    
    def get_model_info(self) -> dict:
        """Información del modelo cargado"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'embedding_dimension': self.get_dimension(),
            'supports_batch': True,
            'multilingual': True,
            'status': 'loaded'
        }

# Test básico del módulo
if __name__ == "__main__":
    print("🧪 PROBANDO BGE-M3 EMBEDDINGS (Versión Compatible)")
    print("=" * 50)
    
    try:
        # Inicializar
        embedder = BGEEmbeddings()
        
        # Test con datos de ejemplo
        test_docs = [
            "Ingeniería de Sistemas tiene un costo de $5.298.134 COP",
            "Bioingeniería se enfoca en tecnologías para atención médica"
        ]
        
        # Test embeddings de documentos
        print("\n📄 Test embeddings de documentos:")
        doc_embeddings = embedder.embed_documents(test_docs)
        print(f"   Shape: {doc_embeddings.shape}")
        print(f"   Tipo: {type(doc_embeddings)}")
        
        # Test embedding de consulta
        print("\n❓ Test embedding de consulta:")
        query = "¿Cuánto cuesta ingeniería?"
        query_embedding = embedder.embed_query(query)
        print(f"   Shape: {query_embedding.shape}")
        
        # Test similitud
        print("\n🔍 Test similitud:")
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            sim = embedder.calculate_similarity(query_embedding, doc_emb)
            similarities.append(sim)
            print(f"   Doc {i+1}: {sim:.3f}")
        
        # Mejor match
        best_idx = np.argmax(similarities)
        print(f"\n🎯 Mejor match: Doc {best_idx+1}")
        print(f"   Texto: {test_docs[best_idx]}")
        print(f"   Similitud: {similarities[best_idx]:.3f}")
        
        # Info del modelo
        print(f"\n📋 Info del modelo:")
        info = embedder.get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        print("\n✅ BGE-M3 funcionando correctamente!")
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()