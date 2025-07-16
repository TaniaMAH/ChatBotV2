"""
Sistema RAG completo para currículums USC
Integra chunking inteligente + BGE-M3 + ChromaDB
"""

import sys
sys.path.append('.')

from src.embeddings.bge_embeddings import BGEEmbeddings
from src.rag.vector_store import LocalVectorStore
from src.rag.curriculum_processor import USCCurriculumProcessor
from typing import Dict, List, Any, Optional
import time
from datetime import datetime

class USCCurriculumRAG:
    def __init__(self, 
                 collection_name: str = "usc_curriculum_rag",
                 vectorstore_dir: str = "./data/vectorstore"):
        """
        Sistema RAG completo para currículums USC
        
        Args:
            collection_name: Nombre de la colección en ChromaDB
            vectorstore_dir: Directorio del vector store
        """
        
        print("🚀 INICIALIZANDO SISTEMA RAG USC")
        print("=" * 40)
        
        # Inicializar componentes
        print("1️⃣ Cargando BGE-M3...")
        self.embedder = BGEEmbeddings()
        
        print("\n2️⃣ Configurando Vector Store...")
        self.vectorstore = LocalVectorStore(
            collection_name=collection_name,
            persist_directory=vectorstore_dir
        )
        
        print("\n3️⃣ Inicializando Procesador de Currículums...")
        self.processor = USCCurriculumProcessor()
        
        # Estado del sistema
        self.is_loaded = False
        self.chunks_count = 0
        self.load_time = None
        
        print("\n✅ Sistema RAG inicializado correctamente")
    
    def load_curriculum_data(self, 
                           curriculum_file: str,
                           force_reload: bool = False) -> bool:
        """
        Carga datos de currículums al sistema RAG
        
        Args:
            curriculum_file: Ruta al archivo .md de currículums
            force_reload: Si True, recarga datos aunque ya existan
            
        Returns:
            True si la carga fue exitosa
        """
        
        start_time = time.time()
        
        print(f"\n📚 CARGANDO DATOS DE CURRÍCULUMS")
        print(f"   Archivo: {curriculum_file}")
        
        # Verificar si ya hay datos cargados
        current_count = self.vectorstore.collection.count()
        if current_count > 0 and not force_reload:
            print(f"   ℹ️  Ya hay {current_count} documentos cargados")
            user_input = input("   ¿Recargar datos? (y/N): ").lower()
            if user_input != 'y':
                self.is_loaded = True
                self.chunks_count = current_count
                return True
            else:
                print("   🗑️  Limpiando datos existentes...")
                self.vectorstore.clear_collection()
        
        try:
            # 1. Procesar archivo con chunking inteligente
            print("\n🧠 Aplicando chunking inteligente...")
            chunks = self.processor.process_curriculum_file(curriculum_file)
            
            if not chunks:
                print("❌ No se pudieron extraer chunks del archivo")
                return False
            
            # 2. Crear embeddings con BGE-M3
            print(f"\n🔄 Creando embeddings para {len(chunks)} chunks...")
            texts = [chunk['content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            embeddings = self.embedder.embed_documents(texts)
            
            # 3. Almacenar en vector store
            print(f"\n💾 Almacenando en vector store...")
            success = self.vectorstore.add_documents(texts, metadatas, embeddings)
            
            if success:
                # Actualizar estado
                self.is_loaded = True
                self.chunks_count = len(chunks)
                self.load_time = time.time() - start_time
                
                print(f"\n🎉 DATOS CARGADOS EXITOSAMENTE")
                print(f"   Chunks procesados: {self.chunks_count}")
                print(f"   Tiempo de carga: {self.load_time:.2f}s")
                
                # Mostrar estadísticas
                self._show_data_statistics()
                
                return True
            else:
                print("❌ Error almacenando datos en vector store")
                return False
                
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
    
    def search_curriculum(self, 
                         query: str, 
                         n_results: int = 5,
                         filter_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Busca información en currículums usando RAG
        
        Args:
            query: Consulta del usuario
            n_results: Número de resultados a retornar
            filter_type: Filtro por tipo de chunk (fee, profile, curriculum, etc.)
            
        Returns:
            Diccionario con resultados de búsqueda
        """
        
        if not self.is_loaded:
            return {
                'success': False,
                'error': 'Datos no cargados. Ejecuta load_curriculum_data() primero.'
            }
        
        print(f"\n🔍 BÚSQUEDA EN CURRÍCULUMS")
        print(f"   Query: '{query}'")
        if filter_type:
            print(f"   Filtro: {filter_type}")
        
        start_time = time.time()
        
        try:
            # 1. Crear embedding de consulta
            query_embedding = self.embedder.embed_query(query)
            
            # 2. Realizar búsqueda (con filtro opcional)
            search_params = {}
            if filter_type:
                search_params['where'] = {'type': filter_type}
            
            results = self.vectorstore.search(
                query_embedding, 
                n_results=n_results,
                **search_params
            )
            
            # 3. Procesar y enriquecer resultados
            processed_results = self._process_search_results(results, query)
            
            search_time = time.time() - start_time
            
            return {
                'success': True,
                'query': query,
                'results': processed_results,
                'total_found': len(processed_results),
                'search_time': search_time,
                'filter_applied': filter_type
            }
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def smart_search(self, query: str) -> Dict[str, Any]:
        """
        Búsqueda inteligente que detecta automáticamente el tipo de consulta
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Resultados de búsqueda optimizada
        """
        
        # Detectar tipo de consulta basado en palabras clave
        query_lower = query.lower()
        
        detected_type = None
        search_params = {'n_results': 5}
        
        # Detección de tipo de consulta
        if any(word in query_lower for word in ['costo', 'precio', 'fee', 'cuánto', 'valor']):
            detected_type = 'fee'
            search_params['filter_type'] = 'fee'
            search_params['n_results'] = 10  # Más resultados para comparar costos
            
        elif any(word in query_lower for word in ['trabajo', 'laboral', 'ocupacional', 'desempeñar', 'campo']):
            detected_type = 'occupational_profile'
            search_params['filter_type'] = 'occupational_profile'
            
        elif any(word in query_lower for word in ['materia', 'semestre', 'curriculum', 'plan', 'asignatura']):
            detected_type = 'curriculum'
            # No filtrar por tipo específico, pero dar preferencia a curriculum
            search_params['n_results'] = 8
            
        elif any(word in query_lower for word in ['primer', 'segundo', 'tercer', 'semestre 1', 'semestre 2']):
            detected_type = 'curriculum_semester'
            search_params['filter_type'] = 'curriculum_semester'
        
        # Realizar búsqueda
        results = self.search_curriculum(query, **search_params)
        
        # Agregar información de detección
        if results['success']:
            results['detected_type'] = detected_type
            results['search_strategy'] = 'smart_detection'
            
            print(f"   🎯 Tipo detectado: {detected_type or 'general'}")
            print(f"   📊 Resultados: {results['total_found']}")
        
        return results
    
    def compare_programs(self, 
                        programs: List[str], 
                        comparison_aspect: str = 'fee') -> Dict[str, Any]:
        """
        Compara múltiples programas en un aspecto específico
        
        Args:
            programs: Lista de nombres de programas
            comparison_aspect: Aspecto a comparar (fee, profile, curriculum)
            
        Returns:
            Comparación estructurada de programas
        """
        
        print(f"\n📊 COMPARANDO PROGRAMAS")
        print(f"   Programas: {', '.join(programs)}")
        print(f"   Aspecto: {comparison_aspect}")
        
        comparison_results = {}
        
        for program in programs:
            query = f"{program} {comparison_aspect}"
            results = self.search_curriculum(
                query, 
                n_results=3,
                filter_type=comparison_aspect if comparison_aspect != 'curriculum' else None
            )
            
            if results['success'] and results['results']:
                comparison_results[program] = results['results'][0]
            else:
                comparison_results[program] = None
        
        return {
            'comparison_aspect': comparison_aspect,
            'programs': comparison_results,
            'summary': self._create_comparison_summary(comparison_results, comparison_aspect)
        }
    
    def _process_search_results(self, raw_results: Dict[str, List], query: str) -> List[Dict[str, Any]]:
        """Procesa y enriquece resultados de búsqueda"""
        
        processed = []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            raw_results['documents'],
            raw_results['metadatas'], 
            raw_results['distances']
        )):
            
            # Calcular similitud (1 - distancia)
            similarity = 1 - distance
            
            result = {
                'rank': i + 1,
                'content': doc,
                'metadata': metadata,
                'similarity': similarity,
                'distance': distance,
                'program_name': metadata.get('program_name', 'Unknown'),
                'chunk_type': metadata.get('type', 'Unknown'),
                'relevance_score': similarity * 100  # Porcentaje
            }
            
            # Agregar información específica según tipo
            if metadata.get('type') == 'fee':
                result['fee_amount'] = metadata.get('fee_amount', 0)
                result['formatted_fee'] = f"${metadata.get('fee_amount', 0):,} COP"
            
            elif metadata.get('type') == 'curriculum_semester':
                result['semester'] = metadata.get('semester_number', 'N/A')
                result['credits'] = metadata.get('total_credits', 0)
                result['subjects'] = metadata.get('subject_count', 0)
            
            processed.append(result)
        
        # Ordenar por similitud descendente
        processed.sort(key=lambda x: x['similarity'], reverse=True)
        
        return processed
    
    def _create_comparison_summary(self, comparison_data: Dict[str, Any], aspect: str) -> str:
        """Crea resumen de comparación entre programas"""
        
        if aspect == 'fee':
            # Comparación de costos
            fees = {}
            for program, data in comparison_data.items():
                if data and 'fee_amount' in data['metadata']:
                    fees[program] = data['metadata']['fee_amount']
            
            if fees:
                sorted_fees = sorted(fees.items(), key=lambda x: x[1])
                cheapest = sorted_fees[0]
                most_expensive = sorted_fees[-1]
                
                summary = f"Programa más económico: {cheapest[0]} (${cheapest[1]:,} COP)\n"
                summary += f"Programa más costoso: {most_expensive[0]} (${most_expensive[1]:,} COP)"
                
                return summary
        
        return "Comparación completada"
    
    def _show_data_statistics(self):
        """Muestra estadísticas de los datos cargados"""
        
        stats = self.vectorstore.get_stats()
        
        print(f"\n📈 ESTADÍSTICAS DE DATOS CARGADOS")
        print(f"   Total documentos: {stats['total_documents']}")
        print(f"   Tipos de chunks: {stats['document_types']}")
        print(f"   Programas: {len(stats['programs'])} programas")
        
        # Mostrar algunos programas
        if stats['programs']:
            programs_sample = stats['programs'][:5]
            print(f"   Ejemplos: {', '.join(programs_sample)}")
            if len(stats['programs']) > 5:
                print(f"   ... y {len(stats['programs']) - 5} más")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene estado completo del sistema"""
        
        vectorstore_stats = self.vectorstore.get_stats()
        
        return {
            'system_loaded': self.is_loaded,
            'chunks_loaded': self.chunks_count,
            'load_time': self.load_time,
            'vectorstore_stats': vectorstore_stats,
            'embedder_info': self.embedder.get_model_info(),
            'processor_stats': self.processor.get_processing_stats()
        }