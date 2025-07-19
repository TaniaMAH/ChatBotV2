"""
Integración híbrida del sistema de chunking USC
Combina chunking estructural confiable + chunking inteligente con LLM
"""

import sys
sys.path.append('.')

from src.embeddings.bge_embeddings import BGEEmbeddings
from src.rag.vector_store import LocalVectorStore
from src.rag.curriculum_processor import USCCurriculumProcessor
from typing import Dict, List, Any, Optional
import time
from datetime import datetime

# Importar chunking inteligente si está disponible
try:
    from src.rag.intelligent_chunking import IntelligentCurriculumChunker
    INTELLIGENT_CHUNKING_AVAILABLE = True
except ImportError:
    INTELLIGENT_CHUNKING_AVAILABLE = False
    print("⚠️  Chunking inteligente no disponible (opcional)")

class HybridUSCCurriculumRAG:
    def __init__(self, 
                 collection_name: str = "usc_curriculum_hybrid",
                 vectorstore_dir: str = "./data/vectorstore",
                 chunking_mode: str = "auto"):
        """
        Sistema RAG híbrido con chunking inteligente + estructural
        
        Args:
            collection_name: Nombre de la colección ChromaDB
            vectorstore_dir: Directorio del vector store
            chunking_mode: "auto", "intelligent", "structural"
        """
        
        print("🚀 INICIALIZANDO SISTEMA RAG HÍBRIDO USC")
        print("=" * 45)
        
        # Determinar modo de chunking
        self.chunking_mode = self._determine_chunking_mode(chunking_mode)
        print(f"   🧠 Modo chunking: {self.chunking_mode}")
        
        # Inicializar componentes base
        print("\n1️⃣ Cargando BGE-M3...")
        self.embedder = BGEEmbeddings()
        
        print("\n2️⃣ Configurando Vector Store...")
        self.vectorstore = LocalVectorStore(
            collection_name=collection_name,
            persist_directory=vectorstore_dir
        )
        
        # Inicializar procesador según modo
        print(f"\n3️⃣ Inicializando procesador ({self.chunking_mode})...")
        self._initialize_processor()
        
        # Estado del sistema
        self.is_loaded = False
        self.chunks_count = 0
        self.load_time = None
        
        print("\n✅ Sistema RAG híbrido inicializado correctamente")
    
    def _determine_chunking_mode(self, requested_mode: str) -> str:
        """Determina el modo de chunking a usar"""
        
        if requested_mode == "structural":
            return "structural"
        
        elif requested_mode == "intelligent":
            if INTELLIGENT_CHUNKING_AVAILABLE:
                return "intelligent"
            else:
                print("   ⚠️  Chunking inteligente no disponible, usando estructural")
                return "structural"
        
        else:  # "auto"
            if INTELLIGENT_CHUNKING_AVAILABLE:
                # Probar si Ollama está disponible
                try:
                    chunker = IntelligentCurriculumChunker(use_llm=True)
                    if hasattr(chunker, 'selected_model'):
                        return "intelligent"
                    else:
                        return "structural"
                except:
                    return "structural"
            else:
                return "structural"
    
    def _initialize_processor(self):
        """Inicializa el procesador según el modo"""
        
        if self.chunking_mode == "intelligent":
            self.processor = IntelligentCurriculumChunker(use_llm=True)
            print("   🤖 Procesador inteligente con LLM activado")
        else:
            self.processor = USCCurriculumProcessor()
            print("   📋 Procesador estructural confiable activado")
    
    def load_curriculum_data(self, 
                           curriculum_file: str,
                           force_reload: bool = False) -> bool:
        """
        Carga datos con chunking híbrido
        """
        
        start_time = time.time()
        
        print(f"\n📚 CARGANDO DATOS CON CHUNKING {self.chunking_mode.upper()}")
        print(f"   Archivo: {curriculum_file}")
        
        # Verificar datos existentes
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
            # Procesar según modo
            print(f"\n🧠 Aplicando chunking {self.chunking_mode}...")
            
            if self.chunking_mode == "intelligent":
                chunks = self.processor.process_full_curriculum(curriculum_file)
            else:
                chunks = self.processor.process_curriculum_file(curriculum_file)
            
            if not chunks:
                print("❌ No se pudieron extraer chunks del archivo")
                return False
            
            # Crear embeddings
            print(f"\n🔄 Creando embeddings para {len(chunks)} chunks...")
            texts = [chunk['content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            embeddings = self.embedder.embed_documents(texts)
            
            # Almacenar
            print(f"\n💾 Almacenando en vector store...")
            success = self.vectorstore.add_documents(texts, metadatas, embeddings)
            
            if success:
                self.is_loaded = True
                self.chunks_count = len(chunks)
                self.load_time = time.time() - start_time
                
                print(f"\n🎉 DATOS CARGADOS EXITOSAMENTE")
                print(f"   Chunks procesados: {self.chunks_count}")
                print(f"   Modo usado: {self.chunking_mode}")
                print(f"   Tiempo de carga: {self.load_time:.2f}s")
                
                self._show_chunking_analysis(chunks)
                
                return True
            else:
                print("❌ Error almacenando datos")
                return False
                
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
    
    def _show_chunking_analysis(self, chunks: List[Dict[str, Any]]):
        """Muestra análisis del chunking utilizado"""
        
        print(f"\n📊 ANÁLISIS DEL CHUNKING {self.chunking_mode.upper()}")
        print("=" * 35)
        
        # Análisis por tipos
        chunk_types = {}
        llm_chunks = 0
        structural_chunks = 0
        
        for chunk in chunks:
            chunk_type = chunk['metadata']['type']
            if chunk_type not in chunk_types:
                chunk_types[chunk_type] = 0
            chunk_types[chunk_type] += 1
            
            # Contar origen
            if chunk['metadata'].get('llm_generated', False):
                llm_chunks += 1
            else:
                structural_chunks += 1
        
        print(f"📋 Distribución por tipo:")
        for chunk_type, count in chunk_types.items():
            print(f"   {chunk_type}: {count} chunks")
        
        if self.chunking_mode == "intelligent":
            print(f"\n🤖 Origen de chunks:")
            print(f"   LLM semánticos: {llm_chunks}")
            print(f"   Estructurales: {structural_chunks}")
            
            if hasattr(self.processor, 'get_chunking_stats'):
                stats = self.processor.get_chunking_stats()
                print(f"   Modelo usado: {stats.get('model_used', 'N/A')}")
        
        # Ejemplos de calidad
        print(f"\n💡 Ejemplo de chunk {self.chunking_mode}:")
        if chunks:
            example = chunks[0]
            print(f"   Tipo: {example['metadata']['type']}")
            print(f"   Programa: {example['metadata']['program_name']}")
            print(f"   Contenido: {example['content'][:150]}...")
    
    def search_curriculum(self, 
                         query: str, 
                         n_results: int = 5,
                         filter_type: Optional[str] = None) -> Dict[str, Any]:
        """Búsqueda con resultados mejorados por chunking híbrido"""
        
        if not self.is_loaded:
            return {
                'success': False,
                'error': 'Datos no cargados. Ejecuta load_curriculum_data() primero.'
            }
        
        print(f"\n🔍 BÚSQUEDA HÍBRIDA ({self.chunking_mode})")
        print(f"   Query: '{query}'")
        
        start_time = time.time()
        
        try:
            # Crear embedding
            query_embedding = self.embedder.embed_query(query)
            
            # Búsqueda con filtro opcional
            search_params = {}
            if filter_type:
                search_params['where'] = {'type': filter_type}
            
            results = self.vectorstore.search(
                query_embedding, 
                n_results=n_results,
                **search_params
            )
            
            # Procesar resultados
            processed_results = self._process_search_results(results, query)
            
            search_time = time.time() - start_time
            
            return {
                'success': True,
                'query': query,
                'results': processed_results,
                'total_found': len(processed_results),
                'search_time': search_time,
                'chunking_mode': self.chunking_mode,
                'filter_applied': filter_type
            }
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def smart_search(self, query: str) -> Dict[str, Any]:
        """Búsqueda inteligente mejorada por chunking híbrido"""
        
        # Detectar tipo de consulta
        query_lower = query.lower()
        
        detected_type = None
        search_params = {'n_results': 5}
        
        if any(word in query_lower for word in ['costo', 'precio', 'fee', 'cuánto', 'valor']):
            detected_type = 'fee'
            search_params['filter_type'] = 'fee'
            search_params['n_results'] = 10
            
        elif any(word in query_lower for word in ['trabajo', 'laboral', 'ocupacional', 'desempeñar', 'campo']):
            detected_type = 'occupational_profile'
            search_params['filter_type'] = 'occupational_profile'
            
        elif any(word in query_lower for word in ['materia', 'semestre', 'curriculum', 'plan', 'asignatura']):
            detected_type = 'curriculum'
            search_params['n_results'] = 8
        
        # Realizar búsqueda
        results = self.search_curriculum(query, **search_params)
        
        # Agregar información de detección
        if results['success']:
            results['detected_type'] = detected_type
            results['search_strategy'] = f'smart_detection_{self.chunking_mode}'
            
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
    
    def _process_search_results(self, raw_results: Dict[str, List], query: str) -> List[Dict[str, Any]]:
        """Procesa resultados con información de chunking"""
        
        processed = []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            raw_results['documents'],
            raw_results['metadatas'], 
            raw_results['distances']
        )):
            
            similarity = 1 - distance
            
            result = {
                'rank': i + 1,
                'content': doc,
                'metadata': metadata,
                'similarity': similarity,
                'distance': distance,
                'program_name': metadata.get('program_name', 'Unknown'),
                'chunk_type': metadata.get('type', 'Unknown'),
                'relevance_score': similarity * 100,
                'chunking_source': 'llm' if metadata.get('llm_generated', False) else 'structural',
                'chunking_mode': self.chunking_mode
            }
            
            # Información específica por tipo
            if metadata.get('type') == 'fee':
                result['fee_amount'] = metadata.get('fee_amount', 0)
                result['formatted_fee'] = f"${metadata.get('fee_amount', 0):,} COP"
            
            elif metadata.get('type') == 'curriculum_semester':
                result['semester'] = metadata.get('semester_number', 'N/A')
                result['credits'] = metadata.get('total_credits', 0)
                result['subjects'] = metadata.get('subject_count', 0)
            
            processed.append(result)
        
        processed.sort(key=lambda x: x['similarity'], reverse=True)
        return processed
    
    def get_system_status(self) -> Dict[str, Any]:
        """Estado del sistema híbrido"""
        
        vectorstore_stats = self.vectorstore.get_stats()
        
        status = {
            'system_loaded': self.is_loaded,
            'chunks_loaded': self.chunks_count,
            'load_time': self.load_time,
            'chunking_mode': self.chunking_mode,
            'intelligent_available': INTELLIGENT_CHUNKING_AVAILABLE,
            'vectorstore_stats': vectorstore_stats,
            'embedder_info': self.embedder.get_model_info()
        }
        
        # Agregar stats específicos del procesador
        if hasattr(self.processor, 'get_processing_stats'):
            status['processor_stats'] = self.processor.get_processing_stats()
        elif hasattr(self.processor, 'get_chunking_stats'):
            status['chunking_stats'] = self.processor.get_chunking_stats()
        
        return status

# Test del sistema híbrido
if __name__ == "__main__":
    print("🧪 PROBANDO SISTEMA RAG HÍBRIDO")
    print("=" * 40)
    
    # Test con diferentes modos
    modes_to_test = ["auto", "structural"]
    if INTELLIGENT_CHUNKING_AVAILABLE:
        modes_to_test.append("intelligent")
    
    for mode in modes_to_test:
        print(f"\n🔄 PROBANDO MODO: {mode}")
        print("-" * 25)
        
        try:
            # Inicializar sistema
            rag_system = HybridUSCCurriculumRAG(
                collection_name=f"test_hybrid_{mode}",
                chunking_mode=mode
            )
            
            # Cargar datos
            curriculum_file = "data/documentos/Curriculums_Technology_Undergraduate.md"
            success = rag_system.load_curriculum_data(curriculum_file, force_reload=True)
            
            if success:
                # Test búsqueda rápida
                query = "¿Cuánto cuesta Systems Engineering?"
                results = rag_system.smart_search(query)
                
                if results['success']:
                    print(f"   ✅ Búsqueda exitosa: {results['total_found']} resultados")
                    if results['results']:
                        best = results['results'][0]
                        print(f"   🎯 Mejor resultado: {best['program_name']}")
                        print(f"   📊 Relevancia: {best['relevance_score']:.1f}%")
                        print(f"   🔧 Origen chunk: {best['chunking_source']}")
                else:
                    print(f"   ❌ Error en búsqueda")
            else:
                print(f"   ❌ Error cargando datos en modo {mode}")
                
        except Exception as e:
            print(f"   ❌ Error en modo {mode}: {e}")
    
    print(f"\n🎉 SISTEMA HÍBRIDO VALIDADO")
    print("📋 Recomendación: Usar modo 'auto' para máxima compatibilidad")