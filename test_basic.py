"""
Test Completo y Mejorado del Sistema RAG USC
Verificación paso a paso con análisis detallado
"""

import sys
sys.path.append('src')

from src.embeddings.bge_embeddings import BGEEmbeddings
from src.rag.vector_store import LocalVectorStore
from src.rag.curriculum_processor import USCCurriculumProcessor
import time

class USCTestSuite:
    def __init__(self):
        self.results = {
            'file_analysis': {},
            'processor_test': {},
            'embedding_test': {},
            'vector_store_test': {},
            'search_test': {},
            'overall_success': False
        }
        
        print("🧪 TEST SUITE COMPLETO - SISTEMA RAG USC")
        print("=" * 60)
    
    def test_1_file_analysis(self):
        """Test 1: Análisis detallado del archivo"""
        
        print("\n1️⃣ ANÁLISIS DEL ARCHIVO DE CURRÍCULUMS")
        print("-" * 45)
        
        file_path = "data/documentos/Curriculums_Technology_Undergraduate.md"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Buscar secciones principales
            tech_start = None
            undergrad_start = None
            
            for i, line in enumerate(lines):
                if line.strip() == "## PROGRAMAS DE TECNOLOGIA":
                    tech_start = i
                elif line.strip() == "## ESTUDIOS DE PREGRADO":
                    undergrad_start = i
            
            # Buscar programas principales manualmente
            tech_programs = []
            undergrad_programs = []
            
            # Programas de tecnología
            if tech_start and undergrad_start:
                for i in range(tech_start, undergrad_start):
                    line = lines[i].strip()
                    if line.startswith('### ') and not any(word in line.lower() for word in ['semestre', 'semester']):
                        program_name = line.replace('###', '').strip()
                        tech_programs.append((program_name, i))
            
            # Programas de pregrado
            if undergrad_start:
                for i in range(undergrad_start, len(lines)):
                    line = lines[i].strip()
                    if line.startswith('### ') and not any(word in line.lower() for word in ['semestre', 'semester']):
                        program_name = line.replace('###', '').strip()
                        undergrad_programs.append((program_name, i))
            
            # Contar semestres totales
            total_semesters = len([line for line in lines if line.strip().startswith('#### Semestre')])
            
            # Resultados
            self.results['file_analysis'] = {
                'file_size': len(content),
                'total_lines': len(lines),
                'tech_section_start': tech_start,
                'undergrad_section_start': undergrad_start,
                'tech_programs': tech_programs,
                'undergrad_programs': undergrad_programs,
                'total_tech': len(tech_programs),
                'total_undergrad': len(undergrad_programs),
                'total_programs': len(tech_programs) + len(undergrad_programs),
                'total_semesters': total_semesters,
                'success': True
            }
            
            print(f"✅ Archivo encontrado: {file_path}")
            print(f"   📊 Tamaño: {len(content):,} caracteres")
            print(f"   📋 Líneas: {len(lines):,}")
            print(f"   📍 Sección Tecnología: línea {tech_start}")
            print(f"   📍 Sección Pregrado: línea {undergrad_start}")
            print(f"")
            print(f"📚 PROGRAMAS DE TECNOLOGÍA ({len(tech_programs)}):")
            for i, (name, line_num) in enumerate(tech_programs, 1):
                print(f"   {i}. {name} (línea {line_num})")
            
            print(f"")
            print(f"🎓 PROGRAMAS DE PREGRADO ({len(undergrad_programs)}):")
            for i, (name, line_num) in enumerate(undergrad_programs, 1):
                print(f"   {i}. {name} (línea {line_num})")
            
            print(f"")
            print(f"📊 RESUMEN:")
            print(f"   Total programas: {len(tech_programs) + len(undergrad_programs)}")
            print(f"   Total semestres: {total_semesters}")
            print(f"   ✅ Análisis completado")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en análisis: {e}")
            self.results['file_analysis']['success'] = False
            return False
    
    def test_2_processor(self):
        """Test 2: Procesador de currículums"""
        
        print(f"\n2️⃣ TEST DEL PROCESADOR")
        print("-" * 25)
        
        try:
            processor = USCCurriculumProcessor()
            file_path = "data/documentos/Curriculums_Technology_Undergraduate.md"
            
            start_time = time.time()
            chunks = processor.process_curriculum_file(file_path)
            processing_time = time.time() - start_time
            
            stats = processor.get_processing_stats()
            
            # Análisis de chunks por tipo
            chunk_types = {}
            programs_found = set()
            
            for chunk in chunks:
                chunk_type = chunk['metadata']['type']
                program_name = chunk['metadata']['program_name']
                
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                programs_found.add(program_name)
            
            self.results['processor_test'] = {
                'total_chunks': len(chunks),
                'tech_programs': stats['technology_programs'],
                'undergrad_programs': stats['undergraduate_programs'],
                'total_programs': stats['programs_processed'],
                'semester_chunks': stats['semester_chunks'],
                'processing_time': processing_time,
                'chunk_types': chunk_types,
                'programs_found': list(programs_found),
                'success': len(chunks) > 0
            }
            
            print(f"✅ Procesador inicializado")
            print(f"   🔄 Tiempo de procesamiento: {processing_time:.3f}s")
            print(f"   📊 Chunks creados: {len(chunks)}")
            print(f"   🏭 Programas tecnología: {stats['technology_programs']}")
            print(f"   🎓 Programas pregrado: {stats['undergraduate_programs']}")
            print(f"   📚 Total programas: {stats['programs_processed']}")
            print(f"   📖 Chunks de semestres: {stats['semester_chunks']}")
            
            print(f"")
            print(f"📋 TIPOS DE CHUNKS:")
            for chunk_type, count in chunk_types.items():
                print(f"   {chunk_type}: {count}")
            
            print(f"")
            print(f"🏷️ PROGRAMAS PROCESADOS ({len(programs_found)}):")
            for i, program in enumerate(sorted(programs_found), 1):
                print(f"   {i}. {program}")
            
            # Comparar con análisis de archivo
            expected_total = self.results['file_analysis']['total_programs']
            actual_total = stats['programs_processed']
            
            if actual_total == expected_total:
                print(f"   ✅ Cantidad correcta: {actual_total}/{expected_total}")
            else:
                print(f"   ⚠️  Diferencia: {actual_total}/{expected_total} programas")
                
                # Mostrar qué falta
                expected_tech = self.results['file_analysis']['total_tech']
                actual_tech = stats['technology_programs']
                
                if actual_tech != expected_tech:
                    print(f"   🔍 Tecnología: {actual_tech}/{expected_tech}")
                    
                    expected_programs = [name for name, _ in self.results['file_analysis']['tech_programs']]
                    missing = [prog for prog in expected_programs if prog not in programs_found]
                    if missing:
                        print(f"   ❌ Programas tecnología faltantes:")
                        for prog in missing:
                            print(f"      - {prog}")
            
            return len(chunks) > 0
            
        except Exception as e:
            print(f"❌ Error en procesador: {e}")
            import traceback
            traceback.print_exc()
            self.results['processor_test']['success'] = False
            return False
    
    def test_3_embeddings(self):
        """Test 3: BGE-M3 Embeddings"""
        
        print(f"\n3️⃣ TEST DE EMBEDDINGS BGE-M3")
        print("-" * 30)
        
        try:
            embedder = BGEEmbeddings()
            
            # Test con datos del procesador
            if not self.results['processor_test']['success']:
                print(f"❌ Saltando test - procesador falló")
                return False
            
            # Crear embeddings de prueba
            test_texts = [
                "Ingeniería de Sistemas cuesta $5.298.134 COP",
                "Bioingeniería perfil ocupacional tecnologías médicas",
                "Primer semestre matemáticas fundamentales programación"
            ]
            
            start_time = time.time()
            embeddings = embedder.embed_documents(test_texts)
            embed_time = time.time() - start_time
            
            # Test embedding de consulta
            query = "¿Cuánto cuesta Ingeniería de Sistemas?"
            query_embedding = embedder.embed_query(query)
            
            self.results['embedding_test'] = {
                'embeddings_shape': embeddings.shape,
                'query_embedding_shape': query_embedding.shape,
                'embedding_time': embed_time,
                'dimension': embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
                'success': embeddings.shape[0] > 0
            }
            
            print(f"✅ BGE-M3 cargado correctamente")
            print(f"   📏 Dimensiones: {embeddings.shape[1]}")
            print(f"   🔄 Tiempo embedding: {embed_time:.3f}s")
            print(f"   📊 Test embeddings: {embeddings.shape}")
            print(f"   🔍 Query embedding: {query_embedding.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en embeddings: {e}")
            self.results['embedding_test']['success'] = False
            return False
    
    def test_4_vector_store(self):
        """Test 4: Vector Store y almacenamiento"""
        
        print(f"\n4️⃣ TEST DEL VECTOR STORE")
        print("-" * 25)
        
        try:
            vs = LocalVectorStore(collection_name="test_complete_usc")
            vs.clear_collection()
            
            # Usar datos reales del procesador
            if not self.results['processor_test']['success']:
                print(f"❌ Saltando test - procesador falló")
                return False
            
            # Procesar archivo completo
            processor = USCCurriculumProcessor()
            embedder = BGEEmbeddings()
            
            chunks = processor.process_curriculum_file("data/documentos/Curriculums_Technology_Undergraduate.md")
            
            texts = [chunk['content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            start_time = time.time()
            embeddings = embedder.embed_documents(texts)
            embed_time = time.time() - start_time
            
            start_time = time.time()
            success = vs.add_documents(texts, metadatas, embeddings)
            store_time = time.time() - start_time
            
            stats = vs.get_stats()
            
            self.results['vector_store_test'] = {
                'documents_stored': stats['total_documents'],
                'embedding_time': embed_time,
                'storage_time': store_time,
                'success': success and stats['total_documents'] > 0
            }
            
            print(f"✅ Vector store configurado")
            print(f"   🔄 Tiempo embeddings: {embed_time:.3f}s")
            print(f"   💾 Tiempo almacenamiento: {store_time:.3f}s")
            print(f"   📊 Documentos almacenados: {stats['total_documents']}")
            print(f"   🏷️ Tipos de documentos: {len(stats['document_types'])}")
            print(f"   🎓 Programas: {len(stats['programs'])}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error en vector store: {e}")
            self.results['vector_store_test']['success'] = False
            return False
    
    def test_5_search_functionality(self):
        """Test 5: Funcionalidad de búsqueda"""
        
        print(f"\n5️⃣ TEST DE BÚSQUEDA SEMÁNTICA")
        print("-" * 32)
        
        try:
            if not self.results['vector_store_test']['success']:
                print(f"❌ Saltando test - vector store falló")
                return False
            
            vs = LocalVectorStore(collection_name="test_complete_usc")
            embedder = BGEEmbeddings()
            
            # Consultas de prueba
            test_queries = [
                {
                    'query': '¿Cuánto cuesta Ingeniería de Sistemas?',
                    'expected_type': 'fee',
                    'expected_program': 'Ingenieria de Sistemas'
                },
                {
                    'query': '¿En qué puede trabajar un bioingeniero?',
                    'expected_type': 'occupational_profile', 
                    'expected_program': 'Bioingenieria'
                },
                {
                    'query': '¿Qué materias tiene primer semestre de Sistemas?',
                    'expected_type': 'curriculum_semester',
                    'expected_program': 'Ingenieria de Sistemas'
                },
                {
                    'query': 'programa más barato',
                    'expected_type': 'fee',
                    'expected_program': None  # Cualquier programa con costo
                }
            ]
            
            search_results = []
            
            print(f"🔍 Probando {len(test_queries)} consultas:")
            
            for i, test_case in enumerate(test_queries, 1):
                query = test_case['query']
                
                start_time = time.time()
                query_emb = embedder.embed_query(query)
                results = vs.search(query_emb, n_results=3)
                search_time = time.time() - start_time
                
                if results['documents']:
                    best_doc = results['documents'][0]
                    best_meta = results['metadatas'][0]
                    similarity = 1 - results['distances'][0]
                    
                    result = {
                        'query': query,
                        'similarity': similarity,
                        'program_found': best_meta.get('program_name', 'N/A'),
                        'type_found': best_meta.get('type', 'N/A'),
                        'search_time': search_time,
                        'success': similarity > 0.3  # Umbral mínimo
                    }
                    
                    search_results.append(result)
                    
                    status = "✅" if result['success'] else "⚠️"
                    print(f"   {i}. {status} '{query}'")
                    print(f"      Similitud: {similarity:.3f}")
                    print(f"      Programa: {result['program_found']}")
                    print(f"      Tipo: {result['type_found']}")
                    print(f"      Tiempo: {search_time:.3f}s")
                else:
                    print(f"   {i}. ❌ '{query}' - Sin resultados")
                    search_results.append({
                        'query': query,
                        'success': False,
                        'similarity': 0,
                        'search_time': search_time
                    })
            
            successful_searches = len([r for r in search_results if r['success']])
            avg_similarity = sum([r['similarity'] for r in search_results]) / len(search_results)
            avg_search_time = sum([r['search_time'] for r in search_results]) / len(search_results)
            
            self.results['search_test'] = {
                'total_queries': len(test_queries),
                'successful_queries': successful_searches,
                'avg_similarity': avg_similarity,
                'avg_search_time': avg_search_time,
                'search_results': search_results,
                'success': successful_searches >= len(test_queries) // 2  # Al menos 50% exitosas
            }
            
            print(f"")
            print(f"📊 RESUMEN BÚSQUEDAS:")
            print(f"   Exitosas: {successful_searches}/{len(test_queries)}")
            print(f"   Similitud promedio: {avg_similarity:.3f}")
            print(f"   Tiempo promedio: {avg_search_time:.3f}s")
            
            return successful_searches > 0
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            self.results['search_test']['success'] = False
            return False
    
    def generate_report(self):
        """Generar reporte final"""
        
        print(f"\n" + "=" * 60)
        print(f"📊 REPORTE FINAL DEL SISTEMA RAG USC")
        print(f"=" * 60)
        
        # Verificar éxito general
        all_tests = [
            self.results['file_analysis'].get('success', False),
            self.results['processor_test'].get('success', False),
            self.results['embedding_test'].get('success', False),
            self.results['vector_store_test'].get('success', False),
            self.results['search_test'].get('success', False)
        ]
        
        self.results['overall_success'] = all(all_tests)
        
        print(f"")
        print(f"✅ TESTS PASADOS:")
        tests = [
            ("Análisis de archivo", self.results['file_analysis'].get('success', False)),
            ("Procesador", self.results['processor_test'].get('success', False)),
            ("Embeddings BGE-M3", self.results['embedding_test'].get('success', False)),
            ("Vector Store", self.results['vector_store_test'].get('success', False)),
            ("Búsqueda semántica", self.results['search_test'].get('success', False))
        ]
        
        for test_name, success in tests:
            status = "✅" if success else "❌"
            print(f"   {status} {test_name}")
        
        print(f"")
        print(f"📈 MÉTRICAS CLAVE:")
        
        if 'file_analysis' in self.results and self.results['file_analysis'].get('success'):
            fa = self.results['file_analysis']
            print(f"   📚 Programas en archivo: {fa['total_programs']} (Tech: {fa['total_tech']}, Pregrado: {fa['total_undergrad']})")
        
        if 'processor_test' in self.results and self.results['processor_test'].get('success'):
            pt = self.results['processor_test']
            print(f"   🔄 Programas procesados: {pt['total_programs']} (Tech: {pt['tech_programs']}, Pregrado: {pt['undergrad_programs']})")
            print(f"   📄 Chunks generados: {pt['total_chunks']}")
            print(f"   ⏱️  Tiempo procesamiento: {pt['processing_time']:.3f}s")
        
        if 'embedding_test' in self.results and self.results['embedding_test'].get('success'):
            et = self.results['embedding_test']
            print(f"   📏 Dimensión embeddings: {et['dimension']}")
        
        if 'vector_store_test' in self.results and self.results['vector_store_test'].get('success'):
            vt = self.results['vector_store_test']
            print(f"   💾 Documentos almacenados: {vt['documents_stored']}")
        
        if 'search_test' in self.results and self.results['search_test'].get('success'):
            st = self.results['search_test']
            print(f"   🔍 Búsquedas exitosas: {st['successful_queries']}/{st['total_queries']}")
            print(f"   📊 Similitud promedio: {st['avg_similarity']:.3f}")
        
        print(f"")
        if self.results['overall_success']:
            print(f"🎉 ¡SISTEMA RAG COMPLETAMENTE FUNCIONAL!")
            print(f"   ✅ Todos los componentes operativos")
            print(f"   ✅ Listo para continuar con PASO 2: Timeouts LLM")
        else:
            print(f"⚠️  SISTEMA PARCIALMENTE FUNCIONAL")
            failed_tests = [name for name, success in tests if not success]
            print(f"   ❌ Tests fallidos: {', '.join(failed_tests)}")
            print(f"   🔧 Revisar componentes antes de continuar")
        
        return self.results['overall_success']

def run_complete_test():
    """Ejecutar suite completa de tests"""
    
    suite = USCTestSuite()
    
    # Ejecutar tests en orden
    tests = [
        suite.test_1_file_analysis,
        suite.test_2_processor,
        suite.test_3_embeddings,
        suite.test_4_vector_store,
        suite.test_5_search_functionality
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"❌ Error crítico en test: {e}")
            break
    
    # Generar reporte final
    return suite.generate_report()

if __name__ == "__main__":
    success = run_complete_test()
    
    if success:
        print(f"\n🚀 SIGUIENTE PASO: Implementar timeouts LLM para chunking avanzado")
    else:
        print(f"\n🔧 ACCIÓN REQUERIDA: Revisar componentes fallidos antes de continuar")