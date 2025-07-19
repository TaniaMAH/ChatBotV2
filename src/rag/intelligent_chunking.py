"""
Chunking Inteligente para USC
Combina análisis estructural + LLM para chunking semántico adaptativo
"""

import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
from datetime import datetime

class IntelligentCurriculumChunker:
    def __init__(self, use_llm: bool = True, ollama_url: str = "http://localhost:11434"):
        """
        Chunker inteligente que combina reglas + LLM
        
        Args:
            use_llm: Si usar LLM para chunking semántico
            ollama_url: URL del servidor Ollama local
        """
        
        self.use_llm = use_llm
        self.ollama_url = ollama_url
        self.fallback_to_structural = True
        
        # Patrones estructurales como fallback
        self.structural_patterns = {
            'program_header': r'### ([^\n]+)',
            'fee': r'\*\*Fee:\*\* \$([0-9,\.]+) \(💰cop\)',
            'occupational_profile': r'\*\*Occupational Profile:\*\*\s*(.*?)(?=\*\*Curriculum:)',
            'curriculum_start': r'\*\*Curriculum:\*\*',
            'semester': r'#### Semester ([IVX\d]+)\s*\n((?:- .+\n)*)',
            'subject': r'- ([^|]+) \| (\d+) Créditos'
        }
        
        self.stats = {
            'llm_chunks': 0,
            'structural_chunks': 0,
            'hybrid_chunks': 0,
            'processing_time': 0
        }
        
        print("🧠 Chunker Inteligente inicializado")
        print(f"   🤖 Modo LLM: {'✅' if use_llm else '❌'}")
        if use_llm:
            self._test_llm_connection()
            
            # Pre-calentar modelo si está disponible
            if hasattr(self, 'selected_model'):
                print(f"🔥 Pre-calentando modelo {self.selected_model}...")
                try:
                    response = requests.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": self.selected_model,
                            "prompt": "Hola",
                            "stream": False,
                            "options": {"num_predict": 5}
                        },
                        timeout=120
                    )
                    if response.status_code == 200:
                        print("✅ Modelo pre-calentado y listo")
                    else:
                        print("⚠️  Modelo no respondió, usando fallback estructural")
                except Exception as e:
                    print(f"⚠️  Error pre-calentando: {e}")
    
    def _test_llm_connection(self) -> bool:
        """Prueba conexión con Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"   🔗 Ollama conectado: {len(models)} modelos disponibles")
                
                # Buscar modelo recomendado
                model_names = [m['name'] for m in models]
                preferred_models = ['llama3.2', 'llama3.1', 'llama3', 'mistral']
                
                self.selected_model = None
                for model in preferred_models:
                    matching = [m for m in model_names if model in m.lower()]
                    if matching:
                        self.selected_model = matching[0]
                        print(f"   🎯 Modelo seleccionado: {self.selected_model}")
                        break
                
                if not self.selected_model and models:
                    self.selected_model = models[0]['name']
                    print(f"   ⚠️  Usando modelo por defecto: {self.selected_model}")
                
                return True
            else:
                print(f"   ❌ Error conectando Ollama: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Ollama no disponible: {e}")
            print(f"   💡 Instala Ollama: https://ollama.ai")
            return False
    
    def chunk_program_intelligently(self, program_text: str, program_name: str) -> List[Dict[str, Any]]:
        """
        Chunking inteligente de un programa específico - VERSIÓN ROBUSTA
        """
        
        print(f"🧠 Analizando programa: {program_name}")
        
        # SIEMPRE generar chunks estructurales como base
        structural_chunks = self._structural_chunking(program_text, program_name)
        self.stats['structural_chunks'] += len(structural_chunks)
        
        # INTENTAR LLM chunking si está disponible
        llm_chunks = []
        if self.use_llm and hasattr(self, 'selected_model'):
            llm_chunks = self._llm_semantic_chunking(program_text, program_name)
            if llm_chunks:
                self.stats['llm_chunks'] += len(llm_chunks)
                print(f"   🤖 LLM generó: {len(llm_chunks)} chunks semánticos")
        
        # DECIDIR ESTRATEGIA DE FUSIÓN
        if llm_chunks and structural_chunks:
            # Fusión inteligente
            hybrid_chunks = self._merge_chunks_intelligently(llm_chunks, structural_chunks)
            self.stats['hybrid_chunks'] += len(hybrid_chunks)
            print(f"   🔄 Fusión híbrida: {len(hybrid_chunks)} chunks finales")
            return hybrid_chunks
        
        elif llm_chunks:
            # Solo LLM chunks
            print(f"   🤖 Solo LLM: {len(llm_chunks)} chunks")
            return llm_chunks
        
        else:
            # Solo estructural (garantizado)
            print(f"   📋 Fallback estructural: {len(structural_chunks)} chunks")
            return structural_chunks
    
    def _llm_semantic_chunking(self, program_text: str, program_name: str) -> List[Dict[str, Any]]:
        """Chunking semántico usando LLM con reintentos"""
        
        # INTENTAR 2 VECES CON TIMEOUTS DIFERENTES
        timeouts = [120, 300]  # 2 min, luego 5 min
        
        for attempt, timeout in enumerate(timeouts, 1):
            print(f"   🔄 Intento {attempt}/{len(timeouts)} (timeout: {timeout}s)")
            
            # PROMPT ADAPTATIVO SEGÚN EL INTENTO
            if attempt == 1:
                # Primer intento: prompt completo
                prompt = f"""
Eres un experto en análisis de documentos académicos. Analiza este programa universitario y divide el contenido en chunks semánticamente coherentes.

PROGRAMA: {program_name}

CONTENIDO:
{program_text}

INSTRUCCIONES:
1. Identifica automáticamente las secciones principales (costo, perfil ocupacional, curriculum, etc.)
2. Crea chunks que mantengan contexto semántico completo
3. Cada chunk debe ser autocontenido y útil para búsquedas
4. Extrae metadatos relevantes (costo numérico, número de semestres, etc.)

FORMATO DE RESPUESTA (JSON):
{{
    "chunks": [
        {{
            "content": "Texto del chunk con contexto completo",
            "type": "fee|occupational_profile|curriculum_summary|curriculum_semester|program_overview",
            "metadata": {{
                "program_name": "{program_name}",
                "semantic_focus": "Enfoque semántico del chunk",
                "extracted_entities": ["entidad1", "entidad2"],
                "chunk_strategy": "llm_semantic"
            }}
        }}
    ]
}}

Responde SOLO con el JSON válido:
"""
            else:
                # Segundo intento: prompt simplificado
                prompt = f"""Analiza este programa y extrae información clave:

PROGRAMA: {program_name}
CONTENIDO: {program_text[:1500]}

Crea chunks JSON:
{{"chunks": [{{"content": "texto", "type": "fee|occupational_profile|curriculum_summary", "metadata": {{"program_name": "{program_name}"}}}}]}}"""
            
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.selected_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.9,
                            "num_predict": 300 if attempt == 2 else 1000  # Menos tokens en reintento
                        }
                    },
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    llm_output = response.json()['response']
                    
                    # Extraer JSON del output
                    json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
                    if json_match:
                        try:
                            chunks_data = json.loads(json_match.group())
                            
                            # Procesar chunks del LLM
                            processed_chunks = []
                            for chunk in chunks_data.get('chunks', []):
                                processed_chunk = {
                                    'content': chunk['content'],
                                    'metadata': {
                                        **chunk.get('metadata', {}),
                                        'type': chunk['type'],
                                        'llm_generated': True,
                                        'confidence': 0.9,
                                        'attempt': attempt
                                    }
                                }
                                processed_chunks.append(processed_chunk)
                            
                            print(f"   ✅ LLM exitoso en intento {attempt}")
                            return processed_chunks
                        
                        except json.JSONDecodeError:
                            print(f"   ⚠️  JSON inválido en intento {attempt}")
                            if attempt == len(timeouts):
                                return []
                            continue
                    else:
                        print(f"   ⚠️  Sin JSON en intento {attempt}")
                        if attempt == len(timeouts):
                            return []
                        continue
                else:
                    print(f"   ❌ Error HTTP {response.status_code} en intento {attempt}")
                    if attempt == len(timeouts):
                        return []
                    continue
                    
            except requests.exceptions.Timeout:
                print(f"   ⏰ Timeout en intento {attempt}")
                if attempt == len(timeouts):
                    return []
                continue
            except Exception as e:
                print(f"   ❌ Error en intento {attempt}: {e}")
                if attempt == len(timeouts):
                    return []
                continue
        
        return []  # Si todos los intentos fallan
    
    def _structural_chunking(self, program_text: str, program_name: str) -> List[Dict[str, Any]]:
        """Chunking estructural basado en patrones - VERSIÓN MEJORADA"""
        
        chunks = []
        
        # CHUNK 1: Programa completo (SIEMPRE crear este)
        complete_chunk = {
            'content': program_text,
            'metadata': {
                'type': 'program_complete',
                'program_name': program_name,
                'chunk_strategy': 'structural_complete',
                'confidence': 1.0
            }
        }
        chunks.append(complete_chunk)
        
        # CHUNK 2: Fee (si existe)
        fee_match = re.search(self.structural_patterns['fee'], program_text)
        if fee_match:
            fee_amount_str = fee_match.group(1).replace(',', '').replace('.', '')
            fee_amount = int(fee_amount_str) if fee_amount_str.isdigit() else 0
            
            fee_chunk = {
                'content': f"{program_name}\n\nCosto: ${fee_match.group(1)} COP por semestre\n\nInformación de matrícula para {program_name}.",
                'metadata': {
                    'type': 'fee',
                    'program_name': program_name,
                    'fee_amount': fee_amount,
                    'chunk_strategy': 'structural_pattern',
                    'confidence': 0.95
                }
            }
            chunks.append(fee_chunk)
        
        # CHUNK 3: Perfil Ocupacional (si existe)
        profile_match = re.search(self.structural_patterns['occupational_profile'], program_text, re.DOTALL)
        if profile_match:
            profile_chunk = {
                'content': f"{program_name} - Perfil Ocupacional\n\n{profile_match.group(1).strip()}\n\nCampo laboral y oportunidades profesionales.",
                'metadata': {
                    'type': 'occupational_profile',
                    'program_name': program_name,
                    'chunk_strategy': 'structural_pattern',
                    'confidence': 0.9
                }
            }
            chunks.append(profile_chunk)
        
        # CHUNK 4: Curriculum Summary (si tiene curriculum)
        if '**Curriculum:**' in program_text:
            curriculum_start = program_text.find('**Curriculum:**')
            curriculum_content = program_text[curriculum_start:curriculum_start + 500]
            
            curriculum_chunk = {
                'content': f"{program_name} - Plan de Estudios\n\n{curriculum_content}{'...' if len(program_text[curriculum_start:]) > 500 else ''}",
                'metadata': {
                    'type': 'curriculum_summary',
                    'program_name': program_name,
                    'chunk_strategy': 'structural_curriculum',
                    'confidence': 0.9
                }
            }
            chunks.append(curriculum_chunk)
        
        # CHUNK 5+: Semestres individuales
        semester_matches = re.findall(self.structural_patterns['semester'], program_text)
        for semester_num, semester_content in semester_matches:
            subjects = re.findall(self.structural_patterns['subject'], semester_content)
            
            if subjects:
                total_credits = sum(int(credits) for _, credits in subjects)
                
                semester_text = f"{program_name} - Semestre {semester_num}\n\n"
                semester_text += f"Total materias: {len(subjects)}\n"
                semester_text += f"Total créditos: {total_credits}\n\nMaterias:\n"
                
                for subject_name, credits in subjects:
                    semester_text += f"- {subject_name.strip()} ({credits} créditos)\n"
                
                semester_chunk = {
                    'content': semester_text,
                    'metadata': {
                        'type': 'curriculum_semester',
                        'program_name': program_name,
                        'semester_number': semester_num,
                        'subject_count': len(subjects),
                        'total_credits': total_credits,
                        'chunk_strategy': 'structural_pattern',
                        'confidence': 0.85
                    }
                }
                chunks.append(semester_chunk)
            else:
                # Crear chunk de semestre aunque no tenga materias detectadas
                semester_chunk = {
                    'content': f"{program_name} - Semestre {semester_num}\n\n{semester_content.strip()}",
                    'metadata': {
                        'type': 'curriculum_semester',
                        'program_name': program_name,
                        'semester_number': semester_num,
                        'chunk_strategy': 'structural_fallback',
                        'confidence': 0.7
                    }
                }
                chunks.append(semester_chunk)
        
        return chunks
    
    def _merge_chunks_intelligently(self, llm_chunks: List[Dict], structural_chunks: List[Dict]) -> List[Dict]:
        """Fusiona chunks de LLM y estructurales inteligentemente"""
        
        merged = []
        
        # Usar LLM chunks como base (mayor calidad semántica)
        for llm_chunk in llm_chunks:
            merged.append(llm_chunk)
        
        # Agregar chunks estructurales que no estén cubiertos por LLM
        for struct_chunk in structural_chunks:
            struct_type = struct_chunk['metadata']['type']
            struct_program = struct_chunk['metadata']['program_name']
            
            # Verificar si LLM ya cubrió este tipo de chunk
            llm_covers = any(
                c['metadata']['type'] == struct_type and 
                c['metadata']['program_name'] == struct_program 
                for c in llm_chunks
            )
            
            if not llm_covers:
                # Agregar chunk estructural no cubierto
                struct_chunk['metadata']['source'] = 'structural_fallback'
                merged.append(struct_chunk)
        
        return merged
    
    def process_full_curriculum(self, file_path: str) -> List[Dict[str, Any]]:
        """Procesa archivo completo con chunking inteligente"""
        
        start_time = datetime.now()
        
        print(f"🧠 PROCESANDO CON CHUNKING INTELIGENTE")
        print(f"   📄 Archivo: {file_path}")
        
        # Leer archivo
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Error leyendo archivo: {e}")
            return []
        
        # Extraer programas individuales
        programs = self._extract_programs(content)
        
        all_chunks = []
        
        for program_name, program_content in programs:
            print(f"\n🔄 Procesando: {program_name}")
            program_chunks = self.chunk_program_intelligently(program_content, program_name)
            all_chunks.extend(program_chunks)
        
        # Estadísticas finales
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        
        print(f"\n📊 CHUNKING INTELIGENTE COMPLETADO")
        print(f"   Programas procesados: {len(programs)}")
        print(f"   Chunks LLM: {self.stats['llm_chunks']}")
        print(f"   Chunks estructurales: {self.stats['structural_chunks']}")
        print(f"   Chunks híbridos: {self.stats['hybrid_chunks']}")
        print(f"   Total chunks: {len(all_chunks)}")
        print(f"   Tiempo: {self.stats['processing_time']:.2f}s")
        
        return all_chunks
    
    def _extract_programs(self, content: str) -> List[tuple]:
        """Extrae programas del contenido - VERSIÓN FINAL CORREGIDA"""
        
        programs = []
        
        print("🔍 Analizando estructura del archivo...")
        
        # MÉTODO ROBUSTO: Buscar secciones con múltiples patrones
        tech_patterns = [
            r'## TECHNOLOGY PROGRAMS\s*\n(.*?)(?=## UNDERGRADUATE STUDIES|\Z)',
            r'## TECHNOLOGY PROGRAMS(.*?)(?=## UNDERGRADUATE|\Z)'
        ]
        
        undergrad_patterns = [
            r'## UNDERGRADUATE STUDIES\s*\n(.*?)$',
            r'## UNDERGRADUATE STUDIES(.*?)$',
            r'## UNDERGRADUATE STUDIES\s*\n(.*?)(?=\Z)',
            r'##\s*UNDERGRADUATE\s*STUDIES\s*\n(.*?)$'
        ]
        
        # Buscar sección Technology
        tech_match = None
        for pattern in tech_patterns:
            tech_match = re.search(pattern, content, re.DOTALL)
            if tech_match:
                break
        
        # Buscar sección Undergraduate
        undergrad_match = None
        for pattern in undergrad_patterns:
            undergrad_match = re.search(pattern, content, re.DOTALL)
            if undergrad_match:
                break
        
        print(f"🔍 Debug secciones:")
        print(f"   Technology encontrada: {'✅' if tech_match else '❌'}")
        print(f"   Undergraduate encontrada: {'✅' if undergrad_match else '❌'}")
        
        # Procesar sección Technology
        if tech_match:
            tech_content = tech_match.group(1)
            print(f"   Technology contenido: {len(tech_content)} chars")
            tech_programs = self._extract_programs_from_section(tech_content, "Technology")
            programs.extend(tech_programs)
            print(f"✅ Technology: {len(tech_programs)} programas extraídos")
        else:
            print("⚠️  Sección Technology no encontrada")
        
        # Procesar sección Undergraduate  
        if undergrad_match:
            undergrad_content = undergrad_match.group(1)
            print(f"   Undergraduate contenido: {len(undergrad_content)} chars")
            undergrad_programs = self._extract_programs_from_section(undergrad_content, "Undergraduate")
            programs.extend(undergrad_programs)
            print(f"✅ Undergraduate: {len(undergrad_programs)} programas extraídos")
        else:
            print("⚠️  Sección Undergraduate no encontrada")
            
            # FALLBACK: Buscar manualmente todos los programas después de UNDERGRADUATE
            undergrad_pos = content.find("## UNDERGRADUATE")
            if undergrad_pos != -1:
                print("🔄 Intentando extracción manual de Undergraduate...")
                undergrad_content = content[undergrad_pos:]
                undergrad_programs = self._extract_programs_from_section(undergrad_content, "Undergraduate")
                programs.extend(undergrad_programs)
                print(f"✅ Undergraduate (manual): {len(undergrad_programs)} programas extraídos")
        
        print(f"📊 Total programas extraídos: {len(programs)}")
        
        # Debug: mostrar programas encontrados
        if programs:
            print("📋 Programas encontrados:")
            for i, (name, _) in enumerate(programs):
                print(f"   {i+1}. {name}")
        
        return programs
    
    def _extract_programs_from_section(self, section_content: str, section_type: str) -> List[tuple]:
        """Extrae programas de una sección específica - VERSIÓN FINAL"""
        
        programs = []
        
        # Dividir por ### pero conservar contexto
        lines = section_content.split('\n')
        current_program = []
        current_program_name = None
        
        for line in lines:
            if line.startswith('### '):
                # Si ya tenemos un programa en proceso, guardarlo
                if current_program_name and current_program:
                    # FILTRO: Solo guardar si NO es un semestre individual
                    if not re.match(r'^Semester [IVX\d]+$', current_program_name.strip()):
                        full_content = '\n'.join(current_program)
                        programs.append((current_program_name, full_content))
                
                # Iniciar nuevo programa
                current_program_name = line[4:].strip()  # Remover "### "
                current_program = [line]  # Incluir el header
                
            elif current_program_name:
                # Agregar línea al programa actual
                current_program.append(line)
        
        # No olvidar el último programa
        if current_program_name and current_program:
            # FILTRO: Solo guardar si NO es un semestre individual
            if not re.match(r'^Semester [IVX\d]+$', current_program_name.strip()):
                full_content = '\n'.join(current_program)
                programs.append((current_program_name, full_content))
        
        return programs
    
    def get_chunking_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del chunking inteligente"""
        return {
            **self.stats,
            'llm_enabled': self.use_llm and hasattr(self, 'selected_model'),
            'model_used': getattr(self, 'selected_model', None),
            'chunking_mode': 'hybrid' if self.stats['hybrid_chunks'] > 0 else
                           'llm_only' if self.stats['llm_chunks'] > 0 else 'structural_only'
        }

# Test del chunking inteligente
if __name__ == "__main__":
    print("🧪 PROBANDO CHUNKING VERDADERAMENTE INTELIGENTE")
    print("=" * 50)
    
    # Crear chunker inteligente
    chunker = IntelligentCurriculumChunker(use_llm=True)
    
    # Test con archivo de curriculum
    curriculum_file = "data/documentos/Curriculums_Technology_Undergraduate.md"
    
    if Path(curriculum_file).exists():
        print(f"\n📄 Procesando archivo: {curriculum_file}")
        
        chunks = chunker.process_full_curriculum(curriculum_file)
        
        # Análisis de resultados
        print(f"\n📊 ANÁLISIS DE CHUNKS INTELIGENTES")
        print("=" * 40)
        
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk['metadata']['type']
            if chunk_type not in chunk_types:
                chunk_types[chunk_type] = []
            chunk_types[chunk_type].append(chunk)
        
        for chunk_type, type_chunks in chunk_types.items():
            print(f"\n📋 {chunk_type}: {len(type_chunks)} chunks")
            
            # Mostrar ejemplo de chunk LLM vs estructural
            llm_chunks = [c for c in type_chunks if c['metadata'].get('llm_generated', False)]
            struct_chunks = [c for c in type_chunks if not c['metadata'].get('llm_generated', False)]
            
            print(f"   🤖 LLM: {len(llm_chunks)} chunks")
            print(f"   📋 Estructural: {len(struct_chunks)} chunks")
            
            if llm_chunks:
                example = llm_chunks[0]
                print(f"   💡 Ejemplo LLM: {example['content'][:100]}...")
        
        # Estadísticas finales
        stats = chunker.get_chunking_stats()
        print(f"\n📈 ESTADÍSTICAS FINALES:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n🎯 CONCLUSIÓN:")
        if stats['llm_enabled']:
            print("   ✅ Chunking híbrido LLM + estructural activo")
            print("   🧠 Calidad semántica mejorada")
        else:
            print("   ⚠️  Solo chunking estructural (instalar Ollama para LLM)")
        
    else:
        print(f"❌ Archivo no encontrado: {curriculum_file}")
        print("💡 Copia el archivo de curriculums para probar")