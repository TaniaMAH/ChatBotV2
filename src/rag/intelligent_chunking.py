import re
import json
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from llm_utils import generate_response  # ✅ nuevo import

class IntelligentCurriculumChunker:
    def __init__(self, use_llm: bool = True, max_workers: int = 4):
        self.use_llm = use_llm
        self.max_workers = max_workers
        self.fallback_to_structural = True
        self.structural_patterns = {
            'program_header': r'### ([^\n]+)',
            'fee': r'\*\*Costo Matricula:\*\* \$([0-9,\.]+) \(💰cop\)',
            'occupational_profile': r'\*\*Perfil Ocupacional:\*\*\s*(.*?)(?=\*\*Curriculo:|$)',
            'curriculum_start': r'\*\*Curriculo:\*\*',
            'semester': r'#### Semestre ([IVX\d]+)\s*\n((?:- .+\n)*)',
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

    def chunk_program_intelligently(self, program_text: str, program_name: str) -> List[Dict[str, Any]]:
        print(f"🧠 Analizando programa: {program_name}")

        structural_chunks = self._structural_chunking(program_text, program_name)
        self.stats['structural_chunks'] += len(structural_chunks)

        llm_chunks = []
        if self.use_llm:
            llm_chunks = self._llm_semantic_chunking(program_text, program_name)
            if llm_chunks:
                self.stats['llm_chunks'] += len(llm_chunks)
                print(f"   🤖 LLM generó: {len(llm_chunks)} chunks semánticos")

        if llm_chunks and structural_chunks:
            hybrid_chunks = self._merge_chunks_intelligently(llm_chunks, structural_chunks)
            self.stats['hybrid_chunks'] += len(hybrid_chunks)
            print(f"   🔄 Fusión híbrida: {len(hybrid_chunks)} chunks finales")
            return hybrid_chunks
        elif llm_chunks:
            print(f"   🤖 Solo LLM: {len(llm_chunks)} chunks")
            return llm_chunks
        else:
            print(f"   📋 Fallback estructural: {len(structural_chunks)} chunks")
            return structural_chunks

    def _llm_semantic_chunking(self, program_text: str, program_name: str) -> List[Dict[str, Any]]:
        print(f"   🤖 Generando chunks con modelo HuggingFace (Mistral)...")

        prompt = f"""
Eres un experto en análisis de documentos académicos. Analiza este programa universitario y divide el contenido en chunks semánticamente coherentes.

PROGRAMA: {program_name}

CONTENIDO:
{program_text}

INSTRUCCIONES:
1. Identifica automáticamente las secciones principales (costo, perfil ocupacional, curriculo, etc.)
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

        try:
            llm_output = generate_response(prompt, max_tokens=1024)
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                chunks_data = json.loads(json_match.group())
                processed_chunks = []
                for chunk in chunks_data.get('chunks', []):
                    processed_chunk = {
                        'content': chunk['content'],
                        'metadata': {
                            **chunk.get('metadata', {}),
                            'type': chunk['type'],
                            'llm_generated': True,
                            'confidence': 0.9,
                            'attempt': 1
                        }
                    }
                    processed_chunks.append(processed_chunk)
                print(f"   ✅ LLM (HF) generó: {len(processed_chunks)} chunks")
                return processed_chunks
            else:
                print("   ⚠️ No se encontró un JSON válido en la respuesta del modelo.")
                return []
        except Exception as e:
            print(f"   ❌ Error generando respuesta con Hugging Face: {e}")
            return []

    def _structural_chunking(self, program_text: str, program_name: str) -> List[Dict[str, Any]]:
        chunks = []

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

        fee_match = re.search(self.structural_patterns['fee'], program_text)
        if fee_match:
            fee_amount_str = fee_match.group(1).replace(',', '').replace('.', '')
            fee_amount = int(fee_amount_str) if fee_amount_str.isdigit() else 0
            chunks.append({
                'content': f"{program_name}\n\nCosto: ${fee_match.group(1)} COP por semestre\n\nInformación de matrícula.",
                'metadata': {
                    'type': 'fee',
                    'program_name': program_name,
                    'fee_amount': fee_amount,
                    'chunk_strategy': 'structural_pattern',
                    'confidence': 0.95
                }
            })

        profile_match = re.search(self.structural_patterns['occupational_profile'], program_text, re.DOTALL)
        if profile_match:
            chunks.append({
                'content': f"{program_name} - Perfil Ocupacional\n\n{profile_match.group(1).strip()}",
                'metadata': {
                    'type': 'occupational_profile',
                    'program_name': program_name,
                    'chunk_strategy': 'structural_pattern',
                    'confidence': 0.9
                }
            })

        if '**Curriculo:**' in program_text:
            curriculum_start = program_text.find('**Curriculo:**')
            curriculum_content = program_text[curriculum_start:curriculum_start + 500]
            chunks.append({
                'content': f"{program_name} - Plan de Estudios\n\n{curriculum_content}{'...' if len(program_text[curriculum_start:]) > 500 else ''}",
                'metadata': {
                    'type': 'curriculum_summary',
                    'program_name': program_name,
                    'chunk_strategy': 'structural_curriculum',
                    'confidence': 0.9
                }
            })

        semester_matches = re.findall(self.structural_patterns['semester'], program_text)
        for semester_num, semester_content in semester_matches:
            subjects = re.findall(self.structural_patterns['subject'], semester_content)
            if subjects:
                total_credits = sum(int(credits) for _, credits in subjects)
                semester_text = f"{program_name} - Semestre {semester_num}\n\nTotal materias: {len(subjects)}\nTotal créditos: {total_credits}\n\n"
                semester_text += "\n".join([f"- {s} ({c} créditos)" for s, c in subjects])
                chunks.append({
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
                })

        return chunks

    def _merge_chunks_intelligently(self, llm_chunks: List[Dict], structural_chunks: List[Dict]) -> List[Dict]:
        merged = llm_chunks[:]
        for struct_chunk in structural_chunks:
            struct_type = struct_chunk['metadata']['type']
            struct_program = struct_chunk['metadata']['program_name']
            llm_covers = any(
                c['metadata']['type'] == struct_type and c['metadata']['program_name'] == struct_program
                for c in llm_chunks
            )
            if not llm_covers:
                struct_chunk['metadata']['source'] = 'structural_fallback'
                merged.append(struct_chunk)
        return merged

    def process_full_curriculum_parallel(self, file_path: str) -> List[Dict[str, Any]]:
        start_time = datetime.now()
        print(f"🧠 PROCESANDO CON CHUNKING INTELIGENTE PARALIZADO")
        print(f"   📄 Archivo: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Error leyendo archivo: {e}")
            return []

        programs = self._extract_programs(content)
        print(f"   Total programas extraídos: {len(programs)}")

        all_chunks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_program = {
                executor.submit(self.chunk_program_intelligently, p[1], p[0]): p for p in programs
            }

            for future in as_completed(future_to_program):
                program_name = future_to_program[future][0]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                    print(f"   ✅ Procesado: {program_name} -> {len(chunks)} chunks")
                except Exception as e:
                    print(f"   ❌ Error procesando {program_name}: {e}")

        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()

        print(f"\n📊 CHUNKING INTELIGENTE COMPLETADO")
        print(f"   Programas procesados: {len(programs)}")
        print(f"   Chunks LLM: {self.stats['llm_chunks']}")
        print(f"   Chunks estructurales: {self.stats['structural_chunks']}")
        print(f"   Chunks híbridos: {self.stats['hybrid_chunks']}")
        print(f"   Total chunks: {len(all_chunks)}")
        print(f"   Tiempo total: {self.stats['processing_time']:.2f}s")

        return all_chunks

    def _extract_programs(self, content: str) -> List[Tuple[str, str]]:
        programs = []
        print("🔍 Analizando estructura del archivo...")

        tech_patterns = [
            r'## PROGRAMAS DE TECNOLOGIA\s*\n(.*?)(?=## ESTUDIOS DE PREGRADO|\Z)',
            r'## PROGRAMAS DE TECNOLOGIA(.*?)(?=## ESTUDIOS DE PREGRADO|\Z)'
        ]

        undergrad_patterns = [
            r'## ESTUDIOS DE PREGRADO\s*\n(.*?)$',
            r'## ESTUDIOS DE PREGRADO(.*?)$',
            r'## ESTUDIOS DE PREGRADO\s*\n(.*?)(?=\Z)',
            r'##\s*ESTUDIOS\s*DE\s*PREGRADO\s*\n(.*?)$'
        ]

        tech_match = None
        for pattern in tech_patterns:
            m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if m:
                tech_match = m.group(1).strip()
                print("   ✔ Sección PROGRAMAS DE TECNOLOGIA encontrada")
                break

        undergrad_match = None
        for pattern in undergrad_patterns:
            m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if m:
                undergrad_match = m.group(1).strip()
                print("   ✔ Sección ESTUDIOS DE PREGRADO encontrada")
                break

        if tech_match:
            programs.extend(self._extract_programs_from_section(tech_match))
        if undergrad_match:
            programs.extend(self._extract_programs_from_section(undergrad_match))

        print(f"   Total programas extraídos: {len(programs)}")
        return programs

    def _extract_programs_from_section(self, section_text: str) -> List[Tuple[str, str]]:
        programs = []
        program_headers = list(re.finditer(self.structural_patterns['program_header'], section_text))
        if not program_headers:
            print("⚠️ No se encontraron programas individuales en la sección.")
            return []

        for i, match in enumerate(program_headers):
            start = match.end()
            program_name = match.group(1).strip()
            if i + 1 < len(program_headers):
                end = program_headers[i + 1].start()
                program_content = section_text[start:end].strip()
            else:
                program_content = section_text[start:].strip()

            full_program_text = match.group(0) + "\n" + program_content
            programs.append((program_name, full_program_text))

        return programs