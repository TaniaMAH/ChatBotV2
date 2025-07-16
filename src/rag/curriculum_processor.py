"""
Procesador inteligente para currículums USC
Chunking específico para programas académicos, costos y perfiles ocupacionales
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

class USCCurriculumProcessor:
    def __init__(self):
        """
        Inicializa el procesador de currículums USC
        """
        
        self.program_patterns = {
            'technology': r'### (.+Technology)\s*\n',
            'undergraduate': r'### (.+Engineering|Commercial Engineering|Energy Engineering)\s*\n'
        }
        
        self.section_patterns = {
            'fee': r'\*\*Fee:\*\* \$([0-9,\.]+) \(💰cop\)',
            'occupational_profile': r'\*\*Occupational Profile:\*\*\s*(.*?)(?=\*\*Curriculum:)',
            'curriculum_header': r'\*\*Curriculum:\*\*',
            'semester': r'#### Semester ([IVX\d]+)\s*\n((?:- .+\n)*)',
        }
        
        self.stats = {
            'programs_processed': 0,
            'chunks_created': 0,
            'processing_time': None
        }
        
        print("🧠 Procesador de Currículums USC inicializado")
        print("   📋 Patrones configurados para Technology & Undergraduate")
    
    def process_curriculum_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Procesa archivo de currículums y crea chunks inteligentes
        
        Args:
            file_path: Ruta al archivo .md de currículums
            
        Returns:
            Lista de chunks con metadata rica
        """
        
        start_time = datetime.now()
        
        print(f"📄 Procesando archivo: {file_path}")
        
        # Leer archivo
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"✅ Archivo leído: {len(content)} caracteres")
        except Exception as e:
            print(f"❌ Error leyendo archivo: {e}")
            return []
        
        # Dividir en secciones principales
        sections = self._split_by_main_sections(content)
        
        all_chunks = []
        
        # Procesar cada sección
        for section_type, section_content in sections.items():
            print(f"\n🔄 Procesando sección: {section_type}")
            
            programs = self._extract_programs(section_content, section_type)
            
            for program in programs:
                program_chunks = self._create_program_chunks(program, section_type)
                all_chunks.extend(program_chunks)
                self.stats['programs_processed'] += 1
        
        # Actualizar estadísticas
        end_time = datetime.now()
        self.stats['chunks_created'] = len(all_chunks)
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        
        print(f"\n📊 PROCESAMIENTO COMPLETADO")
        print(f"   Programas procesados: {self.stats['programs_processed']}")
        print(f"   Chunks creados: {self.stats['chunks_created']}")
        print(f"   Tiempo: {self.stats['processing_time']:.2f}s")
        
        return all_chunks
    
    def _split_by_main_sections(self, content: str) -> Dict[str, str]:
        """Divide contenido en secciones principales"""
        
        # Encontrar secciones principales
        technology_match = re.search(r'## TECHNOLOGY PROGRAMS\s*\n(.*?)(?=## UNDERGRADUATE STUDIES|\Z)', 
                                   content, re.DOTALL)
        undergraduate_match = re.search(r'## UNDERGRADUATE STUDIES\s*\n(.*?)(?=## |\Z)', 
                                       content, re.DOTALL)
        
        sections = {}
        
        if technology_match:
            sections['technology'] = technology_match.group(1)
            print(f"✅ Sección Technology encontrada: {len(sections['technology'])} chars")
        
        if undergraduate_match:
            sections['undergraduate'] = undergraduate_match.group(1)
            print(f"✅ Sección Undergraduate encontrada: {len(sections['undergraduate'])} chars")
        
        return sections
    
    def _extract_programs(self, section_content: str, section_type: str) -> List[Dict[str, Any]]:
        """Extrae programas individuales de una sección"""
        
        programs = []
        
        # Encontrar programas usando pattern ### Program Name
        program_pattern = r'### ([^\n]+)\n(.*?)(?=### |\Z)'
        program_matches = re.findall(program_pattern, section_content, re.DOTALL)
        
        for program_name, program_content in program_matches:
            program_data = {
                'name': program_name.strip(),
                'content': program_content.strip(),
                'type': section_type,
                'raw_data': f"### {program_name}\n{program_content}"
            }
            
            # Extraer información específica
            program_data.update(self._extract_program_details(program_content))
            
            programs.append(program_data)
            print(f"   📚 Programa extraído: {program_name}")
        
        return programs
    
    def _extract_program_details(self, content: str) -> Dict[str, Any]:
        """Extrae detalles específicos de un programa"""
        
        details = {}
        
        # Extraer fee
        fee_match = re.search(self.section_patterns['fee'], content)
        if fee_match:
            fee_str = fee_match.group(1).replace(',', '').replace('.', '')
            details['fee_raw'] = fee_match.group(1)
            details['fee_amount'] = int(fee_str) if fee_str.isdigit() else 0
        
        # Extraer perfil ocupacional
        profile_match = re.search(self.section_patterns['occupational_profile'], 
                                content, re.DOTALL)
        if profile_match:
            details['occupational_profile'] = profile_match.group(1).strip()
        
        # Extraer semestres
        semester_matches = re.findall(self.section_patterns['semester'], content)
        details['semesters'] = []
        details['total_semesters'] = len(semester_matches)
        
        for semester_num, semester_content in semester_matches:
            # Extraer materias del semestre
            subjects = re.findall(r'- ([^|]+) \| (\d+) Créditos', semester_content)
            total_credits = sum(int(credits) for _, credits in subjects)
            
            semester_data = {
                'number': semester_num,
                'subjects': [{'name': name.strip(), 'credits': int(credits)} 
                           for name, credits in subjects],
                'total_credits': total_credits,
                'subject_count': len(subjects)
            }
            details['semesters'].append(semester_data)
        
        return details
    
    def _create_program_chunks(self, program: Dict[str, Any], section_type: str) -> List[Dict[str, Any]]:
        """Crea chunks inteligentes para un programa específico"""
        
        chunks = []
        program_name = program['name']
        
        # CHUNK 1: Información completa del programa
        complete_chunk = {
            'content': program['raw_data'],
            'metadata': {
                'type': 'program_complete',
                'program_name': program_name,
                'program_type': section_type,
                'level': 'complete',
                'fee_amount': program.get('fee_amount', 0),
                'total_semesters': program.get('total_semesters', 0),
                'chunk_strategy': 'complete_program_info'
            }
        }
        chunks.append(complete_chunk)
        
        # CHUNK 2: Fee específico
        if 'fee_amount' in program:
            fee_content = f"{program_name}\n\nFee: ${program['fee_raw']} COP\n\nCosto por semestre para el programa {program_name}."
            
            fee_chunk = {
                'content': fee_content,
                'metadata': {
                    'type': 'fee',
                    'program_name': program_name,
                    'program_type': section_type,
                    'level': 'specific',
                    'fee_amount': program['fee_amount'],
                    'currency': 'COP',
                    'chunk_strategy': 'fee_focused'
                }
            }
            chunks.append(fee_chunk)
        
        # CHUNK 3: Perfil ocupacional
        if 'occupational_profile' in program:
            profile_content = f"{program_name} - Perfil Ocupacional\n\n{program['occupational_profile']}\n\nCampo laboral y competencias profesionales para egresados de {program_name}."
            
            profile_chunk = {
                'content': profile_content,
                'metadata': {
                    'type': 'occupational_profile',
                    'program_name': program_name,
                    'program_type': section_type,
                    'level': 'specific',
                    'word_count': len(program['occupational_profile'].split()),
                    'chunk_strategy': 'occupational_focus'
                }
            }
            chunks.append(profile_chunk)
        
        # CHUNK 4: Curriculum completo (para búsquedas sobre plan de estudios)
        if 'semesters' in program and program['semesters']:
            curriculum_summary = self._create_curriculum_summary(program)
            
            curriculum_chunk = {
                'content': curriculum_summary,
                'metadata': {
                    'type': 'curriculum_summary',
                    'program_name': program_name,
                    'program_type': section_type,
                    'level': 'summary',
                    'total_semesters': len(program['semesters']),
                    'chunk_strategy': 'curriculum_overview'
                }
            }
            chunks.append(curriculum_chunk)
        
        # CHUNKS 5+: Semestres individuales
        for semester_data in program.get('semesters', []):
            semester_content = self._create_semester_content(program_name, semester_data)
            
            semester_chunk = {
                'content': semester_content,
                'metadata': {
                    'type': 'curriculum_semester',
                    'program_name': program_name,
                    'program_type': section_type,
                    'level': 'detail',
                    'semester_number': semester_data['number'],
                    'subject_count': semester_data['subject_count'],
                    'total_credits': semester_data['total_credits'],
                    'chunk_strategy': 'semester_detail'
                }
            }
            chunks.append(semester_chunk)
        
        print(f"      ✅ {len(chunks)} chunks creados para {program_name}")
        return chunks
    
    def _create_curriculum_summary(self, program: Dict[str, Any]) -> str:
        """Crea resumen del curriculum completo"""
        
        program_name = program['name']
        semesters = program['semesters']
        
        summary_parts = [
            f"{program_name} - Plan de Estudios Completo",
            f"\nDuración: {len(semesters)} semestres",
            "\nResumen por semestre:"
        ]
        
        for semester in semesters:
            semester_summary = f"\nSemestre {semester['number']}: {semester['subject_count']} materias, {semester['total_credits']} créditos"
            summary_parts.append(semester_summary)
            
            # Incluir algunas materias principales
            main_subjects = semester['subjects'][:3]  # Primeras 3 materias
            subject_names = [subj['name'] for subj in main_subjects]
            summary_parts.append(f"  Materias principales: {', '.join(subject_names)}")
        
        return '\n'.join(summary_parts)
    
    def _create_semester_content(self, program_name: str, semester_data: Dict[str, Any]) -> str:
        """Crea contenido detallado de un semestre"""
        
        content_parts = [
            f"{program_name} - Semestre {semester_data['number']}",
            f"\nTotal de materias: {semester_data['subject_count']}",
            f"Total de créditos: {semester_data['total_credits']}",
            "\nMaterias:"
        ]
        
        for subject in semester_data['subjects']:
            subject_line = f"- {subject['name']} ({subject['credits']} créditos)"
            content_parts.append(subject_line)
        
        return '\n'.join(content_parts)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del procesamiento"""
        return self.stats.copy()
    
    def analyze_curriculum_structure(self, file_path: str) -> Dict[str, Any]:
        """Analiza estructura del archivo sin procesar completamente"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'file_size': len(content),
                'programs_found': len(re.findall(r'### ([^\n]+)', content)),
                'technology_programs': len(re.findall(r'### (.+Technology)', content)),
                'undergraduate_programs': len(re.findall(r'### (.+Engineering)', content)),
                'total_semesters': len(re.findall(r'#### Semester', content)),
                'has_fees': bool(re.search(r'\*\*Fee:\*\*', content)),
                'has_profiles': bool(re.search(r'\*\*Occupational Profile:\*\*', content))
            }
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}

# Test del procesador
if __name__ == "__main__":
    print("🧪 PROBANDO PROCESADOR DE CURRÍCULUMS")
    print("=" * 50)
    
    processor = USCCurriculumProcessor()
    
    # Verificar si existe el archivo
    curriculum_file = "data/documentos/Curriculums_Technology_Undergraduate.md"
    
    if Path(curriculum_file).exists():
        print(f"✅ Archivo encontrado: {curriculum_file}")
        
        # Análisis rápido de estructura
        analysis = processor.analyze_curriculum_structure(curriculum_file)
        print(f"\n📊 Análisis de estructura:")
        for key, value in analysis.items():
            print(f"   {key}: {value}")
        
        # Procesar archivo completo
        print(f"\n🔄 Procesando archivo completo...")
        chunks = processor.process_curriculum_file(curriculum_file)
        
        # Mostrar estadísticas
        stats = processor.get_processing_stats()
        print(f"\n📈 Estadísticas finales:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Mostrar ejemplos de chunks
        print(f"\n💡 Ejemplos de chunks creados:")
        
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk['metadata']['type']
            if chunk_type not in chunk_types:
                chunk_types[chunk_type] = []
            chunk_types[chunk_type].append(chunk)
        
        for chunk_type, type_chunks in chunk_types.items():
            print(f"\n📋 Tipo: {chunk_type} ({len(type_chunks)} chunks)")
            if type_chunks:
                example = type_chunks[0]
                print(f"   Programa: {example['metadata']['program_name']}")
                print(f"   Contenido: {example['content'][:100]}...")
    
    else:
        print(f"❌ Archivo no encontrado: {curriculum_file}")
        print(f"💡 Por favor copia tu archivo .md a esa ubicación")