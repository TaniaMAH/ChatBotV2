"""
Procesador USC - LÓGICA EXACTA QUE FUNCIONA
Basado en el análisis manual exitoso
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

class USCCurriculumProcessor:
    def __init__(self):
        self.stats = {
            'programs_processed': 0,
            'chunks_created': 0,
            'technology_programs': 0,
            'undergraduate_programs': 0,
            'semester_chunks': 0,
            'processing_time': None
        }
        
        print("🧠 Procesador USC - LÓGICA EXACTA QUE FUNCIONA")
    
    def process_curriculum_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Usa la MISMA lógica que el análisis manual exitoso"""
        
        start_time = datetime.now()
        print(f"📄 Procesando: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"✅ Archivo leído: {len(content)} caracteres")
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
        
        lines = content.split('\n')
        
        # PASO 1: Encontrar límites de secciones (IGUAL que análisis)
        tech_start = None
        undergrad_start = None
        
        for i, line in enumerate(lines):
            if line.strip() == "## PROGRAMAS DE TECNOLOGIA":
                tech_start = i
            elif line.strip() == "## ESTUDIOS DE PREGRADO":
                undergrad_start = i
        
        print(f"✅ Tecnología: línea {tech_start}")
        print(f"✅ Pregrado: línea {undergrad_start}")
        
        all_chunks = []
        
        # PASO 2: Extraer programas de tecnología (IGUAL que análisis manual)
        if tech_start is not None and undergrad_start is not None:
            print(f"🔍 Buscando en tecnología: líneas {tech_start} a {undergrad_start}")
            
            tech_programs = []
            # Buscar EXACTAMENTE como en el análisis manual
            for i in range(tech_start, undergrad_start):
                line = lines[i].strip()
                if line.startswith('### ') and not any(word in line.lower() for word in ['semestre', 'semester']):
                    program_name = line.replace('###', '').strip()
                    
                    # Extraer contenido del programa
                    program_lines = [line]  # Empezar con la línea del título
                    
                    # Buscar hasta el siguiente programa o fin de sección
                    for j in range(i + 1, undergrad_start):
                        next_line = lines[j].strip()
                        if next_line.startswith('### ') and not any(word in next_line.lower() for word in ['semestre', 'semester']):
                            break  # Encontró el siguiente programa
                        program_lines.append(lines[j])
                    
                    program_content = '\n'.join(program_lines)
                    
                    # Validar y parsear
                    if self._is_valid_program(program_content):
                        program_data = self._parse_program(program_name, program_content, 'technology')
                        if program_data:
                            tech_programs.append(program_data)
                            print(f"   📚 {program_name} ({len(program_data.get('semesters', []))} semestres) - technology")
            
            # Crear chunks para programas de tecnología
            for program in tech_programs:
                chunks = self._create_all_chunks(program, 'technology')
                all_chunks.extend(chunks)
                self.stats['technology_programs'] += 1
                self.stats['programs_processed'] += 1
        
        # PASO 3: Extraer programas de pregrado
        if undergrad_start is not None:
            print(f"🔍 Buscando en pregrado: línea {undergrad_start} en adelante")
            
            undergrad_programs = []
            for i in range(undergrad_start, len(lines)):
                line = lines[i].strip()
                if line.startswith('### ') and not any(word in line.lower() for word in ['semestre', 'semester']):
                    program_name = line.replace('###', '').strip()
                    
                    # Extraer contenido del programa
                    program_lines = [line]
                    
                    # Buscar hasta el siguiente programa o fin
                    for j in range(i + 1, len(lines)):
                        if j < len(lines):
                            next_line = lines[j].strip()
                            if next_line.startswith('### ') and not any(word in next_line.lower() for word in ['semestre', 'semester']):
                                break
                            program_lines.append(lines[j])
                    
                    program_content = '\n'.join(program_lines)
                    
                    # Validar y parsear
                    if self._is_valid_program(program_content):
                        program_data = self._parse_program(program_name, program_content, 'undergraduate')
                        if program_data:
                            undergrad_programs.append(program_data)
                            print(f"   📚 {program_name} ({len(program_data.get('semesters', []))} semestres) - undergraduate")
            
            # Crear chunks para programas de pregrado
            for program in undergrad_programs:
                chunks = self._create_all_chunks(program, 'undergraduate')
                all_chunks.extend(chunks)
                self.stats['undergraduate_programs'] += 1
                self.stats['programs_processed'] += 1
        
        self.stats['chunks_created'] = len(all_chunks)
        self.stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        print(f"\n📊 Procesamiento completado:")
        print(f"   Programas de Tecnología: {self.stats['technology_programs']}")
        print(f"   Programas de Pregrado: {self.stats['undergraduate_programs']}")
        print(f"   Total programas: {self.stats['programs_processed']}")
        print(f"   Chunks de semestres: {self.stats['semester_chunks']}")
        print(f"   Total chunks: {self.stats['chunks_created']}")
        print(f"   Tiempo: {self.stats['processing_time']:.2f}s")
        
        return all_chunks
    
    def _is_valid_program(self, content: str) -> bool:
        """Valida que sea un programa real"""
        if len(content) < 200:
            return False
        
        required = ['Costo Matricula', 'Perfil Ocupacional', 'Curriculo']
        return all(req in content for req in required)
    
    def _parse_program(self, name: str, content: str, section_type: str) -> Optional[Dict[str, Any]]:
        """Parsea programa individual"""
        
        # Extraer costo
        cost_amount = 0
        cost_raw = ""
        cost_match = re.search(r'\*\*Costo Matricula:\*\* \$([0-9,\.]+) \(💰cop\)', content)
        if cost_match:
            cost_raw = cost_match.group(1)
            cost_str = cost_raw.replace(',', '').replace('.', '')
            cost_amount = int(cost_str) if cost_str.isdigit() else 0
        
        # Extraer perfil
        profile = ""
        profile_match = re.search(r'\*\*Perfil Ocupacional:\*\*\s*(.*?)(?=\*\*Curriculo:|$)', content, re.DOTALL)
        if profile_match:
            profile = profile_match.group(1).strip()
        
        # Extraer semestres
        semester_matches = re.findall(r'#### Semestre ([IVX\d]+)\s*\n((?:- [^\n]+\n)*)', content)
        
        semesters = []
        for sem_num, sem_content in semester_matches:
            subjects = re.findall(r'- ([^|]+) \| (\d+) Créditos', sem_content)
            if subjects:
                semester_data = {
                    'number': sem_num,
                    'subjects': [{'name': name.strip(), 'credits': int(credits)} for name, credits in subjects],
                    'total_credits': sum(int(credits) for _, credits in subjects),
                    'subject_count': len(subjects)
                }
                semesters.append(semester_data)
        
        return {
            'name': name,
            'content': content,
            'type': section_type,
            'cost_amount': cost_amount,
            'cost_raw': cost_raw,
            'profile': profile,
            'semesters': semesters,
            'total_semesters': len(semesters)
        }
    
    def _create_all_chunks(self, program: Dict[str, Any], section_type: str) -> List[Dict[str, Any]]:
        """Crea chunks para un programa"""
        
        chunks = []
        program_name = program['name']
        
        # Chunk completo
        chunks.append({
            'content': program['content'],
            'metadata': {
                'type': 'program_complete',
                'program_name': program_name,
                'program_type': section_type,
                'fee_amount': program['cost_amount'],
                'total_semesters': program['total_semesters']
            }
        })
        
        # Chunk costo
        if program['cost_amount'] > 0:
            chunks.append({
                'content': f"{program_name}\nCosto: ${program['cost_raw']} COP\nPrograma de {section_type} USC.",
                'metadata': {
                    'type': 'fee',
                    'program_name': program_name,
                    'program_type': section_type,
                    'fee_amount': program['cost_amount']
                }
            })
        
        # Chunk perfil
        if program['profile']:
            chunks.append({
                'content': f"{program_name} - Perfil Ocupacional\n{program['profile']}",
                'metadata': {
                    'type': 'occupational_profile',
                    'program_name': program_name,
                    'program_type': section_type
                }
            })
        
        # Chunk curricular
        chunks.append({
            'content': f"{program_name} - Plan de Estudios\nDuración: {program['total_semesters']} semestres",
            'metadata': {
                'type': 'curriculum_summary',
                'program_name': program_name,
                'program_type': section_type,
                'total_semesters': program['total_semesters']
            }
        })
        
        # Chunks semestres
        for semester in program['semesters']:
            semester_content = f"{program_name} - Semestre {semester['number']}\n"
            semester_content += f"Total: {semester['subject_count']} materias, {semester['total_credits']} créditos\n"
            semester_content += "Materias:\n"
            for subject in semester['subjects']:
                semester_content += f"- {subject['name']} | {subject['credits']} créditos\n"
            
            chunks.append({
                'content': semester_content,
                'metadata': {
                    'type': 'curriculum_semester',
                    'program_name': program_name,
                    'program_type': section_type,
                    'semester_number': semester['number'],
                    'subject_count': semester['subject_count'],
                    'total_credits': semester['total_credits']
                }
            })
            
            self.stats['semester_chunks'] += 1
        
        print(f"      ✅ {len(chunks)} chunks creados para {program_name}")
        return chunks
    
    def get_processing_stats(self) -> Dict[str, Any]:
        return self.stats.copy()
