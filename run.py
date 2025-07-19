import sys
from pathlib import Path

sys.path.append('src')

def run_chunker():
    print("\n🧠 Ejecutando chunker inteligente con procesamiento paralelo...\n")
    try:
        from src.rag.intelligent_chunking import IntelligentCurriculumChunker
    except ImportError as e:
        print(f"❌ No se pudo importar IntelligentCurriculumChunker: {e}")
        return

    data_dir = Path("data/documentos")
    input_file = data_dir / "Curriculums_Technology_Undergraduate.md"
    if not input_file.exists():
        print(f"⚠️ Archivo no encontrado: {input_file}")
        print("💡 Por favor, coloca el archivo y vuelve a ejecutar")
        return

    chunker = IntelligentCurriculumChunker(use_llm=True, max_workers=4)
    chunks = chunker.process_full_curriculum_parallel(str(input_file))

    print(f"\n✅ Chunking completado con {len(chunks)} chunks generados.")

def main():
    print("🚀 USC CHATBOT HÍBRIDO")
    print("=" * 30)
    
    try:
        print("🔍 Verificando dependencias...")
        from src.embeddings.bge_embeddings import BGEEmbeddings
        from src.rag.vector_store import LocalVectorStore
        print("✅ Dependencias OK")
        
        print("\n📁 Verificando estructura...")
        data_dir = Path("data/documentos")
        if not data_dir.exists():
            print(f"⚠️  Creando directorio: {data_dir}")
            data_dir.mkdir(parents=True)
        print("✅ Estructura OK")
        
        print("\n🎯 SISTEMA LISTO PARA USAR")
        print("=" * 30)
        print("Siguiente paso:")
        print("1. 📄 Coloca tu archivo 'Curriculums_Technology_Undergraduate.md' en data/documentos/")
        print("2. 🧪 Ejecuta: python test_basic.py")
        print("3. 🚀 Si todo funciona, crearemos el chunking inteligente\n")
        
        # Aquí llamamos la función que ejecuta el chunker
        run_chunker()
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("💡 Solución: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()