"""
Script principal para ejecutar el sistema USC Chatbot
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.append('src')

def main():
    print("🚀 USC CHATBOT HÍBRIDO")
    print("=" * 30)
    
    try:
        # Verificar dependencias críticas
        print("🔍 Verificando dependencias...")
        
        from src.embeddings.bge_embeddings import BGEEmbeddings
        from src.rag.vector_store import LocalVectorStore
        
        print("✅ Dependencias OK")
        
        # Verificar estructura de archivos
        print("\n📁 Verificando estructura...")
        
        data_dir = Path("data/documentos")
        if not data_dir.exists():
            print(f"⚠️  Creando directorio: {data_dir}")
            data_dir.mkdir(parents=True)
        
        print("✅ Estructura OK")
        
        # Mensaje de siguiente paso
        print("\n🎯 SISTEMA LISTO PARA USAR")
        print("=" * 30)
        print("Siguiente paso:")
        print("1. 📄 Coloca tu archivo 'Curriculums_Technology_Undergraduate.md' en data/documentos/")
        print("2. 🧪 Ejecuta: python test_basic.py")
        print("3. 🚀 Si todo funciona, crearemos el chunking inteligente")
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("💡 Solución: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()