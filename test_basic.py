"""Test básico de BGE-M3 + Vector Store"""

import sys
sys.path.append('src')

from embeddings.bge_embeddings import BGEEmbeddings
from rag.vector_store import LocalVectorStore

def test_basic_system():
    print("🧪 PROBANDO SISTEMA BÁSICO")
    print("=" * 50)
    
    # 1. Test BGE-M3
    print("1️⃣ Inicializando BGE-M3...")
    embedder = BGEEmbeddings()
    
    # 2. Test Vector Store  
    print("\n2️⃣ Inicializando Vector Store...")
    vs = LocalVectorStore(collection_name="test_usc")
    vs.clear_collection()
    
    # 3. Datos de prueba
    test_texts = [
        "Systems Engineering Fee: $5.298.134 COP per semester",
        "Bioengineering Occupational Profile: Designer of medical technologies",
        "First Semester Systems: MATEMÁTICAS FUNDAMENTALES, INTRODUCCIÓN A LA INGENIERÍA"
    ]
    
    test_metadatas = [
        {"type": "fee", "program": "Systems Engineering", "amount": 5298134},
        {"type": "profile", "program": "Bioengineering"},
        {"type": "curriculum", "program": "Systems Engineering", "semester": 1}
    ]
    
    # 4. Crear embeddings
    print("\n3️⃣ Creando embeddings...")
    embeddings = embedder.embed_documents(test_texts)
    
    # 5. Almacenar en vector store
    print("\n4️⃣ Almacenando en vector store...")
    vs.add_documents(test_texts, test_metadatas, embeddings)
    
    # 6. Test búsqueda
    print("\n5️⃣ Probando búsqueda semántica...")
    query = "¿Cuánto cuesta ingeniería de sistemas?"
    query_emb = embedder.embed_query(query)
    results = vs.search(query_emb, n_results=3)
    
    print(f"\n🔍 Query: '{query}'")
    print(f"📊 Resultados encontrados: {len(results['documents'])}")
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'], results['metadatas'], results['distances']
    )):
        similarity = 1 - distance
        print(f"\n   {i+1}. Similitud: {similarity:.3f}")
        print(f"      Programa: {metadata.get('program', 'N/A')}")  
        print(f"      Tipo: {metadata.get('type', 'N/A')}")
        print(f"      Texto: {doc}")
    
    # 7. Estadísticas
    print(f"\n📊 Estadísticas del sistema:")
    stats = vs.get_stats()
    print(f"   Total documentos: {stats['total_documents']}")
    print(f"   Tipos: {stats['document_types']}")
    print(f"   Programas: {stats['programs']}")
    
    print(f"\n✅ ¡Sistema básico funcionando!")

if __name__ == "__main__":
    test_basic_system()