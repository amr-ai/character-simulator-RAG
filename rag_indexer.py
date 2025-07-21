import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def load_documents_for_character(character_folder):
    """تحميل جميع مستندات النصية من مجلد شخصية معينة"""
    documents = []
    if not os.path.exists(character_folder):
        print(f"❌ Folder not found: {character_folder}")
        return documents

    for filename in os.listdir(character_folder):
        if filename.endswith(".txt"):
            path = os.path.join(character_folder, filename)
            try:
                loader = TextLoader(path, encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                print(f"✅ Loaded {len(docs)} docs from {filename}")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {str(e)}")
    return documents

def build_index_for_character(character_name, base_data="data", base_index="indexes"):
    """بناء فهرس البحث لمجلد شخصية معينة"""
    folder_path = os.path.join(base_data, character_name)
    docs = load_documents_for_character(folder_path)
    if not docs:
        print(f"❌ No documents found for {character_name}")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="all-minilm")
    index = FAISS.from_documents(chunks, embedding=embeddings)

    # إنشاء مجلد indexes إذا لم يكن موجودًا
    os.makedirs(base_index, exist_ok=True)
    
    save_path = os.path.join(base_index, character_name)
    index.save_local(save_path)
    print(f"📦 Saved vector index for {character_name} at {save_path}")

def discover_character_folders(base_data="data"):
    """اكتشاف جميع مجلدات الشخصيات تلقائيًا"""
    if not os.path.exists(base_data):
        print(f"❌ Data folder not found: {base_data}")
        return []

    # الحصول على جميع المجلدات في data وتجاهل الملفات
    character_folders = [
        name for name in os.listdir(base_data) 
        if os.path.isdir(os.path.join(base_data, name))
    ]
    
    print(f"🔍 Discovered {len(character_folders)} character folders")
    return character_folders

if __name__ == "__main__":
    # اكتشاف المجلدات تلقائيًا بدلاً من القائمة الثابتة
    characters = discover_character_folders()
    
    if not characters:
        print("❌ No character folders found in 'data' directory")
    else:
        print("📂 Character folders to process:", ", ".join(characters))
        
        for name in characters:
            print(f"\n🔁 Indexing {name}...")
            build_index_for_character(name)
        
        print("\n🎉 All characters processed successfully!")