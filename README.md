# RAG-corto
Este repositorio contiene un código diseñado para implementar un sistema de RAG (Retrieval-Augmented Generation) utilizando herramientas de la biblioteca LangChain.  El sistema combina capacidades de recuperación de información con generación de texto para responder a consultas, apoyándose en un documento específico como base de conocimiento.

    import subprocess
    import os

#Configurar el entorno e instalar paquetes

    try:
        subprocess.run(['pip', 'install', 'langchain', 'langchain_community', 'langchain-openai',
                        'scikit-learn', 'langchain-ollama', 'pymupdf', 'langchain_huggingface',
                        'faiss-gpu'], check=True)
        print("Paquetes instalados correctamente.")
    except subprocess.CalledProcessError as e:
        print("Error al instalar paquetes:", e)
    
    from langchain.chains import RetrievalQA
    
    from langchain_community.document_loaders import  PyMuPDFLoader
    
    from langchain_community.llms import Ollama
    
    from langchain_community.vectorstores import FAISS
    
    from langchain_huggingface import HuggingFaceEmbeddings
    
    from langchain_text_splitters import CharacterTextSplitter
    
    loader =  PyMuPDFLoader('/content/drive/MyDrive/Colab Notebooks/CIDE/Introducción a la Estadística.pdf')
    
    docs = loader.load()

#Instala y prepara el modelo localmente.

    !curl -fsSL https://ollama.com/install.sh | sh
    
    !pip install colab-xterm
    
    %load_ext colabxterm
    
    %xterm
    
    ##ollama serve
    
    !ollama pull llama3 

#Se divide el contenido del documento en fragmentos manejables utilizando un separador y configurando el tamaño de los fragmentos

    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=200
    )
# Valida si la variable docs está vacío
    if not docs:
        raise ValueError("La lista de documentos 'docs' está vacía. Proporcione documentos válidos.")
    
    
    texts = text_splitter.split_documents(docs)

# Crear los embeddings
    embeddings = HuggingFaceEmbeddings()

# Ruta para guardar el índice FAISS
    index_directory = "vector_store"
    index_path = os.path.join(index_directory, "faiss_index")

# Asegurar que el directorio exista
    ensure_directory_exists(index_directory)

# Cargar o crear el índice FAISS
    if os.path.exists(index_path):
        print(f"Cargando índice FAISS desde {index_path}...")
        db = FAISS.load_local(index_path, embeddings)
    else:
        print("Creando un nuevo índice FAISS...")
        db = FAISS.from_documents(texts, embeddings)
        print(f"Guardando índice FAISS en {index_path}...")
        db.save_local(index_path)

# Configurar el modelo LLM (Ollama Llama3)
    llm = Ollama(model="llama3")

# Crear la cadena de preguntas y respuestas
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever()
    )

# Hacer una consulta
    question = "¿Que es un estadístico?"

# Validación de entrada para la pregunta
    if not question.strip():
        raise ValueError("La pregunta esta vacia. Proporcione una consulta valida.")
    
    print(f"Haciendo la consulta: {question}")
    result = chain.invoke({"query": question})

# Mostrar el resultado
    print("Resultado:")
    print(result['result'])
