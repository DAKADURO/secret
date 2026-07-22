import os
import base64
import pdfplumber
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datos", "manuales")
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db_vectorial")
EMBEDDING_MODEL = "mxbai-embed-large"

def extract_pdf_content(pdf_path, llm=None, progress_callback=None, marca="General", modelo="General"):
    """
    Extrae texto, tablas en formato Markdown estricto e imágenes analizadas vía LLaVA.
    """
    pages_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                msg = f"Leyendo página {i+1}/{total_pages}..."
                print(f"  {msg}")
                if progress_callback:
                    progress_callback(msg, i+1, total_pages)

                page_text = page.extract_text() or ""
                tables_markdown = ""

                # 1. Extracción y Formateo Estructurado de Tablas (Markdown estricto)
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        if not table or len(table) < 1:
                            continue
                        
                        cleaned_headers = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in table[0]]
                        tables_markdown += "\n\n| " + " | ".join(cleaned_headers) + " |\n"
                        tables_markdown += "| " + " | ".join(["---"] * len(cleaned_headers)) + " |\n"
                        
                        for row in table[1:]:
                            cleaned_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                            tables_markdown += "| " + " | ".join(cleaned_row) + " |\n"

                # 2. Extracción de Imágenes y Análisis Visual (LLaVA)
                image_descriptions = ""
                image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datos", "imagenes", marca, modelo)
                os.makedirs(image_dir, exist_ok=True)

                try:
                    for img_idx, img in enumerate(page.images):
                        width = img.get('width', 0)
                        height = img.get('height', 0)
                        if width * height < 20000:
                            continue
                        
                        bbox = (img['x0'], img['top'], img['x1'], img['bottom'])
                        cropped = page.crop(bbox).to_image(resolution=150).original
                        img_filename = f"{os.path.basename(pdf_path)}_pag{i+1}_img{img_idx}.png"
                        img_filepath = os.path.join(image_dir, img_filename)

                        min_size = 200
                        if cropped.width >= min_size and cropped.height >= min_size:
                            if cropped.mode in ('RGBA', 'LA') or (cropped.mode == 'P' and 'transparency' in cropped.info):
                                bg = Image.new("RGB", cropped.size, (255, 255, 255))
                                bg.paste(cropped, mask=cropped.split()[3])
                                bg.save(img_filepath, "PNG")
                            else:
                                cropped.convert("RGB").save(img_filepath, "PNG")

                            with open(img_filepath, "rb") as image_file:
                                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

                            try:
                                if progress_callback:
                                    progress_callback(f"Analizando diagrama {img_filename} con LLaVA...", i+1, total_pages)
                                llava = ChatOllama(model="llava", temperature=0.0)
                                message = HumanMessage(
                                    content=[
                                        {
                                            "type": "text",
                                            "text": "Revisa esta imagen de un manual industrial. Si es un diagrama, plano o esquema, extrae los PNs, nombres de piezas y relaciones de ensamblaje. Si no es relevante, responde 'Imagen no relevante'."
                                        },
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}
                                    ]
                                )
                                vision_response = llava.invoke([message])
                                if "no relevante" not in vision_response.content.lower():
                                    image_descriptions += f"\n\n[IMAGEN: {marca}/{modelo}/{img_filename}]\nDescripción visual y relaciones espaciales: {vision_response.content}\n"
                            except Exception as llava_e:
                                print(f"  LLaVA no disponible para {img_filename}: {llava_e}")
                except Exception as img_e:
                    print(f"Error procesando imágenes en página {i+1}: {img_e}")

                # 3. Consolidación de Contenido de la Página
                full_page_text = f"--- PÁGINA {i+1} ---\n{page_text}\n{tables_markdown}\n{image_descriptions}"
                if full_page_text.strip():
                    pages_content.append({"text": full_page_text, "pagina": i+1})

    except Exception as e:
        print(f"Error leyendo {pdf_path}: {e}")

    return pages_content

def procesar_e_ingestar_pdf(pdf_path, marca="General", modelo="General"):
    """
    Divide el documento en chunks optimizados de 2000 caracteres y los guarda en ChromaDB.
    """
    print(f"Procesando PDF: {pdf_path} ({marca} - {modelo})...")
    paginas = extract_pdf_content(pdf_path, marca=marca, modelo=modelo)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        separators=["\n--- PÁGINA", "\n\n| ", "\n\n", "\n", ".", " "]
    )
    
    documents = []
    for p in paginas:
        chunks = splitter.split_text(p["text"])
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "marca": marca,
                    "modelo": modelo,
                    "source": os.path.basename(pdf_path),
                    "pagina": str(p["pagina"])
                }
            )
            documents.append(doc)
            
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma.from_documents(documents, embeddings, persist_directory=DB_DIR)
    print(f"✅ Ingesta completada. Se añadieron {len(documents)} chunks a la base vectorial.")
    return True, len(documents)

def ingestar_archivo_unico(pdf_path, marca, modelo, filename, progress_callback=None):
    print(f"Ingestando archivo individual: {filename} ({marca}/{modelo})")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        separators=["\n--- PÁGINA", "\n\n| ", "\n\n", "\n", ".", " "]
    )

    pages_data = extract_pdf_content(pdf_path, progress_callback=progress_callback, marca=marca, modelo=modelo)
    docs = []
    
    if not pages_data:
        return False, 0
        
    for page_data in pages_data:
        chunks = text_splitter.split_text(page_data["text"])
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "marca": marca,
                    "modelo": modelo,
                    "source": filename,
                    "pagina": str(page_data["pagina"])
                }
            ))
            
    if docs:
        Chroma.from_documents(docs, embeddings, persist_directory=DB_DIR)
        return True, len(docs)
    return False, 0

import json

def ingestar_json_unico(json_path, marca, modelo, filename, progress_callback=None):
    print(f"Ingestando archivo JSON: {filename} ({marca}/{modelo})")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    docs = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print("El archivo JSON debe contener una lista de fragmentos.")
            return False, 0
            
        for i, item in enumerate(data):
            if progress_callback:
                progress_callback("Cargando y procesando JSON...", i+1, len(data))
                
            content = item.get("content", "")
            if not content: continue
            
            docs.append(Document(
                page_content=content,
                metadata={
                    "marca": marca,
                    "modelo": modelo,
                    "source": filename,
                    "chunk_id": item.get("chunk_id", str(i)),
                    "category": item.get("category", "")
                }
            ))
            
        if docs:
            print(f"Se cargaron {len(docs)} fragmentos del JSON. Guardando en ChromaDB...")
            Chroma.from_documents(docs, embeddings, persist_directory=DB_DIR)
            return True, len(docs)
    except Exception as e:
        print(f"Error procesando JSON {filename}: {e}")
        
    return False, 0

def process_documents():
    print(f"Inicializando modelo de embeddings ({EMBEDDING_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        separators=["\n--- PÁGINA", "\n\n| ", "\n\n", "\n", ".", " "]
    )
    
    docs = []
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        return

    for marca in os.listdir(DOCS_DIR):
        marca_path = os.path.join(DOCS_DIR, marca)
        if not os.path.isdir(marca_path): continue
            
        for modelo in os.listdir(marca_path):
            modelo_path = os.path.join(marca_path, modelo)
            if not os.path.isdir(modelo_path): continue
                
            for filename in os.listdir(modelo_path):
                if filename.lower().endswith(".pdf"):
                    pdf_path = os.path.join(modelo_path, filename)
                    print(f"Procesando: [{marca} - {modelo}] {filename}")

                    pages_data = extract_pdf_content(pdf_path, progress_callback=None, marca=marca, modelo=modelo)

                    if not pages_data:
                        continue

                    for page_data in pages_data:
                        chunks = text_splitter.split_text(page_data["text"])
                        for chunk in chunks:
                            docs.append(Document(
                                page_content=chunk,
                                metadata={
                                    "marca": marca,
                                    "modelo": modelo,
                                    "source": filename,
                                    "pagina": str(page_data["pagina"])
                                }
                            ))

                elif filename.lower().endswith(".json"):
                    json_path = os.path.join(modelo_path, filename)
                    print(f"Procesando JSON: [{marca} - {modelo}] {filename}")
                    try:
                        with open(json_path, 'r', encoding='utf-8') as jf:
                            data = json.load(jf)
                    except Exception as e:
                        print(f"  Error leyendo {filename}: {e}")
                        continue

                    if not isinstance(data, list):
                        print(f"  {filename} no contiene una lista de fragmentos, se omite.")
                        continue

                    for i, item in enumerate(data):
                        content = item.get("content", "")
                        if not content:
                            continue
                        docs.append(Document(
                            page_content=content,
                            metadata={
                                "marca": marca,
                                "modelo": modelo,
                                "source": filename,
                                "chunk_id": item.get("chunk_id", str(i)),
                                "category": item.get("category", "")
                            }
                        ))

    if docs:
        print(f"Se generaron {len(docs)} fragmentos. Guardando en ChromaDB...")
        Chroma.from_documents(docs, embeddings, persist_directory=DB_DIR)
        print(f"¡Proceso completado con éxito! Base de datos guardada en: {DB_DIR}")

if __name__ == "__main__":
    process_documents()
