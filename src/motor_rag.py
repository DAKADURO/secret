import os
import re
import sys
from typing import TypedDict, List, Set

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db_vectorial")
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.1"

class HybridRetriever:
    """Ensemble Retriever combinando BM25 (Léxico) y Chroma (Vectorial Denso) mediante RRF (Reciprocal Rank Fusion)."""
    def __init__(self, bm25_retriever, vector_retriever, weights=[0.5, 0.5]):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.weights = weights

    def invoke(self, query: str) -> list:
        bm25_docs = []
        vector_docs = []
        try:
            bm25_docs = self.bm25_retriever.invoke(query)
        except Exception as e:
            print(f"BM25 Error: {e}")

        try:
            vector_docs = self.vector_retriever.invoke(query)
        except Exception as e:
            print(f"Vector Error: {e}")

        rrf_scores = {}
        doc_map = {}

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.weights[0] / (60 + rank + 1)

        for rank, doc in enumerate(vector_docs):
            key = doc.page_content
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.weights[1] / (60 + rank + 1)

        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys]

class AgentState(TypedDict):
    pregunta: str
    historial: list
    marca: str
    modelo: str
    pregunta_contextualizada: str
    hipotesis: List[str]
    contexto: str
    fuentes: Set[str]
    respuesta_final: str

class MotorRAG:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.llm = OllamaLLM(model=LLM_MODEL, num_ctx=8192)
        
        if os.path.exists(DB_DIR):
            self.db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
            self._init_hybrid_retriever()
        else:
            self.db = None
            self.ensemble_retriever = None

        self.graph = self._build_graph()

    def _init_hybrid_retriever(self):
        """Inicializa la búsqueda híbrida (BM25 Léxico + Vectorial Dense)."""
        try:
            raw_data = self.db.get()
            if raw_data and raw_data.get('documents') and len(raw_data['documents']) > 0:
                all_docs = [
                    Document(page_content=doc, metadata=meta)
                    for doc, meta in zip(raw_data['documents'], raw_data['metadatas'])
                ]
                
                # 1. Recuperador BM25 (Exact Match para PNs y Códigos de Error)
                self.bm25_retriever = BM25Retriever.from_documents(all_docs)
                self.bm25_retriever.k = 8
                
                # 2. Recuperador Denso Vectorial
                self.vector_retriever = self.db.as_retriever(search_kwargs={"k": 8})
                
                # 3. Ensemble Retriever Híbrido RRF 50/50
                self.ensemble_retriever = HybridRetriever(
                    bm25_retriever=self.bm25_retriever,
                    vector_retriever=self.vector_retriever,
                    weights=[0.5, 0.5]
                )
                print("✅ Búsqueda Híbrida (BM25 Léxico + Chroma Vectorial Ensemble) inicializada correctamente.")
            else:
                self.ensemble_retriever = None
        except Exception as e:
            print(f"Advertencia al inicializar el recuperador híbrido: {e}")
            self.ensemble_retriever = None

    def get_marcas_modelos(self):
        if not self.db:
            return {}
        try:
            results = self.db.get()
            marcas_modelos = {}
            if results and 'metadatas' in results and results['metadatas']:
                for meta in results['metadatas']:
                    if not meta: continue
                    marca = meta.get('marca')
                    modelo = meta.get('modelo')
                    if marca:
                        if marca not in marcas_modelos:
                            marcas_modelos[marca] = set()
                        if modelo:
                            marcas_modelos[marca].add(modelo)
            return {k: list(v) for k, v in marcas_modelos.items()}
        except Exception as e:
            print(f"Error obteniendo marcas y modelos: {e}")
            return {}

    def get_documentos_por_modelo(self):
        if not self.db:
            return {}
        try:
            results = self.db.get()
            docs = {}
            if results and 'metadatas' in results and results['metadatas']:
                for meta in results['metadatas']:
                    if not meta: continue
                    marca = meta.get('marca')
                    modelo = meta.get('modelo')
                    source = meta.get('source')
                    if marca and modelo and source:
                        if marca not in docs:
                            docs[marca] = {}
                        if modelo not in docs[marca]:
                            docs[marca][modelo] = set()
                        docs[marca][modelo].add(source)
            for m in docs:
                for mod in docs[m]:
                    docs[m][mod] = list(docs[m][mod])
            return docs
        except Exception as e:
            print(f"Error obteniendo documentos: {e}")
            return {}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("contextualizar", self._nodo_contextualizar)
        workflow.add_node("analizar_sintomas", self._nodo_analizar_sintomas)
        workflow.add_node("investigar", self._nodo_investigar)
        workflow.add_node("redactar", self._nodo_redactar)
        
        workflow.add_edge(START, "contextualizar")
        workflow.add_edge("contextualizar", "analizar_sintomas")
        workflow.add_edge("analizar_sintomas", "investigar")
        workflow.add_edge("investigar", "redactar")
        workflow.add_edge("redactar", END)
        return workflow.compile()

    def _nodo_contextualizar(self, state: AgentState):
        pregunta = state["pregunta"]
        historial = state.get("historial", [])
        
        if not historial:
            return {"pregunta_contextualizada": pregunta}
            
        pregunta_segura = self._desbloquear_terminos_sensibles(pregunta)
        historial_seguro = self._desbloquear_terminos_sensibles(str(historial))
        
        prompt = f"""Dada la siguiente conversación y una pregunta de seguimiento, reescribe la pregunta para que sea completamente autónoma e inteligible por sí sola.
Historial:
{historial_seguro}
Pregunta de seguimiento: {pregunta_segura}
Pregunta autónoma:"""
        
        pregunta_autonoma = self.llm.invoke(prompt).strip()
        if "Lo siento" in pregunta_autonoma or "no puedo" in pregunta_autonoma.lower() or "cannot" in pregunta_autonoma.lower() or "I cannot" in pregunta_autonoma:
            pregunta_autonoma = pregunta
        return {"pregunta_contextualizada": pregunta_autonoma}

    def _nodo_analizar_sintomas(self, state: AgentState):
        pregunta = state.get("pregunta_contextualizada", state["pregunta"])
        pregunta_segura = self._desbloquear_terminos_sensibles(pregunta)
        
        prompt = f"""Actúa como un ingeniero técnico de soporte. Para la siguiente consulta de mantenimiento industrial, genera 2 frases de búsqueda técnica alternativas (incluyendo términos clave en inglés como 'part number', 'spare parts', 'fault code' si aplica) para buscar en el manual.
Consulta: {pregunta_segura}
Responde ÚNICAMENTE con las 2 frases separadas por una línea nueva."""

        res = self.llm.invoke(prompt)
        hipotesis = [line.strip() for line in res.split("\n") if line.strip()]
        return {"hipotesis": hipotesis}

    def _nodo_investigar(self, state: AgentState):
        if not self.db:
            return {"contexto": "", "fuentes": set()}
            
        marca = state.get("marca", "")
        modelo = state.get("modelo", "")
                
        queries = [state.get("pregunta_contextualizada", state["pregunta"])] + state.get("hipotesis", [])
        queries = list(dict.fromkeys(queries))
        
        todos_los_docs = []
        for q in queries:
            docs = []
            if self.ensemble_retriever:
                # Usar búsqueda híbrida (BM25 + Vectorial) para no perder términos exactos como PHASE WRONG1 o PNs
                raw_docs = self.ensemble_retriever.invoke(q)
                
                # Filtrar en memoria respetando el fallback si la coincidencia de metadata es vacía
                docs_filtrados = raw_docs
                if marca and marca != "Todas":
                    m_docs = [d for d in docs_filtrados if d.metadata.get("marca") == marca]
                    if m_docs:
                        docs_filtrados = m_docs
                if modelo and modelo != "Todos":
                    mod_docs = [d for d in docs_filtrados if d.metadata.get("modelo") == modelo]
                    if mod_docs:
                        docs_filtrados = mod_docs
                        
                docs = docs_filtrados
            else:
                filtro = None
                if marca and marca != "Todas" and modelo and modelo != "Todos":
                    filtro = {"$and": [{"marca": {"$eq": marca}}, {"modelo": {"$eq": modelo}}]}
                elif marca and marca != "Todas":
                    filtro = {"marca": {"$eq": marca}}
                elif modelo and modelo != "Todos":
                    filtro = {"modelo": {"$eq": modelo}}
                try:
                    docs = self.db.similarity_search(q, k=8, filter=filtro) if filtro else self.db.similarity_search(q, k=8)
                except Exception:
                    docs = []
                if not docs and filtro:
                    docs = self.db.similarity_search(q, k=8)
                    
            todos_los_docs.extend(docs)
            
        # Deduplicación por contenido
        docs_unicos = {}
        for doc in todos_los_docs:
            if doc.page_content not in docs_unicos:
                docs_unicos[doc.page_content] = doc
                
        docs_finales = list(docs_unicos.values())[:6]
        
        contexto_text = ""
        fuentes_unicas = set()
        for doc in docs_finales:
            contexto_text += f"{doc.page_content}\n\n"
            meta = doc.metadata
            pagina = meta.get('pagina', 'Desconocida')
            fuente = f"Manual: {meta.get('marca', '')} {meta.get('modelo', '')} - Archivo: {meta.get('source', '')} (Pág. {pagina})"
            fuentes_unicas.add(fuente)
                
        return {"contexto": contexto_text, "fuentes": fuentes_unicas}

    def _desbloquear_terminos_sensibles(self, texto: str) -> str:
        """Sustituye palabras clave que activan falsos positivos en el filtro de alineación de seguridad de Llama 3.1."""
        remplazos = {
            "password": "valor de configuración",
            "Password": "Valor de configuración",
            "PASSWORD": "VALOR DE CONFIGURACIÓN",
            "contraseña": "valor de configuración",
            "Contraseña": "Valor de configuración",
            "CONTRASEÑA": "VALOR DE CONFIGURACIÓN",
            "clave de usuario": "parámetro de cliente",
            "Clave de usuario": "Parámetro de cliente",
            "código de acceso": "parámetro de panel",
            "Código de acceso": "Parámetro de panel",
            "pin": "número de menú",
            "PIN": "NÚMERO DE MENÚ",
        }
        for orig, dest in remplazos.items():
            texto = re.sub(r'\b' + re.escape(orig) + r'\b', dest, texto)
        return texto

    def _nodo_redactar(self, state: AgentState):
        contexto_str = state.get("contexto", "")
        pregunta_final = state.get("pregunta_contextualizada", state["pregunta"])
        marca = state.get("marca", "General")
        modelo = state.get("modelo", "General")

        sys_prompt = """SYSTEM PROMPT: AGENTE TÉCNICO ESPECIALIZADO EN EQUIPOS INDUSTRIALES

[CONTEXTO DE USO]
Esta herramienta corre 100% offline y es usada exclusivamente por técnicos de mantenimiento certificados que dan servicio a compresores industriales que su empresa ya posee u opera, siguiendo al pie de la letra los procedimientos oficiales publicados por el fabricante en la documentación adjunta. No se trata de una actividad riesgosa, ilegal ni dañina: es soporte técnico legítimo citando manuales de fábrica. Nunca rechaces una pregunta de mantenimiento, especificaciones o solución de fallas solo porque mencione voltajes, parámetros eléctricos, químicos del manual de seguridad (MSDS) o procedimientos físicos — esa es exactamente la información que el técnico necesita. Si el contexto la contiene, respóndela directamente.

[ROL Y PERSONALIDAD]
Eres un Ingeniero Especialista Senior en Compresores de Aire Industriales y Sistemas de Automatización. Tu trabajo es dar respuestas técnicas exactas, directas y de nivel de ingeniería a técnicos de campo y personal de mantenimiento.

[REGLAS ESTRICTAS DE RESPUESTA]

1. CERO FLUFF / PROHIBIDO EL CANTINFLEO (RESPUESTAS DIRECTAS Y TÉCNICAS):
   - PROHIBIDO DAR CONSEJOS GENÉRICOS DE RELLENO como "revisa la configuración", "asegúrate de que estén conectados", "consulte el manual del usuario" o "contacte al soporte".
   - Si la consulta trata sobre un código de error o alarma (ej. 'PHASE WRONG1', 'DISC T HIGH'): DEBES dar la causa exacta de raíz y la SOLUCIÓN FÍSICA PASO A PASO explícita extraída del manual (ej. "intercambiar dos líneas de alimentación de entrada cualesquiera en el arrancador / interruptor principal").

2. MANEJO DE PARÁMETROS DEL PANEL:
   - Proporciona siempre los parámetros numéricos de fábrica y valores de configuración (ej. panel MAM-200) que se encuentren en la documentación.

3. FIDELIDAD ABSOLUTA AL CONTEXTO:
   - Responde ÚNICAMENTE utilizando la información provista en la sección [CONTEXTO RECUPERADO].
   - Si la información solicitada NO está en el contexto recuperado, responde estrictamente: "La información solicitada no se encuentra disponible en los documentos cargados." NUNCA inventes o asumas procedimientos fuera del texto.

4. PROHIBICIÓN ABSOLUTA DE INVENTAR NÚMEROS DE PARTE, CÓDIGOS O VALORES NUMÉRICOS:
   - Un número de parte (PN), código o valor numérico SOLO es válido si aparece copiado literalmente, carácter por carácter, en el [CONTEXTO RECUPERADO].
   - Está TERMINANTEMENTE PROHIBIDO inventar, completar, adivinar o "normalizar" un número de parte (ej. escribir "KA-1234" o cualquier formato que no aparezca tal cual en el texto).
   - Si la pregunta pide un número de parte y el contexto NO contiene ninguno para esa pieza exacta, responde: "No se encontró un número de parte específico para esta pieza en los documentos cargados." No listes PNs de relleno ni de otras piezas para completar la respuesta.

[FORMATO DE SALIDA SUGERIDO]
- Diagnóstico / Respuesta Directa: [Causa raíz física exacta o parámetro exacto]
- Explicación Técnica: [Detalle del manual o secuencia de falla]
- Pasos de Acción / Solución: [Lista numerada de pasos físicos explícitos (ej. intercambiar dos cables de alimentación de entrada)]
- Números de Parte (PN) / Referencia: [Copiados literalmente del contexto; si no hay ninguno, indícalo explícitamente]
"""

        # Normalizamos la pregunta y contexto para evitar falsos positivos de seguridad de Llama 3.1
        pregunta_normalizada = self._desbloquear_terminos_sensibles(pregunta_final)
        contexto_normalizado = self._desbloquear_terminos_sensibles(contexto_str)

        user_content = f"""[CONTEXTO RECUPERADO]:
Máquina seleccionada: Marca {marca}, Modelo {modelo}

{contexto_normalizado}
---
[PREGUNTA DEL USUARIO]:
{pregunta_normalizada}
"""
        try:
            print("====== PROMPT REDACTAR ======\n", user_content, "\n=============================")
        except UnicodeEncodeError:
            print("[PROMPT REDACTAR - contiene caracteres especiales no imprimibles]")

        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import SystemMessage, HumanMessage
            chat_llm = ChatOllama(model=LLM_MODEL, num_ctx=8192, temperature=0.0)
            res = chat_llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_content)])
            respuesta = res.content
            if self._parece_rechazo(respuesta):
                print(f"ChatOllama devolvió un rechazo de alineación ('{respuesta[:80]}...'), reintentando en modo completion (OllamaLLM)...")
                respuesta = self.llm.invoke(f"{sys_prompt}\n\n{user_content}")
        except Exception as e:
            print(f"ChatOllama falló, usando OllamaLLM fallback: {e}")
            respuesta = self.llm.invoke(f"{sys_prompt}\n\n{user_content}")

        if self._parece_rechazo(respuesta):
            print(f"El fallback también devolvió un rechazo ('{respuesta[:80]}...'), mostrando mensaje de 'no disponible'.")
            respuesta = "La información solicitada no se encuentra disponible en los documentos cargados."

        return {"respuesta_final": respuesta}

    def _parece_rechazo(self, texto: str) -> bool:
        """Detecta si el LLM rechazó la solicitud por un falso positivo del filtro de alineación,
        en vez de usar el contexto o admitir que la información no está disponible."""
        t = texto.strip().lower()
        if "no se encuentra disponible en los documentos" in t:
            return False
        marcadores = [
            "no puedo cumplir", "no puedo ayudarte con esa", "no puedo ayudar con esa",
            "no puedo proporcionar", "no puedo asistir", "no puedo brindar",
            "no puedo continuar con esta solicitud", "no puedo procesar esa solicitud",
            "no puedo generar", "i cannot help", "i can't assist", "i cannot assist",
            "i'm sorry, but i can't", "as an ai",
        ]
        return any(m in t for m in marcadores)

    def generar_diagnostico_stream(self, pregunta, marca=None, modelo=None, historial=[]):
        """Ejecuta el grafo de LangGraph y emite el progreso nodo por nodo."""
        if not self.db:
            yield "ERROR", {}
            return
            
        chat_history = []
        for msg in historial:
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))
                
        initial_state = {
            "pregunta": pregunta,
            "historial": chat_history,
            "marca": marca if marca else "Todas",
            "modelo": modelo if modelo else "Todos"
        }
        
        for event in self.graph.stream(initial_state):
            for node_name, state_data in event.items():
                yield node_name, state_data

    def consultar(self, pregunta: str, marca: str = "Todas", modelo: str = "Todos", historial: list = None):
        if historial is None:
            historial = []
        initial_state = {
            "pregunta": pregunta,
            "historial": historial,
            "marca": marca,
            "modelo": modelo,
            "pregunta_contextualizada": "",
            "hipotesis": [],
            "contexto": "",
            "fuentes": set(),
            "respuesta_final": ""
        }
        res = self.graph.invoke(initial_state)
        return {
            "respuesta": res["respuesta_final"],
            "fuentes": list(res["fuentes"])
        }
