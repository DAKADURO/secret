import streamlit as st
import pandas as pd
from motor_rag import MotorRAG
import time
import os
import json
import uuid
import glob
from datetime import datetime
from ingesta import ingestar_archivo_unico

st.set_page_config(page_title="Asistente de Compresores", layout="wide", page_icon="⚙️")

# --- ESTILOS INDUSTRIALES (PANEL DE CONTROL TÉCNICO) ---
industrial_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg-0: #121212;
    --bg-1: #1a1a1a;
    --bg-2: #212121;
    --panel-border: #333333;
    --amber: #f0a020;
    --amber-dim: #8a5c14;
    --text-main: #e8e6e1;
    --text-dim: #9a9a95;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
}
code, pre, .stCodeBlock {
    font-family: 'IBM Plex Mono', monospace !important;
}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Fondo base Global - Graphite plano, sin degradados de color */
.stApp {
    background: var(--bg-0) !important;
    background-attachment: fixed !important;
}

/* Franja de advertencia industrial bajo el header */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: repeating-linear-gradient(
        45deg,
        var(--amber), var(--amber) 12px,
        #1a1a1a 12px, #1a1a1a 24px
    );
    z-index: 999;
    opacity: 0.85;
}

/* Sidebar - panel sólido, sin blur */
[data-testid="stSidebar"] {
    background: var(--bg-1) !important;
    border-right: 1px solid var(--panel-border) !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--amber) !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}

/* Chat Input */
[data-testid="stChatInput"] {
    background: var(--bg-2) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 4px !important;
    box-shadow: none !important;
    transition: border-color 0.2s ease-in-out !important;
}
[data-testid="stChatInput"]:focus-within {
    border: 1px solid var(--amber) !important;
    box-shadow: 0 0 0 1px rgba(240, 160, 32, 0.25) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-main) !important;
}

/* Chat Messages Generales */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    padding: 1.25rem 1.5rem !important;
    margin-bottom: 0.85rem;
    border-radius: 4px !important;
    animation: fadeIn 0.25s ease-out;
}

/* User Message - panel neutro con acento lateral gris acero */
[data-testid="stChatMessage"]:has(div:contains("👤")) {
    background: var(--bg-1) !important;
    border: 1px solid var(--panel-border) !important;
    border-left: 3px solid #5b6b7a !important;
    box-shadow: none !important;
}

/* Bot Message - panel con acento lateral ámbar (señal industrial) */
[data-testid="stChatMessage"]:has(div:contains("✨")), [data-testid="stChatMessage"]:has(div:contains("⚙️")) {
    background: var(--bg-2) !important;
    border: 1px solid var(--panel-border) !important;
    border-left: 3px solid var(--amber) !important;
    box-shadow: none !important;
}

/* Botones - estilo botón de panel de control, rectos */
.stButton > button {
    background: var(--bg-2) !important;
    border: 1px solid var(--panel-border) !important;
    color: var(--text-main) !important;
    border-radius: 3px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: #2a2a2a !important;
    border-color: var(--amber) !important;
    color: var(--amber) !important;
    box-shadow: none !important;
    transform: none;
}

/* Expanders */
[data-testid="stExpander"] {
    background: var(--bg-1) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 4px !important;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background-color: var(--bg-2) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 3px !important;
    color: var(--text-main) !important;
}

/* Tab Headers */
[data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid var(--panel-border) !important;
}
[data-baseweb="tab"] {
    color: var(--text-dim) !important;
    font-weight: 500 !important;
}
[aria-selected="true"] {
    color: var(--amber) !important;
    background-color: transparent !important;
}

/* Métricas del panel de feedback */
[data-testid="stMetricValue"] {
    color: var(--amber) !important;
}

/* Texto principal */
p, span {
    color: var(--text-main) !important;
}
h1, h2, h3, h4, h5, h6 {
    color: var(--text-main) !important;
    font-weight: 600 !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
"""
st.markdown(industrial_style, unsafe_allow_html=True)

CHATS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datos", "chats")
os.makedirs(CHATS_DIR, exist_ok=True)

# --- FUNCIONES DE HISTORIAL ---
def save_chat(chat_id, messages):
    if not messages:
        return
    title = messages[0]["content"][:30] + "..." if len(messages[0]["content"]) > 30 else messages[0]["content"]
    chat_data = {
        "id": chat_id,
        "title": title,
        "date": datetime.now().isoformat(),
        "messages": messages
    }
    with open(os.path.join(CHATS_DIR, f"{chat_id}.json"), "w", encoding="utf-8") as f:
        json.dump(chat_data, f, ensure_ascii=False, indent=4)

def get_chats():
    chats = []
    for file in glob.glob(os.path.join(CHATS_DIR, "*.json")):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                chats.append(data)
        except:
            pass
    chats.sort(key=lambda x: x.get("date", ""), reverse=True)
    return chats

def load_chat(chat_id):
    file_path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("messages", [])
    return []

def delete_chat(chat_id):
    file_path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(file_path):
        os.remove(file_path)

def delete_all_chats():
    for file in glob.glob(os.path.join(CHATS_DIR, "*.json")):
        try:
            os.remove(file)
        except:
            pass

# --- FUNCIÓN FEEDBACK ---
def guardar_feedback(pregunta, respuesta, calificacion):
    log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datos", "feedback_log.json")
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append({
            "fecha": datetime.now().isoformat(),
            "pregunta": pregunta,
            "respuesta": respuesta,
            "calificacion": calificacion
        })
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error guardando feedback: {e}")

# Inicializamos el motor RAG
@st.cache_resource
def init_motor():
    return MotorRAG()

motor = init_motor()

# --- ESTADO INICIAL ---
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("⚙️ Asistente IA para Manuales de Compresores")
st.markdown("Consulta especificaciones técnicas, mantenimiento y resolución de problemas basándote estrictamente en tus manuales locales.")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("🕒 Historial de Chats")
    col_new, col_del_all = st.columns([1, 1])
    with col_new:
        if st.button("➕ Nuevo", use_container_width=True):
            st.session_state.current_chat_id = uuid.uuid4().hex
            st.session_state.messages = []
            st.rerun()
    with col_del_all:
        if st.button("🗑️ Borrar Todo", use_container_width=True):
            delete_all_chats()
            st.session_state.current_chat_id = uuid.uuid4().hex
            st.session_state.messages = []
            st.toast("¡Historial borrado por completo!")
            st.rerun()
        
    historial = get_chats()
    if historial:
        st.caption("Conversaciones Anteriores")
        for chat in historial:
            col_chat, col_del = st.columns([4, 1])
            with col_chat:
                if st.button(f"💬 {chat['title']}", key=chat["id"], use_container_width=True):
                    st.session_state.current_chat_id = chat["id"]
                    st.session_state.messages = load_chat(chat["id"])
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_{chat['id']}", use_container_width=True):
                    delete_chat(chat["id"])
                    if st.session_state.current_chat_id == chat["id"]:
                        st.session_state.current_chat_id = uuid.uuid4().hex
                        st.session_state.messages = []
                    st.toast("Chat eliminado.")
                    st.rerun()
    else:
        st.caption("No hay chats guardados aún.")

    st.divider()
    
    st.header("🔍 Filtros de Búsqueda")
    marcas_modelos = motor.get_marcas_modelos()
    if not marcas_modelos:
        st.warning("⚠️ No se encontraron manuales.")
        marca_seleccionada = "Todas"
        modelo_seleccionado = "Todos"
    else:
        lista_marcas = ["Todas"] + list(marcas_modelos.keys())
        marca_seleccionada = st.selectbox("Selecciona la Marca:", lista_marcas)
        if marca_seleccionada != "Todas":
            lista_modelos = ["Todos"] + sorted(marcas_modelos[marca_seleccionada])
            modelo_seleccionado = st.selectbox("Selecciona el Modelo:", lista_modelos)
        else:
            modelo_seleccionado = "Todos"
            
        # --- NUEVO: Mostrar manuales indexados ---
        documentos_indexados = motor.get_documentos_por_modelo()
        if marca_seleccionada != "Todas":
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("📄 Manuales Indexados:")
            if modelo_seleccionado != "Todos":
                archivos = documentos_indexados.get(marca_seleccionada, {}).get(modelo_seleccionado, [])
                for arch in archivos:
                    st.markdown(f"• `{arch}`")
            else:
                for mod, archivos in documentos_indexados.get(marca_seleccionada, {}).items():
                    for arch in archivos:
                        st.markdown(f"• **{mod}**: `{arch}`")

    st.divider()
    st.header("📄 Subir Manual (PDF o JSON)")
    with st.expander("Añadir a la base de datos"):
        nueva_marca = st.text_input("Marca del equipo:")
        nuevo_modelo = st.text_input("Modelo del equipo:")
        archivo_pdf = st.file_uploader("Selecciona el archivo", type=["pdf", "json"])
        
        if st.button("Subir e Ingestar", use_container_width=True):
            if not nueva_marca or not nuevo_modelo or not archivo_pdf:
                st.error("Rellena todos los campos.")
            else:
                with st.spinner("Procesando con IA..."):
                    target_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datos", "manuales", nueva_marca, nuevo_modelo)
                    os.makedirs(target_dir, exist_ok=True)
                    file_path = os.path.join(target_dir, archivo_pdf.name)
                    with open(file_path, "wb") as f:
                        f.write(archivo_pdf.getbuffer())
                        
                    prog_bar = st.progress(0)
                    status_text = st.empty()
                    def actualizar_progreso(msg, actual, total):
                        if total > 0:
                            status_text.text(msg)
                            prog_bar.progress(min(actual / total, 1.0))
                        
                    if archivo_pdf.name.lower().endswith(".json"):
                        from ingesta import ingestar_json_unico
                        exito, num_chunks = ingestar_json_unico(file_path, nueva_marca, nuevo_modelo, archivo_pdf.name, progress_callback=actualizar_progreso)
                    else:
                        exito, num_chunks = ingestar_archivo_unico(file_path, nueva_marca, nuevo_modelo, archivo_pdf.name, progress_callback=actualizar_progreso)
                    if exito:
                        st.success(f"¡Añadido! {num_chunks} fragmentos indexados.")
                        st.cache_resource.clear()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("Error al procesar.")

    st.divider()
    st.header("🧠 Subir PDF (Modo ETL Avanzado)")
    with st.expander("Ingestar con LLM — Máxima Precisión"):
        st.caption("El LLM leerá cada página, eliminará ruido, asociará tablas de partes y generará chunks semánticos autónomos. Más lento, pero mucho más preciso para preguntas técnicas complejas y números de parte.")
        etl_marca = st.text_input("Marca:", key="etl_marca")
        etl_modelo = st.text_input("Modelo:", key="etl_modelo")
        etl_pdf = st.file_uploader("Selecciona el PDF para ETL", type=["pdf"], key="etl_uploader")

        if st.button("⚡ Ingestar con ETL Avanzado", use_container_width=True):
            if not etl_marca or not etl_modelo or not etl_pdf:
                st.error("Rellena todos los campos.")
            else:
                with st.spinner("🧠 ETL Avanzado en curso... Esto puede tardar varios minutos por página."):
                    from ingesta import ingestar_archivo_unico
                    import traceback
                    target_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datos", "manuales", etl_marca, etl_modelo)
                    os.makedirs(target_dir, exist_ok=True)
                    file_path_etl = os.path.join(target_dir, etl_pdf.name)
                    with open(file_path_etl, "wb") as f:
                        f.write(etl_pdf.getbuffer())

                    prog_bar_etl = st.progress(0)
                    status_etl = st.empty()
                    def actualizar_etl(msg, actual, total):
                        if total > 0:
                            status_etl.text(msg)
                            prog_bar_etl.progress(min(actual / total, 1.0))

                    try:
                        exito_etl, num_etl = ingestar_archivo_unico(file_path_etl, etl_marca, etl_modelo, etl_pdf.name, progress_callback=actualizar_etl)
                        if exito_etl:
                            st.success(f"✅ ETL completado: {num_etl} chunks semánticos indexados.")
                            st.cache_resource.clear()
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ El pipeline ETL no generó chunks. Revisa la consola para más detalles.")
                    except Exception as etl_err:
                        st.error(f"❌ Error en ETL: {etl_err}")
                        st.code(traceback.format_exc(), language="python")

tab_chat, tab_admin = st.tabs(["💬 Chat Asistente", "📊 Panel de Feedback"])

with tab_chat:
    # --- MOSTRAR CHAT ---
    for i, message in enumerate(st.session_state.messages):
        avatar_icon = "👤" if message["role"] == "user" else "✨"
        with st.chat_message(message["role"], avatar=avatar_icon):
            import re
            content = message["content"]
            # Buscar todas las etiquetas [IMAGEN: ruta]
            imagenes = re.findall(r'\[IMAGEN:\s*(.+?)\]', content)
            
            # Limpiar el texto para no mostrar la etiqueta literal
            clean_content = re.sub(r'\[IMAGEN:\s*.+?\]', '', content)
            st.markdown(clean_content)
            
            # Renderizar cada imagen encontrada
            if imagenes:
                for img_path in imagenes:
                    full_img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datos", "imagenes", img_path)
                    if os.path.exists(full_img_path):
                        st.image(full_img_path, caption="Diagrama extraído del manual")

            if message["role"] == "assistant":
                if "fuentes" in message and message["fuentes"]:
                    with st.expander("📚 Fuentes consultadas"):
                        for f in message["fuentes"]:
                            st.caption(f"- {f}")
                
                col1, col2, _ = st.columns([1, 1, 8])
                with col1:
                    if st.button("👍", key=f"up_{st.session_state.current_chat_id}_{i}"):
                        pregunta = st.session_state.messages[i-1]["content"] if i > 0 else ""
                        guardar_feedback(pregunta, message["content"], "Positivo")
                        st.toast("¡Gracias por tu feedback!")
                with col2:
                    if st.button("👎", key=f"down_{st.session_state.current_chat_id}_{i}"):
                        pregunta = st.session_state.messages[i-1]["content"] if i > 0 else ""
                        guardar_feedback(pregunta, message["content"], "Negativo")
                        st.toast("¡Feedback guardado!")

    # --- INPUT DEL USUARIO ---
    if prompt := st.chat_input("Ej: ¿Cuál es el aceite recomendado y qué cantidad lleva?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat(st.session_state.current_chat_id, st.session_state.messages)
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="✨"):
            with st.status("Iniciando análisis...", expanded=True) as status:
                respuesta_final = ""
                fuentes = []
                
                for step_name, state_data in motor.generar_diagnostico_stream(
                    pregunta=prompt,
                    marca=marca_seleccionada,
                    modelo=modelo_seleccionado,
                    historial=st.session_state.messages[:-1]
                ):
                    if step_name == "ERROR":
                        status.update(label="Error: Base de datos no encontrada", state="error")
                        st.error("Por favor ejecuta ingesta.py primero.")
                        st.stop()
                    elif step_name == "contextualizar":
                        status.update(label="Analizando historial...", state="running")
                    elif step_name == "analizar_sintomas":
                        hip = state_data.get("hipotesis", [])
                        status.update(label=f"Generando hipótesis: {', '.join(hip)}", state="running")
                    elif step_name == "investigar":
                        fuentes = list(state_data.get("fuentes", []))
                        status.update(label=f"Investigando manuales ({len(fuentes)} fuentes)...", state="running")
                    elif step_name == "redactar":
                        respuesta_final = state_data.get("respuesta_final", "")
                        status.update(label="Diagnóstico completado", state="complete")
                        
            st.markdown(respuesta_final)
            if fuentes:
                with st.expander("📚 Fuentes consultadas"):
                    for f in fuentes:
                        st.caption(f"- {f}")
                            
        st.session_state.messages.append({
            "role": "assistant", 
            "content": respuesta_final,
            "fuentes": fuentes
        })
        save_chat(st.session_state.current_chat_id, st.session_state.messages)
        st.rerun()

with tab_admin:
    st.header("📊 Panel de Administración de Feedback")
    st.markdown("Revisa las calificaciones de los usuarios sobre las respuestas del bot para identificar áreas de mejora.")
    
    log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datos", "feedback_log.json")
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
            
            if logs:
                df = pd.DataFrame(logs)
                
                # Métricas
                total = len(df)
                positivos = len(df[df["calificacion"] == "Positivo"])
                negativos = len(df[df["calificacion"] == "Negativo"])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Feedback", total)
                col2.metric("👍 Positivos", positivos)
                col3.metric("👎 Negativos", negativos)
                
                st.divider()
                st.subheader("Registro detallado")
                
                # Mostrar tabla con formato
                st.dataframe(
                    df[["fecha", "calificacion", "pregunta", "respuesta"]].sort_values("fecha", ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.divider()
                if st.button("🗑️ Limpiar historial de feedback", type="primary"):
                    os.remove(log_file)
                    st.success("Historial de feedback limpiado.")
                    time.sleep(1)
                    st.rerun()
            else:
                st.info("El archivo de feedback está vacío.")
        except Exception as e:
            st.error(f"Error al leer el archivo de feedback: {e}")
    else:
        st.info("No hay registros de feedback todavía. Los usuarios deben calificar respuestas primero usando los botones 👍/👎 en el chat.")
