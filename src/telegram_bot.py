import telebot
import os
import json
from motor_rag import MotorRAG
import time
import speech_recognition as sr
import imageio_ffmpeg
from dotenv import load_dotenv

load_dotenv()

# TOKEN DE TELEGRAM (definir en variable de entorno TELEGRAM_BOT_TOKEN o en un archivo .env)
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError(
        "Falta la variable de entorno TELEGRAM_BOT_TOKEN. "
        "Copia .env.example a .env y coloca ahí tu token de @BotFather."
    )
bot = telebot.TeleBot(TOKEN)

# Iniciar Motor RAG
print("Iniciando Motor RAG para Telegram...")
motor = MotorRAG()
marcas_modelos = motor.get_marcas_modelos()

# Memoria de sesiones de Telegram
# Estructura: {chat_id: {"marca": "KRSB", "modelo": "Serie_KRSB", "historial": []}}
sesiones = {}

def obtener_estado_sesion(chat_id):
    if chat_id not in sesiones:
        sesiones[chat_id] = {"marca": "Todas", "modelo": "Todos", "historial": []}
    return sesiones[chat_id]

from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    estado = obtener_estado_sesion(message.chat.id)
    
    texto = (
        "⚙️ *Bienvenido al Asistente IA de Compresores Airpipe*\n\n"
        "Estoy conectado a la base de datos de manuales técnicos.\n\n"
        "*Equipos disponibles en la BD:*\n"
    )
    
    if not marcas_modelos:
        texto += "⚠️ No hay manuales cargados todavía.\n"
    else:
        for marca, modelos in marcas_modelos.items():
            texto += f"- *{marca}*: {', '.join(modelos)}\n"
            
    texto += (
        "\n*Comandos útiles:*\n"
        "`/configurar` - Seleccionar marca y modelo interactivo\n"
        "`/clear` - Limpia tu historial de chat\n\n"
        f"Actualmente estás configurado para: *{estado['marca']} / {estado['modelo']}*\n"
        "¡Hazme cualquier pregunta técnica!"
    )
    
    # También enviamos el menú directamente en el start
    bot.send_message(message.chat.id, texto, parse_mode='Markdown')
    enviar_menu_marcas(message.chat.id)

@bot.message_handler(commands=['configurar', 'set'])
def configurar_equipo(message):
    enviar_menu_marcas(message.chat.id)

def enviar_menu_marcas(chat_id):
    if not marcas_modelos:
        bot.send_message(chat_id, "⚠️ No hay manuales cargados todavía. Sube un PDF desde la interfaz web primero.")
        return
        
    markup = InlineKeyboardMarkup()
    # Opción para buscar en "Todas"
    markup.add(InlineKeyboardButton("🔍 Buscar en Todas", callback_data="marca_Todas"))
    
    for marca in marcas_modelos.keys():
        markup.add(InlineKeyboardButton(f"🏭 {marca}", callback_data=f"marca_{marca}"))
        
    bot.send_message(chat_id, "👇 *Selecciona la Marca del compresor:*", reply_markup=markup, parse_mode="Markdown")

@bot.callback_query_handler(func=lambda call: call.data.startswith('marca_'))
def callback_marca(call):
    marca = call.data.split('marca_')[1]
    
    if marca == "Todas":
        estado = obtener_estado_sesion(call.message.chat.id)
        estado['marca'] = "Todas"
        estado['modelo'] = "Todos"
        estado['historial'] = []
        bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, 
                              text="✅ *Configurado para buscar en TODAS las marcas y modelos.*\nHistorial limpiado. ¿En qué te ayudo?", 
                              parse_mode="Markdown")
        return

    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("🔍 Todos los modelos", callback_data=f"modelo_{marca}_Todos"))
    
    for modelo in marcas_modelos.get(marca, []):
        markup.add(InlineKeyboardButton(f"⚙️ {modelo}", callback_data=f"modelo_{marca}_{modelo}"))
        
    bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                          text=f"🏭 Marca: *{marca}*\n👇 *Ahora selecciona el Modelo:*", 
                          reply_markup=markup, parse_mode="Markdown")

@bot.callback_query_handler(func=lambda call: call.data.startswith('modelo_'))
def callback_modelo(call):
    # Formato data: modelo_MARCA_MODELO
    partes = call.data.split('_', 2)
    if len(partes) == 3:
        _, marca, modelo = partes
        estado = obtener_estado_sesion(call.message.chat.id)
        estado['marca'] = marca
        estado['modelo'] = modelo
        estado['historial'] = [] # Limpiamos historial al cambiar de máquina
        
        bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, 
                              text=f"✅ Equipo configurado: *{marca} - {modelo}*\nHistorial limpiado. ¿En qué te ayudo?", 
                              parse_mode="Markdown")

@bot.message_handler(commands=['clear'])
def clear_history(message):
    estado = obtener_estado_sesion(message.chat.id)
    estado['historial'] = []
    bot.reply_to(message, "🧹 Historial de conversación borrado. Empezamos de cero.")

@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    try:
        msg_espera = bot.reply_to(message, "🎙️ *Procesando audio...*", parse_mode='Markdown')
        
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Guardar OGG temporal
        tmp_ogg = f"tmp_{message.chat.id}.ogg"
        with open(tmp_ogg, 'wb') as new_file:
            new_file.write(downloaded_file)
            
        # Convertir a WAV usando subprocess + imageio_ffmpeg (sin pydub para evitar el bug de ffprobe)
        tmp_wav = f"tmp_{message.chat.id}.wav"
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        import subprocess
        subprocess.run([ffmpeg_exe, "-i", tmp_ogg, tmp_wav, "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Reconocimiento de voz
        r = sr.Recognizer()
        with sr.AudioFile(tmp_wav) as source:
            audio_data = r.record(source)
            texto = r.recognize_google(audio_data, language="es-ES")
            
        # Limpiar temporales
        os.remove(tmp_ogg)
        os.remove(tmp_wav)
        
        bot.edit_message_text(chat_id=message.chat.id, message_id=msg_espera.message_id, 
                              text=f"🎤 *Escuché:* _{texto}_", parse_mode='Markdown')
                              
        # Pasar texto al flujo de RAG
        message.text = texto
        responder_pregunta(message)
        
    except sr.UnknownValueError:
        bot.edit_message_text(chat_id=message.chat.id, message_id=msg_espera.message_id, 
                              text="⚠️ No pude entender el audio. Por favor, intenta hablar más claro o con menos ruido.")
        try:
            os.remove(tmp_ogg)
            os.remove(tmp_wav)
        except:
            pass
    except Exception as e:
        import traceback
        print(f"Error procesando voz: {e}")
        traceback.print_exc()
        bot.edit_message_text(chat_id=message.chat.id, message_id=msg_espera.message_id, 
                              text="⚠️ Ocurrió un error al procesar el audio.")
        try:
            os.remove(tmp_ogg)
            os.remove(tmp_wav)
        except:
            pass

@bot.message_handler(func=lambda message: True)
def responder_pregunta(message):
    chat_id = message.chat.id
    estado = obtener_estado_sesion(chat_id)
    pregunta = message.text
    
    # Enviar mensaje de espera
    msg_espera = bot.reply_to(message, "⏳ *Analizando tu consulta...*", parse_mode='Markdown')
    
    try:
        respuesta_final = ""
        fuentes = []
        
        # Inyectar al flujo
        estado['historial'].append({"role": "user", "content": pregunta})
        
        for step_name, state_data in motor.generar_diagnostico_stream(
            pregunta=pregunta,
            marca=estado['marca'],
            modelo=estado['modelo'],
            historial=estado['historial'][:-1]
        ):
            if step_name == "ERROR":
                bot.edit_message_text(chat_id=chat_id, message_id=msg_espera.message_id, text="❌ Error: Base de datos no encontrada.")
                return
            elif step_name == "analizar_sintomas":
                bot.edit_message_text(chat_id=chat_id, message_id=msg_espera.message_id, text="🧠 *Generando hipótesis técnicas...*", parse_mode='Markdown')
            elif step_name == "investigar":
                bot.edit_message_text(chat_id=chat_id, message_id=msg_espera.message_id, text="📚 *Consultando manuales PDF...*", parse_mode='Markdown')
            elif step_name == "redactar":
                respuesta_final = state_data.get("respuesta_final", "No pude generar una respuesta.")
                fuentes = state_data.get("fuentes", [])
                
        # Formatear fuentes
        texto_fuentes = ""
        if fuentes:
            texto_fuentes = "\n\n*📚 Fuentes:*\n"
            for f in fuentes:
                texto_fuentes += f"- {f}\n"
                
        # Guardar en el historial local del bot
        estado['historial'].append({"role": "assistant", "content": respuesta_final})
        
        # Enviar respuesta final y/o fotos
        try:
            import re
            mensaje_completo = respuesta_final + texto_fuentes
            imagenes = re.findall(r'\[IMAGEN:\s*(.+?)\]', mensaje_completo)
            clean_mensaje = re.sub(r'\[IMAGEN:\s*.+?\]', '', mensaje_completo)
            
            if len(clean_mensaje) > 4000:
                bot.edit_message_text(chat_id=chat_id, message_id=msg_espera.message_id, text=clean_mensaje[:4000])
                bot.send_message(chat_id, clean_mensaje[4000:])
            else:
                bot.edit_message_text(chat_id=chat_id, message_id=msg_espera.message_id, text=clean_mensaje)
                
            if imagenes:
                for img_path in imagenes:
                    full_img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datos", "imagenes", img_path)
                    if os.path.exists(full_img_path):
                        with open(full_img_path, 'rb') as f:
                            bot.send_photo(chat_id, f, caption="Diagrama extraído del manual")
                            
        except Exception as e:
            bot.edit_message_text(chat_id=chat_id, message_id=msg_espera.message_id, text="Error enviando la respuesta formateada.")
            bot.edit_message_text(chat_id=chat_id, message_id=msg_espera.message_id, text=respuesta_final + texto_fuentes)
            
    except Exception as e:
        print(f"Error procesando mensaje: {e}")
        bot.edit_message_text(chat_id=chat_id, message_id=msg_espera.message_id, text="⚠️ Ocurrió un error al procesar tu mensaje.")

print("Bot de Telegram escuchando...")
bot.infinity_polling()
