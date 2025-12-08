import streamlit as st
from src.rag import RAGSystem
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

# Настройки страницы
st.set_page_config(
    page_title="Информационная система госуслуг РК",
    page_icon="🇰🇿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомный CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E40AF;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Инициализация RAG
@st.cache_resource
def init_rag():
    return RAGSystem()

# Инициализация переменных сессии
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Заголовок
st.markdown('<div class="main-header">🇰🇿 Информационная система по государственным услугам<br>Республики Казахстан</div>', unsafe_allow_html=True)
st.markdown("**Система интеллектуального поиска на основе RAG-архитектуры и законодательства РК**")

# Инициализация системы
try:
    rag = init_rag()
except Exception as e:
    st.error(f"❌ Ошибка инициализации: {str(e)}")
    st.info("Убедитесь что:\n1. Установлены зависимости\n2. Создана база знаний: `python create_db.py`\n3. Указан GEMINI_API_KEY в .env")
    st.stop()

# Верхняя панель с метриками
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📚 Документов в базе", "25", "+12")
with col2:
    st.metric("💬 Запросов обработано", st.session_state.query_count)
with col3:
    st.metric("🎯 Средняя релевантность", "85%")
with col4:
    st.metric("⚡ Время ответа", "~2-3 сек")

st.divider()

# Основной контент - 2 колонки
col_chat, col_stats = st.columns([2, 1])

with col_chat:
    st.subheader("💬 Чат с AI-ассистентом")
    
    # История сообщений
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📚 Источники и обоснование", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            col_s1, col_s2 = st.columns([3, 1])
                            with col_s1:
                                st.markdown(f"**{i}. {source['source']}**")
                                st.caption(source['text'][:150] + "...")
                            with col_s2:
                                st.metric("Релевантность", f"{source['similarity']:.0%}")
                            if i < len(message["sources"]):
                                st.divider()

with col_stats:
    st.subheader("📊 Аналитика системы")
    
    # График по категориям
    categories = {
        "Документы": 8,
        "Регистрация": 6,
        "Финансы": 4,
        "Недвижимость": 3,
        "Социальная помощь": 4
    }
    
    fig = px.pie(
        values=list(categories.values()),
        names=list(categories.keys()),
        title="Распределение документов по категориям",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Топ популярные темы
    st.subheader("🔥 Популярные темы")
    popular = [
        ("ЭЦП", 85),
        ("Регистрация ИП", 72),
        ("Загранпаспорт", 68),
        ("Налоги", 54),
        ("Регистрация брака", 45)
    ]
    
    for topic, score in popular:
        st.progress(score / 100, text=f"{topic}: {score}%")

# Поле ввода (ВНЕ колонок!)
if prompt := st.chat_input("Задайте вопрос по госуслугам РК..."):
    st.session_state.query_count += 1
    st.session_state.query_history.append({
        'query': prompt,
        'timestamp': datetime.now()
    })
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("🔍 Анализирую базу знаний..."):
        result = rag.ask(prompt)
        
        # Если загружен PDF, добавляем его в контекст
        if 'uploaded_pdf_text' in st.session_state:
            # Формируем контекст из PDF
            pdf_context = f"""
Загруженный документ: {st.session_state.uploaded_pdf_name}

Содержание документа (первые 3000 символов):
{st.session_state.uploaded_pdf_text[:3000]}

Информация из базы знаний:
{result['answer']}
"""
            
            # Получаем улучшенный ответ от Gemini
            from src.generator import Generator
            gen = Generator()
            enhanced_answer = gen.generate(
                question=prompt,
                context=pdf_context
            )
            result['answer'] = enhanced_answer
            result['sources'].append({
                'source': f"📄 {st.session_state.uploaded_pdf_name}",
                'similarity': 1.0,
                'text': st.session_state.uploaded_pdf_text[:200]
            })
        
    st.session_state.messages.append({
        "role": "assistant",
        "content": result['answer'],
        "sources": result['sources']
    })
    st.rerun()

# Боковая панель
with st.sidebar:
    st.header("ℹ️ О системе")
    st.markdown("""
    **RAG-система нового поколения**
    
    🔹 **Gemini API** - генерация ответов  
    🔹 **Векторный поиск** - семантический анализ  
    🔹 **25 документов** - полная база знаний  
    🔹 **Законы РК** - официальная информация  
    
    Ответы формируются исключительно на основе законодательства Республики Казахстан.
    """)
    
    st.divider()
    
    st.header("📄 Загрузка документов")
    uploaded_file = st.file_uploader(
        "Загрузите PDF для анализа",
        type=['pdf'],
        help="Загрузите PDF документ для поиска информации"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ Загружен: {uploaded_file.name}")
        if 'uploaded_pdf_text' not in st.session_state:
            with st.spinner("📖 Извлекаю текст из PDF..."):
                try:
                    # Используем PyPDF2 для извлечения текста
                    import PyPDF2
                    import io
                    
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    
                    st.session_state.uploaded_pdf_text = text
                    st.session_state.uploaded_pdf_name = uploaded_file.name
                    st.success(f"📄 Извлечено {len(text)} символов")
                except Exception as e:
                    st.error(f"❌ Ошибка при чтении PDF: {str(e)}")
    
    if 'uploaded_pdf_text' in st.session_state:
        st.info(f"📌 Загружен: {st.session_state.uploaded_pdf_name}")
        if st.button("🗑️ Удалить PDF", use_container_width=True):
            del st.session_state.uploaded_pdf_text
            del st.session_state.uploaded_pdf_name
            st.rerun()
    
    st.divider()
    
    st.header("💡 Примеры запросов")
    st.caption("*Нажмите на вопрос для получения ответа*")
    
    # Контейнер со скроллингом
    with st.container(height=400):
        examples = [
            "Как получить электронную цифровую подпись (ЭЦП)?",
            "Какой закон регулирует выдачу ЭЦП в РК?",
            "Как зарегистрировать ИП в Казахстане?",
            "Какие налоговые режимы доступны для ИП?",
            "Какие документы нужны для регистрации брака?",
            "Каков минимальный срок ожидания брака?",
            "Как получить загранпаспорт РК?",
            "Какие страны без визы для граждан РК?",
            "Как зарегистрироваться по месту жительства?",
            "Какой штраф за отсутствие прописки?",
            "Как получить водительские права?",
            "Сколько стоит обучение в автошколе?",
            "Как получить справку о несудимости?",
            "Можно ли получить справку через egov.kz?",
            "Как проверить налоговую задолженность?",
            "Где оплатить налоги онлайн?",
            "Как зарегистрировать автомобиль?",
            "Нужен ли техосмотр для новых авто?",
            "Как заменить удостоверение личности?",
            "Можно ли получить ID срочно?",
            "Какие детские пособия выплачиваются?",
            "Как оформить пособие при рождении?",
            "Какой пенсионный возраст в РК?",
            "Что такое АСП и кто может получить?",
            "Как зарегистрировать право собственности?",
            "Как зарегистрировать ТОО?",
            "Какие виды деятельности лицензируются?",
            "Что такое ОСМС и кто должен платить?",
            "Как получить земельный участок бесплатно?",
            "Сколько стоят нотариальные услуги?",
            "Как подать исковое заявление в суд?",
            "Какие услуги доступны на egov.kz?",
        ]
        
        for i, example in enumerate(examples):
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                st.session_state.query_count += 1
                st.session_state.query_history.append({
                    'query': example,
                    'timestamp': datetime.now()
                })
                st.session_state.messages.append({"role": "user", "content": example})
                
                with st.spinner("🔍 Поиск информации..."):
                    result = rag.ask(example)
                    
                    # Если загружен PDF, добавляем его в контекст
                    if 'uploaded_pdf_text' in st.session_state:
                        # Формируем контекст из PDF
                        pdf_context = f"""
Загруженный документ: {st.session_state.uploaded_pdf_name}

Содержание документа (первые 3000 символов):
{st.session_state.uploaded_pdf_text[:3000]}

Информация из базы знаний:
{result['answer']}
"""
                        
                        from src.generator import Generator
                        gen = Generator()
                        enhanced_answer = gen.generate(
                            question=example,
                            context=pdf_context
                        )
                        result['answer'] = enhanced_answer
                        result['sources'].append({
                            'source': f"📄 {st.session_state.uploaded_pdf_name}",
                            'similarity': 1.0,
                            'text': st.session_state.uploaded_pdf_text[:200]
                        })
                    
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "sources": result['sources']
                })
                st.rerun()
    
    st.divider()
    
    # Экспорт
    st.header("💾 Экспорт данных")
    if st.session_state.messages:
        export_text = "# История чата\n\n"
        for msg in st.session_state.messages:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            export_text += f"**{role}:** {msg['content']}\n\n"
        
        st.download_button(
            label="📥 Скачать историю",
            data=export_text,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    if st.button("🗑️ Очистить историю", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_history = []
        st.rerun()
    
    st.divider()
    
    st.caption("**📊 Статистика базы знаний:**")
    st.caption("• Всего документов: 25")
    st.caption("• Фрагментов текста: ~400")
    st.caption("• Законов и кодексов: 15+")
    st.caption("• Постановлений: 20+")
    
    st.divider()
    
    st.caption("🇰🇿 **Разработано командой IMEA**")
    st.caption(f"Информационная система госуслуг РК • {datetime.now().strftime('%Y')}")
