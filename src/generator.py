import google .generativeai as genai 
from config import GEMINI_API_KEY 
from google .api_core import exceptions 


class Generator :
    """Генерация ответов через Gemini API с гибридным режимом RAG"""

    def __init__ (self ):
        if not GEMINI_API_KEY :

            raise ValueError ("Не найден GEMINI_API_KEY в .env файле!")

        genai .configure (api_key =GEMINI_API_KEY )

        self .model =genai .GenerativeModel ('gemini-2.5-flash')
        print ("✅ Gemini API подключен")

    def generate (self ,question :str ,context :str )->str :
        """
        Сгенерировать ответ. Использует контекст, если он есть, 
        иначе переключается на общие знания (Hybrid Mode).
        """


        prompt =f"""Ты - помощник по государственным услугам Казахстана.

Контекст из базы знаний:
{context}

Вопрос пользователя: {question}

Инструкции:
- Всегда отвечай на вопрос пользователя.
- Если **контекст релевантен и достаточен**, используй его как **основной источник** для точного и детального ответа. Струткурируй ответ с помощью заголовков, списков или жирного шрифта.
- Если **контекст нерелевантен, недостаточен или пуст** (например, содержит только фразу "Контекст не найден..."), используй свои общие знания, чтобы дать общий, полезный ответ.
- Если ты использовал **только общие знания**, обязательно начни ответ с фразы: "На основе общих знаний, не из базы:"
- Если ты использовал **контекст из базы знаний**, обязательно укажи источники или ссылку на них в конце ответа (это поле будет заполнено в app.py, тебе нужно только оставить место).
- Пиши простым и дружелюбным языком.

Ответ:"""

        try :
            response =self .model .generate_content (prompt )

            if response .text :
                return response .text 
            else :

                block_reason =response .prompt_feedback .block_reason .name if response .prompt_feedback .block_reason else "Неизвестно"
                print (f"⚠️ Gemini вернул пустой ответ. Причина: {block_reason}")
                return "К сожалению, не удалось сгенерировать ответ. Возможно, запрос был заблокирован фильтрами AI."

        except exceptions .NotFound as e :
            print (f"❌ Ошибка 404 (NotFound): {e}")
            return "Ошибка: Модель AI не найдена. Убедитесь, что имя модели ('gemini-2.5-flash') корректно."
        except exceptions .ResourceExhausted as e :
            print (f"❌ Ошибка лимита (ResourceExhausted): {e}")
            return "Ошибка: Превышен лимит использования API. Попробуйте позже."
        except Exception as e :

            print (f"❌ Критическая ошибка API: {e}")
            return "Произошла техническая ошибка при обращении к AI."