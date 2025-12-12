import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import TimedOut
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

# Allow running both as `python -m backend.telegram_bot` and as a script.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.chatbot import RAGChatbot

# Simple in-memory chat history per chat_id
_chat_histories: Dict[int, List[Dict[str, str]]] = {}
_bot_instance: RAGChatbot | None = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def get_bot() -> RAGChatbot:
    global _bot_instance
    if _bot_instance is None:
        api_key = os.environ.get("API_KEY")
        if not api_key:
            raise RuntimeError("API_KEY environment variable is not set.")
        _bot_instance = RAGChatbot(api_key=api_key)
    return _bot_instance


def get_history(chat_id: int) -> List[Dict[str, str]]:
    return _chat_histories.setdefault(chat_id, [])


def add_turn(chat_id: int, role: str, content: str) -> None:
    history = get_history(chat_id)
    history.append({"role": role, "content": content})
    # Keep last few turns to bound context
    if len(history) > 10:
        _chat_histories[chat_id] = history[-10:]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("سلام! پیام خود را بفرستید تا پاسخ دهم.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    bot = get_bot()
    history = get_history(chat_id)
    add_turn(chat_id, "user", user_text)

    allow_general_knowledge = os.environ.get("ALLOW_TELEGRAM_GENERAL_ANSWERS", "false").lower() in ("1", "true", "yes")
    try:
        # Run the heavy RAG pipeline off the event loop to prevent Telegram request timeouts.
        answer = await asyncio.to_thread(
            bot.run_agent,
            user_text,
            history,
            allow_general_knowledge=allow_general_knowledge,
        )
    except TimedOut:
        answer = "Telegram timed out while sending/receiving. Please try again in a moment."
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error in run_agent: %s", exc)
        answer = "یک خطا رخ داد؛ لطفا بعداً تلاش کنید."  # Farsi: An error occurred; please try later.

    add_turn(chat_id, "assistant", answer)
    await update.message.reply_text(answer)


def create_application():
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN environment variable is not set.")

    # Increase Telegram API timeouts to tolerate slow model responses or network hiccups.
    request = HTTPXRequest(connect_timeout=30, read_timeout=30, write_timeout=30, pool_timeout=30)
    application = ApplicationBuilder().token(token).request(request).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return application


if __name__ == "__main__":
    application = create_application()
    logger.info("Starting Telegram bot...")
    application.run_polling(close_loop=False)
