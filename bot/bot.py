import os
import logging
import asyncio
import traceback
import html
import json
import tempfile
import pydub
from pathlib import Path
from datetime import datetime
import openai

import telegram
from pyrogram import Client, filters
from pyrogram.types import BotCommand, Message, User, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery, Update
from pyrogram.enums import ParseMode, ChatAction, ChatType
from pyrogram.errors import BadRequest
from telegram import Update

import config
import database
import openai_utils


# setup
db = database.Database()
app = Client(
    config.pyrogram_session, 
    api_id=config.telegram_api_id, 
    api_hash=config.telegram_api_hash, 
    session_string=config.pyrogram_session, 
    bot_token=config.telegram_token
    )

def allowed_users(_, client, message):
    allowed_telegram_usernames = config.allowed_telegram_usernames
    if not allowed_telegram_usernames:
        return True

    usernames = [x for x in allowed_telegram_usernames if isinstance(x, str)]
    user_ids = [x for x in allowed_telegram_usernames if isinstance(x, int)]

    return message.from_user.username in usernames or message.from_user.id in user_ids

user_filter = filters.create(allowed_users)

def command_filter(message: Message):
    return message.text.startswith("/")

async def not_command_filter(_, __, message: Message):
    return not command_filter(message)

filters.not_command = filters.create(not_command_filter)

logger = logging.getLogger(__name__)

user_semaphores = {}
user_tasks = {}

HELP_MESSAGE = """Commands:
⚪ /retry – Regenerate last bot answer
⚪ /new – Start new dialog
⚪ /mode – Select chat mode
⚪ /settings – Show settings
⚪ /balance – Show balance
⚪ /help – Show help

🎨 Generate images from text prompts in <b>👩‍🎨 Artist</b> /mode
👥 Add bot to <b>group chat</b>: /help_group_chat
🎤 You can send <b>Voice Messages</b> instead of text
"""

HELP_GROUP_CHAT_MESSAGE = """You can add bot to any <b>group chat</b> to help and entertain its participants!

Instructions (see <b>video</b> below):
1. Add the bot to the group chat
2. Make it an <b>admin</b>, so that it can see messages (all other rights can be restricted)
3. You're awesome!

To get a reply from the bot in the chat – @ <b>tag</b> it or <b>reply</b> to its message.
For example: "{bot_username} write a poem about Telegram"
"""


def split_text_into_chunks(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


async def register_user_if_not_exists(message: Message, user: User):
    if not db.check_if_user_exists(user.id):
        db.add_new_user(
            user.id,
            message.chat.id,
            username=user.username,
            first_name=user.first_name,
            last_name= user.last_name
        )
        db.start_new_dialog(user.id)

    if db.get_user_attribute(user.id, "current_dialog_id") is None:
        db.start_new_dialog(user.id)

    if user.id not in user_semaphores:
        user_semaphores[user.id] = asyncio.Semaphore(1)

    if db.get_user_attribute(user.id, "current_model") is None:
        db.set_user_attribute(user.id, "current_model", config.models["available_text_models"][0])

    # back compatibility for n_used_tokens field
    n_used_tokens = db.get_user_attribute(user.id, "n_used_tokens")
    if isinstance(n_used_tokens, int):  # old format
        new_n_used_tokens = {
            "gpt-3.5-turbo": {
                "n_input_tokens": 0,
                "n_output_tokens": n_used_tokens
            }
        }
        db.set_user_attribute(user.id, "n_used_tokens", new_n_used_tokens)

    # voice message transcription
    if db.get_user_attribute(user.id, "n_transcribed_seconds") is None:
        db.set_user_attribute(user.id, "n_transcribed_seconds", 0.0)

    # image generation
    if db.get_user_attribute(user.id, "n_generated_images") is None:
        db.set_user_attribute(user.id, "n_generated_images", 0)


async def is_bot_mentioned(client: Client, message: Message):
    try:
        if message.chat.type == ChatType.PRIVATE:
            return True

        if message.text is not None and ("@" + (await client.get_me()).username) in message.text:
            return True

        if message.reply_to_message is not None:
            if message.reply_to_message.from_user.id == (await client.get_me()).id:
                return True
    except:
        return True
    else:
        return False

@app.on_message(filters.command("start") & user_filter)
async def start_handle(client: Client, message: Message):
    await register_user_if_not_exists(message, message.from_user)
    user_id = message.from_user.id

    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    db.start_new_dialog(user_id)

    reply_text = "Hi! I'm <b>ChatGPT</b> bot implemented with OpenAI API 🤖\n\n"
    reply_text += HELP_MESSAGE

    await message.reply_text(reply_text, parse_mode=ParseMode.HTML)
    await show_chat_modes_handle(client, message)

    await client.send_chat_action(message.chat.id, ChatAction.TYPING)
    await client.set_bot_commands(
        [
            BotCommand("ask", "Ask bot (for group chats)"),
            BotCommand("new", "Start new dialog"),
            BotCommand("mode", "Select chat mode"),
            BotCommand("retry", "Re-generate response for previous query"),
            BotCommand("balance", "Show balance"),
            BotCommand("settings", "Show settings"),
            BotCommand("help", "Show help message"),
        ]
    )

@app.on_message(filters.command("help") & user_filter)
async def help_handle(client: Client, message: Message):
    user = message.from_user
    await register_user_if_not_exists(message, user)
    user_id = user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    await message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)

@app.on_message(filters.command("help_group_chat") & user_filter)
async def help_group_chat_handle(client: Client, message: Message):
    user = message.from_user
    await register_user_if_not_exists(message, user)
    user_id = user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    bot_username = (await client.get_me()).username
    text = HELP_GROUP_CHAT_MESSAGE.format(bot_username="@" + bot_username)

    await message.reply_text(text, parse_mode=ParseMode.HTML)
    await message.reply_video(config.help_group_chat_video_path)

@app.on_message(filters.command("retry") & user_filter)
async def retry_handle(client: Client, message: Message):
    await register_user_if_not_exists(message, message.from_user)
    if await is_previous_message_not_answered_yet(client, message): return

    user_id = message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
    if len(dialog_messages) == 0:
        await message.reply_text("No message to retry 🤷‍♂️")
        return

    last_dialog_message = dialog_messages.pop()
    db.set_dialog_messages(user_id, dialog_messages, dialog_id=None)  # last message was removed from the context

    await message_handle(client, message, text=last_dialog_message["user"], use_new_dialog_timeout=False)

@app.on_message(filters.command("ask"))
async def ask_handle(client: Client, message: Message):
    await message_handle(client, message)

@app.on_message(filters.text & filters.not_command & user_filter)
async def message_handle(client: Client, message: Message, text=None, use_new_dialog_timeout=True):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(message, client):
        return

    # check if message is edited
    if message.edit_date is not None:
        await edited_message_handle(client, message)
        return

    _message = text or message.text

    # remove bot mention (in group chats)
    if message.chat.type != "private":
        me = await client.get_me()
        _message = _message.replace("@" + me.username, "").strip()

    await register_user_if_not_exists(message, message.from_user)
    if await is_previous_message_not_answered_yet(client, message): return

    user_id = message.from_user.id
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    if chat_mode == "artist":
        await generate_image_handle(client, message, text=text)
        return

    async def message_handle_fn():
        # new dialog timeout
        if use_new_dialog_timeout:
            if (datetime.now() - db.get_user_attribute(user_id, "last_interaction")).seconds > config.new_dialog_timeout and len(db.get_dialog_messages(user_id)) > 0:
                db.start_new_dialog(user_id)
                await message.reply_text(f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅", parse_mode=ParseMode.HTML)
        db.set_user_attribute(user_id, "last_interaction", datetime.now())

        # in case of CancelledError
        n_input_tokens, n_output_tokens = 0, 0
        current_model = db.get_user_attribute(user_id, "current_model")

        try:
            # send placeholder message to user
            placeholder_message = await message.reply_text("...")

            # send typing action
            await client.send_chat_action(message.chat.id, action=ChatAction.TYPING)

            if _message is None or len(_message) == 0:
                 await message.reply_text("🥲 You sent <b>empty message</b>. Please, try again!", parse_mode=ParseMode.HTML)
                 return

            dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
            parse_mode = {
                "html": ParseMode.HTML,
                "markdown": ParseMode.MARKDOWN
            }[config.chat_modes[chat_mode]["parse_mode"]]

            chatgpt_instance = openai_utils.ChatGPT(model=current_model)
            if config.enable_message_streaming:
                gen = chatgpt_instance.send_message_stream(_message, dialog_messages=dialog_messages, chat_mode=chat_mode)
            else:
                answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = await chatgpt_instance.send_message(
                    _message,
                    dialog_messages=dialog_messages,
                    chat_mode=chat_mode
                )

                async def fake_gen():
                    yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

                gen = fake_gen()

            prev_answer = ""
            async for gen_item in gen:
                status, answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = gen_item

                answer = answer[:4096]  # telegram message limit

                # update only when 100 new symbols are ready
                if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                    continue

                try:
                    await client.edit_message_text(text=answer, chat_id=placeholder_message.chat.id, message_id=placeholder_message.id, parse_mode=parse_mode)
                except telegram.error.BadRequest as e:
                    if str(e).startswith("Message is not modified"):
                        continue
                    else:
                        await client.edit_message_text(text=answer, chat_id=placeholder_message.chat.id, message_id=placeholder_message.id)

                await asyncio.sleep(0.01)  # wait a bit to avoid flooding

                prev_answer = answer

            # update user data
            new_dialog_message = {"user": _message, "bot": answer, "date": datetime.now()}
            db.set_dialog_messages(
                user_id,
                db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
                dialog_id=None
            )

            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)

        except asyncio.CancelledError:
            # note: intermediate token updates only work when enable_message_streaming=True (config.yml)
            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
            raise

        except Exception as e:
            error_text = f"Something went wrong during completion. Reason: {e}"
            logger.error(error_text)
            await message.reply_text(error_text)
            return

        # send message if some messages were removed from the context
        if n_first_dialog_messages_removed > 0:
            if n_first_dialog_messages_removed == 1:
                text = "✍️ <i>Note:</i> Your current dialog is too long, so your <b>first message</b> was removed from the context.\n Send /new command to start new dialog"
            else:
                text = f"✍️ <i>Note:</i> Your current dialog is too long, so <b>{n_first_dialog_messages_removed} first messages</b> were removed from the context.\n Send /new command to start new dialog"
            await message.reply_text(text, parse_mode=ParseMode.HTML)

    async with user_semaphores[user_id]:
        task = asyncio.create_task(message_handle_fn())
        user_tasks[user_id] = task
    
        try:
            await task
        except asyncio.CancelledError:
            await message.reply_text("✅ Canceled", parse_mode=ParseMode.HTML)
        else:
            pass
        finally:
            if user_id in user_tasks:
                del user_tasks[user_id]



async def is_previous_message_not_answered_yet(client: Client, message: Message):
    await register_user_if_not_exists(message, message.from_user)

    user_id = message.from_user.id
    if user_semaphores[user_id].locked():
        text = "⏳ Please <b>wait</b> for a reply to the previous message\n"
        text += "Or you can /cancel it"
        await message.reply_text(text, reply_to_message_id=message.message_id, parse_mode='html')
        return True
    else:
        return False

@app.on_message(filters.voice & user_filter)
async def voice_message_handle(client: Client, message: Message):
    # Check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(client, message):
        return

    await register_user_if_not_exists(message, message.from_user)
    if await is_previous_message_not_answered_yet(client, message):
        return

    user_id = message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    voice = message.voice
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        voice_ogg_path = tmp_dir / "voice.ogg"

        # Download
        try:
            await client.download_media(voice, file_name=voice_ogg_path)
        except BadRequest:
            return

        # Convert to mp3
        voice_mp3_path = tmp_dir / "voice.mp3"
        pydub.AudioSegment.from_file(voice_ogg_path).export(voice_mp3_path, format="mp3")

        # Transcribe
        with open(voice_mp3_path, "rb") as f:
            transcribed_text = await openai_utils.transcribe_audio(f)

            if transcribed_text is None:
                transcribed_text = ""

    text = f"🎤: <i>{transcribed_text}</i>"
    await message.reply_text(text, parse_mode=ParseMode.HTML)

    # Update n_transcribed_seconds
    db.set_user_attribute(user_id, "n_transcribed_seconds", voice.duration + db.get_user_attribute(user_id, "n_transcribed_seconds"))

    await message_handle(client, message, transcribed_text)


async def generate_image_handle(client: Client, message: Message, text=None):
    user = message.from_user
    await register_user_if_not_exists(message, user)
    if await is_previous_message_not_answered_yet(client, message): return

    user_id = user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    await client.send_chat_action(chat_id=message.chat.id, action=ChatAction.UPLOAD_PHOTO)

    text = text or message.text

    try:
        image_urls = await openai_utils.generate_images(text, n_images=config.return_n_generated_images)
    except openai.error.InvalidRequestError as e:
        if str(e).startswith("Your request was rejected as a result of our safety system"):
            text = "🥲 Your request <b>doesn't comply</b> with OpenAI's usage policies.\nWhat did you write there, huh?"
            await message.reply_text(text, parse_mode=ParseMode.HTML)
            return
        else:
            raise

    # token usage
    db.set_user_attribute(user_id, "n_generated_images", config.return_n_generated_images + db.get_user_attribute(user_id, "n_generated_images"))

    for i, image_url in enumerate(image_urls):
        await client.send_chat_action(chat_id=message.chat.id, action=ChatAction.UPLOAD_PHOTO)
        await message.reply_photo(photo=image_url, parse_mode=ParseMode.HTML)

@app.on_message(filters.command("new") & user_filter)
async def new_dialog_handle(client: Client, message: Message):
    user = message.from_user
    await register_user_if_not_exists(message, user)
    if await is_previous_message_not_answered_yet(client, message): return

    user_id = user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    db.start_new_dialog(user_id)
    await message.reply_text("Starting new dialog ✅")

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    await message.reply_text(f"{config.chat_modes[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML)

@app.on_message(filters.command("cancel") & user_filter)
async def cancel_handle(client: Client, message: Message):
    user = message.from_user
    await register_user_if_not_exists(message, user)

    user_id = user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    if user_id in user_tasks:
        task = user_tasks[user_id]
        task.cancel()
    else:
        await message.reply_text("<i>Nothing to cancel...</i>", parse_mode=ParseMode.HTML)


def get_chat_mode_menu(page_index: int):
    n_chat_modes_per_page = config.n_chat_modes_per_page
    text = f"Select <b>chat mode</b> ({len(config.chat_modes)} modes available):"

    # buttons
    chat_mode_keys = list(config.chat_modes.keys())
    page_chat_mode_keys = chat_mode_keys[page_index * n_chat_modes_per_page:(page_index + 1) * n_chat_modes_per_page]

    keyboard = []
    for chat_mode_key in page_chat_mode_keys:
        name = config.chat_modes[chat_mode_key]["name"]
        keyboard.append([InlineKeyboardButton(name, callback_data=f"set_chat_mode|{chat_mode_key}")])

    # pagination
    if len(chat_mode_keys) > n_chat_modes_per_page:
        is_first_page = (page_index == 0)
        is_last_page = ((page_index + 1) * n_chat_modes_per_page >= len(chat_mode_keys))

        if is_first_page:
            keyboard.append([
                InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")
            ])
        elif is_last_page:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}"),
            ])
        else:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}"),
                InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")
            ])

    reply_markup = InlineKeyboardMarkup(keyboard)

    return text, reply_markup

@app.on_message(filters.command("mode") & user_filter)
async def show_chat_modes_handle(client: Client, message: Message):
    await register_user_if_not_exists(message, message.from_user)
    if await is_previous_message_not_answered_yet(client, message): return

    user_id = message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text, reply_markup = get_chat_mode_menu(0)
    await message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

@app.on_callback_query(filters.regex("^show_chat_modes"))
async def show_chat_modes_callback_handle(client: Client, callback_query: CallbackQuery):
    user = callback_query.from_user
    if await is_previous_message_not_answered_yet(client, callback_query): return

    user_id = user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    await callback_query.answer()

    page_index = int(callback_query.data.split("|")[1])
    if page_index < 0:
        return

    text, reply_markup = get_chat_mode_menu(page_index)
    try:
        await callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass

@app.on_callback_query(filters.regex("^set_chat_mode"))
async def set_chat_mode_handle(client: Client, callback_query: CallbackQuery):
    user = callback_query.from_user
    user_id = user.id

    await callback_query.answer()

    chat_mode = callback_query.data.split("|")[1]

    db.set_user_attribute(user_id, "current_chat_mode", chat_mode)
    db.start_new_dialog(user_id)

    await client.send_message(
        callback_query.message.chat.id,
        f"{config.chat_modes[chat_mode]['welcome_message']}",
        parse_mode=ParseMode.HTML
    )


def get_settings_menu(user_id: int):
    current_model = db.get_user_attribute(user_id, "current_model")
    text = config.models["info"][current_model]["description"]

    text += "\n\n"
    score_dict = config.models["info"][current_model]["scores"]
    for score_key, score_value in score_dict.items():
        text += "🟢" * score_value + "⚪️" * (5 - score_value) + f" – {score_key}\n\n"

    text += "\nSelect <b>model</b>:"

    # buttons to choose models
    buttons = []
    for model_key in config.models["available_text_models"]:
        title = config.models["info"][model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_settings|{model_key}")
        )
    reply_markup = InlineKeyboardMarkup([buttons])

    return text, reply_markup

@app.on_message(filters.command("settings") & user_filter)
async def settings_handle(client: Client, message: Message):
    await register_user_if_not_exists(message, message.from_user)
    if await is_previous_message_not_answered_yet(client, message): return

    user_id = message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text, reply_markup = get_settings_menu(user_id)
    await message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

@app.on_callback_query(filters.regex("^set_settings"))
async def set_settings_handle(client: Client, callback_query: CallbackQuery):
    user_id = callback_query.from_user.id

    query = callback_query
    await query.answer()

    _, model_key = query.data.split("|")
    db.set_user_attribute(user_id, "current_model", model_key)
    db.start_new_dialog(user_id)

    text, reply_markup = get_settings_menu(user_id)
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass

@app.on_message(filters.command("balance") & user_filter)
async def show_balance_handle(client: Client, message: Message):
    await register_user_if_not_exists(message, message.from_user)

    user_id = message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    # count total usage statistics
    total_n_spent_dollars = 0
    total_n_used_tokens = 0

    n_used_tokens_dict = db.get_user_attribute(user_id, "n_used_tokens")
    n_generated_images = db.get_user_attribute(user_id, "n_generated_images")
    n_transcribed_seconds = db.get_user_attribute(user_id, "n_transcribed_seconds")

    details_text = "🏷️ Details:\n"
    for model_key in sorted(n_used_tokens_dict.keys()):
        n_input_tokens, n_output_tokens = n_used_tokens_dict[model_key]["n_input_tokens"], n_used_tokens_dict[model_key]["n_output_tokens"]
        total_n_used_tokens += n_input_tokens + n_output_tokens

        n_input_spent_dollars = config.models["info"][model_key]["price_per_1000_input_tokens"] * (n_input_tokens / 1000)
        n_output_spent_dollars = config.models["info"][model_key]["price_per_1000_output_tokens"] * (n_output_tokens / 1000)
        total_n_spent_dollars += n_input_spent_dollars + n_output_spent_dollars

        details_text += f"- {model_key}: <b>{n_input_spent_dollars + n_output_spent_dollars:.03f}$</b> / <b>{n_input_tokens + n_output_tokens} tokens</b>\n"

    # image generation
    image_generation_n_spent_dollars = config.models["info"]["dalle-2"]["price_per_1_image"] * n_generated_images
    if n_generated_images != 0:
        details_text += f"- DALL·E 2 (image generation): <b>{image_generation_n_spent_dollars:.03f}$</b> / <b>{n_generated_images} generated images</b>\n"

    total_n_spent_dollars += image_generation_n_spent_dollars

    # voice recognition
    voice_recognition_n_spent_dollars = config.models["info"]["whisper"]["price_per_1_min"] * (n_transcribed_seconds / 60)
    if n_transcribed_seconds != 0:
        details_text += f"- Whisper (voice recognition): <b>{voice_recognition_n_spent_dollars:.03f}$</b> / <b>{n_transcribed_seconds:.01f} seconds</b>\n"

    total_n_spent_dollars += voice_recognition_n_spent_dollars


    text = f"You spent <b>{total_n_spent_dollars:.03f}$</b>\n"
    text += f"You used <b>{total_n_used_tokens}</b> tokens\n\n"
    text += details_text

    await message.reply_text(text, parse_mode=ParseMode.HTML)


async def edited_message_handle(client: Client, message: Message):
    if client.edited_message.chat.type == "private":
        text = "🥲 Unfortunately, message <b>editing</b> is not supported"
        await client.edited_message.reply_text(text, parse_mode=ParseMode.HTML)

#@app.on_error()
async def error_handle(client: Client, update: Update, error) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=error)

    try:
        # collect error message
        tb_list = traceback.format_exception(None, error, error.__traceback__)
        tb_string = "".join(tb_list)
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            f"An exception was raised while handling an update\n"
            f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
            "</pre>\n\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )

        # split text into multiple messages due to 4096 character limit
        for message_chunk in split_text_into_chunks(message, 4096):
            try:
                await client.send_message(update.chat.id, message_chunk, parse_mode=ParseMode.HTML)
            except BadRequest:
                # answer has invalid characters, so we send it without parse_mode
                await client.send_message(update.chat.id, message_chunk)
    except:
        await client.send_message(update.chat.id, "Some error in error handler")

app.run()