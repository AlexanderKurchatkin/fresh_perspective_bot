import os
import io
import time
import logging
import torch
import json
import random
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
from collections import Counter
from aiogram import Bot, Dispatcher, executor, types
from functions import image_captioning, image_detection, image_segmentation, final_label_outputs, generate_text, first_author, second_author, third_author


logging.basicConfig(filename='log.txt',level=logging.INFO)

TOKEN = 'YOUR TOKEN'

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMG-TO-TEXT-МОДЕЛИ И ДАННЫЕ
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

det_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
det_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101").to(DEVICE)

segm_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
segm_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic").to(DEVICE)

translation_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru").to(DEVICE)

with open(os.path.join(os.getcwd(), 'coco-panoptic-id2label.json')) as json_file: # УКАЗАТЬ КОРРЕКТНЫЙ АДРЕС ДЖСОНА!!!!
    segm_labels_dict = json.load(json_file)

# TEXT GENERATION-МОДЕЛИ И ДАННЫЕ

tokenizer_gpt = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

model_gpt_beliy = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2").to(DEVICE)
model_gpt_beliy.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models/model_gpt_beliy_288.pt'), map_location=DEVICE)) # УКАЗАТЬ КОРРЕКТНЫЙ АДРЕС И ВЕСА

model_gpt_eco = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2").to(DEVICE)
model_gpt_eco.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models/model_gpt_eco_300.pt'), map_location=DEVICE)) # УКАЗАТЬ КОРРЕКТНЫЙ АДРЕС И ВЕСА

model_gpt_kafka = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2").to(DEVICE)
model_gpt_kafka.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models/model_gpt_kafka_231.pt'), map_location=DEVICE)) # УКАЗАТЬ КОРРЕКТНЫЙ АДРЕС И ВЕСА

model_gpt_lovecraft = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2").to(DEVICE)
model_gpt_lovecraft.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models/model_gpt_lovecraft_300.pt'), map_location=DEVICE)) # УКАЗАТЬ КОРРЕКТНЫЙ АДРЕС И ВЕСА

model_gpt_platonov = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2").to(DEVICE)
model_gpt_platonov.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models/model_gpt_platonov_298.pt'), map_location=DEVICE)) # УКАЗАТЬ КОРРЕКТНЫЙ АДРЕС И ВЕСА

model_gpt_sokolov = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2").to(DEVICE)
model_gpt_sokolov.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models/model_gpt_sokolov_270.pt'), map_location=DEVICE)) # УКАЗАТЬ КОРРЕКТНЫЙ АДРЕС И ВЕСА

## Список моделей: модель, автор, n_beams, temperature, top_p, max_len
models = [(model_gpt_beliy, 'Андрея Белого', 2, 1.8, 0.9, 100),  (model_gpt_eco, 'Умберто Эко', 3, 1.9, 0.92, 120), 
        (model_gpt_kafka, 'Франца Кафки', 2, 2.1, 0.87, 110), (model_gpt_lovecraft, 'Говарда Лавкрафта', 3, 2.0, 0.84, 120), 
        (model_gpt_platonov, 'Андрея Платонова', 3, 3.1, 0.89, 110), (model_gpt_sokolov, 'Саши Соколова', 2, 2.1, 0.94, 130)] # ЗНАЧЕНИЯ - ФАЙНТЬЮНИТЬ

prompt_beginnings = ['Вокруг себя я постоянно вижу ', 'Сегодня, как и обычно, я вижу перед собой ', 'Какой вид открывается передо мной? Я вижу ', 'Моему взору открывается вид на ',
                      'Разные вещи находятся передо мной, такие как '] # ДОФАЙНТЬЮНИТЬ


def get_keyboard_one():
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    button_upload_image = types.InlineKeyboardButton("Тайные мысли писателей о вашем быте", callback_data="upload_image")
    button_about_bot = types.InlineKeyboardButton("Подробнее о проекте и его целях", callback_data="about_bot")
    keyboard.add(button_upload_image, button_about_bot)
    return keyboard


@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    greeting_message = '''Не приходилось ли вам ощущать порой неподвижность обстановки собственной жизни?
Не случалось ли замечать, что взор ваш привык к одинаковым формам вещей, а ум - к однообразным сюжетам дня?
...
Повседневность и привычка тривиализируют мир и его объёмы.
Но остаётся ли в привычном скрытая, косвенная новизна?
Что, если какая-то - быть может, сырая, не оформленная ещё - мысль именно о вас и о том, что находится у вас постоянно перед глазами, нечаянно промелькнула однажды в голове какого-нибудь автора-провидца во время работы над своим текстом?
Я подслушаю эти фантазии и расскажу о них вам.
Мне нужна от вас только фотография того, что вас окружает.'''
    await message.reply(greeting_message, reply_markup=get_keyboard_one())

# Загрузка фотографии
@dp.callback_query_handler(lambda c: c.data.startswith("upload_image"))
async def process_upload_image(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    
    # Prompt the user to upload an image
    await bot.send_message(callback_query.from_user.id, "Направьте, пожалуйста, фотографию.")

# О проекте
@dp.callback_query_handler(lambda c: c.data.startswith("about_bot"))
async def process_about_bot(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    about_message = '''Данное приложение БЛАБЛАБЛА
    ЛАЛАЛА
    ФФФФФФФФФФ'''  # НАПИСАТЬ
    await bot.send_message(callback_query.from_user.id, about_message)
    time.sleep(5)
    await bot.send_message(callback_query.from_user.id, "Что ещё могло бы быть вам интересно?", reply_markup=get_keyboard_one())

# Главный пайплайн
@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_uploaded_image(message: types.Message):
    user_id = message.from_user.id
    file = await bot.get_file(message.photo[0].file_id)
    img_bytes = await bot.download_file(file.file_path)
    raw_image = Image.open(img_bytes).convert('RGB')
    caption_ru = image_captioning(raw_image, caption_processor, caption_model, translation_tokenizer, translation_model)
    detected_objects_str_ru, detected_objects_list_ru = image_detection(raw_image, det_processor, det_model, translation_tokenizer, translation_model)
    segm_objects_str_ru, segm_objects_list_ru = image_segmentation(raw_image, segm_processor, segm_model, translation_tokenizer, translation_model, segm_labels_dict)
    objects_str_ru, objects_list_ru = final_label_outputs(detected_objects_str_ru, detected_objects_list_ru, segm_objects_str_ru, segm_objects_list_ru)
    await bot.send_message(user_id, "Я слышу что-то...")
    time.sleep(2)
    await bot.send_message(user_id, "...")
    authors = random.sample(models, 3)
    first_author_text = first_author(authors[0][0], authors[0][1], authors[0][2], authors[0][3], authors[0][4], authors[0][5], caption_ru, objects_str_ru, objects_list_ru, prompt_beginnings, tokenizer_gpt)
    await bot.send_message(user_id, first_author_text)
    second_author_text = second_author(authors[1][0], authors[1][1], authors[1][2], authors[1][3], authors[1][4], authors[1][5], caption_ru, objects_str_ru, objects_list_ru, prompt_beginnings, tokenizer_gpt)
    await bot.send_message(user_id, second_author_text)
    third_author_text = third_author(authors[2][0], authors[2][1], authors[2][2], authors[2][3], authors[2][4], authors[2][5], caption_ru, objects_str_ru, objects_list_ru, prompt_beginnings, tokenizer_gpt)
    await bot.send_message(user_id, third_author_text)
    time.sleep(5)
    final_message = '''...
    Неогранённые мысли обрывочны и странны - но они могут дать творчес
    Писательский разум редко рождает готовый текст. 
    Может быть, в фантазиях авторов о вашем собственном быте вы увидели фрагмент какого-то интересного сюжета или эхо необычной метафоры?
    Позаимствуйте их образы.
    Придумайте и напишите о привычных вещах что-то такое, что вы никогда прежде не обдумывали и о чём не писали - хотя бы в одном-двух предложениях.
    Пусть обычное каждый день становится необычным.
    :)''' # НАПИСАТЬ НОРМАЛЬНО
    time.sleep(5)
    await bot.send_message(user_id, "Что ещё могло бы быть вам интересно?", reply_markup=get_keyboard_one())


# Run your bot
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

