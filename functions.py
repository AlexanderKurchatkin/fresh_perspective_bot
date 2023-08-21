import torch
import random
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import Counter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_captioning(raw_image, caption_processor, caption_model, translation_tokenizer, translation_model):
    caption_inputs = caption_processor(raw_image, return_tensors="pt").to(DEVICE)
    caption_out = caption_model.generate(**caption_inputs)
    caption_eng = caption_processor.decode(caption_out[0], skip_special_tokens=True)
    translation_input = translation_tokenizer(caption_eng, return_tensors="pt").to(DEVICE)
    translation_output = translation_model.generate(**translation_input)
    caption_ru = translation_tokenizer.batch_decode(translation_output, skip_special_tokens=True)[0]
    if caption_ru[-1] == '.':
        caption_ru = caption_ru[:-1]
    return caption_ru

def image_detection(raw_image, det_processor, det_model, translation_tokenizer, translation_model):
    det_inputs = det_processor(images=raw_image, return_tensors="pt").to(DEVICE)
    det_outputs = det_model(**det_inputs)
    det_target_sizes = torch.tensor([raw_image.size[::-1]])
    det_results = det_processor.post_process_object_detection(det_outputs, target_sizes=det_target_sizes, threshold=0.8)[0]
    det_labels_eng = []
    for label in det_results["labels"]:
        det_labels_eng.append(det_model.config.id2label[label.item()])
    if det_labels_eng == []:
        return '', []
    objects_counter = Counter(det_labels_eng)
    objects = []
    for obj in objects_counter:
        if objects_counter[obj] == 1:
            objects.append(obj)
        else:
            objects.append(obj + 's')
    objects_str = ', '.join(objects)
    translation_input = translation_tokenizer(objects_str, return_tensors="pt").to(DEVICE)
    translation_output = translation_model.generate(**translation_input)
    detected_objects_str_ru = translation_tokenizer.batch_decode(translation_output, skip_special_tokens=True)[0]
    detected_objects_list_ru = detected_objects_str_ru.split(', ')
    return detected_objects_str_ru, detected_objects_list_ru

def image_segmentation(raw_image, segm_processor, segm_model, translation_tokenizer, translation_model, segm_labels_dict):
    segm_inputs = segm_processor(images=raw_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        segm_outputs = segm_model(**segm_inputs)
    segm_result = segm_processor.post_process_panoptic_segmentation(segm_outputs, target_sizes=[raw_image.size[::-1]])[0]['segments_info']
    segm_labels_eng = [segm_labels_dict[str(item['label_id'])].split('-')[0] for item in segm_result if item['score'] >= 0.8]
    if segm_labels_eng == []:
        return '', []
    objects_counter = Counter(segm_labels_eng)
    objects = []
    for obj in objects_counter:
        if objects_counter[obj] == 1:
            objects.append(obj)
        else:
            objects.append(obj + 's') 
    objects_str = ', '.join(objects)
    translation_input = translation_tokenizer(objects_str, return_tensors="pt").to(DEVICE)
    translation_output = translation_model.generate(**translation_input)
    segm_objects_str_ru = translation_tokenizer.batch_decode(translation_output, skip_special_tokens=True)[0]
    segm_objects_list_ru = segm_objects_str_ru.split(', ')
    return segm_objects_str_ru, segm_objects_list_ru

def final_label_outputs(detected_objects_str_ru, detected_objects_list_ru, segm_objects_str_ru, segm_objects_list_ru):
    if len(segm_objects_list_ru) >= len(detected_objects_list_ru):
        objects_str_ru, objects_list_ru = segm_objects_str_ru, segm_objects_list_ru
    else:
        objects_str_ru, objects_list_ru = detected_objects_str_ru, detected_objects_list_ru
    if objects_list_ru == []:
        objects_list_ru = random.sample(['пустота', 'неизвестное', 'загадочный вид', 'неявное'], 2)
        objects_str_ru = ', '.join(objects_list_ru)
    return objects_str_ru, objects_list_ru

def generate_text(model, num_beams, temperature, top_p, max_length, caption_ru, objects_str_ru, objects_list_ru, prompt_beginnings, tokenizer_gpt):
    prompt_beginning_one, prompt_beginning_two = random.sample(prompt_beginnings, 2) # ПОДЛЕЖИТ ТЬЮНИНГУ
    word_one_for_prompt_one, word_two_for_prompt_one = random.sample(objects_list_ru, 2)
    starter_word = random.choice(objects_list_ru) # ПОДЛЕЖИТ ТЬЮНИНГУ
    text = objects_str_ru + '. ' + prompt_beginning_one + word_one_for_prompt_one + ' и ' + word_two_for_prompt_one + '. ' + prompt_beginning_two + caption_ru + '. ' + starter_word # ПОДЛЕЖИТ ТЬЮНИНГУ
    len_to_skip = len(text) - len(starter_word)
    input_ids = tokenizer_gpt.encode(text, return_tensors="pt").to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model.generate(input_ids,
                            do_sample=True,
                            num_beams=num_beams,
                            temperature=temperature,
                            top_p=top_p,
                            max_length=max_length,
                            )

    generated_text = '...' + list(map(tokenizer_gpt.decode, out))[0][len_to_skip:] + '...' # ПОДЛЕЖИТ ТЬЮНИНГУ
    return generated_text

def first_author(model, author, num_beams, temperature, top_p, max_length, caption_ru, objects_str_ru, objects_list_ru, prompt_beginnings, tokenizer_gpt):
    starting_sentences = [f'Я слышу первый голос!.. Это внутренний голос {author}: ',  f'Первая фантазия - фантазия  {author}: ',
                         f'Я слышу первым шёпот {author}: ', f'Первой мне явлена полуночная, дремотная мысль {author}: '] # РЕДАКТИРОВАТЬ
    starting_sentence = random.choice(starting_sentences)
    generated_text = generate_text(model, num_beams, temperature, top_p, max_length, caption_ru, objects_str_ru, objects_list_ru, prompt_beginnings, tokenizer_gpt)
    first_author_text = starting_sentence + generated_text
    return first_author_text

def second_author(model, author, num_beams, temperature, top_p, max_length, caption_ru, objects_str_ru, objects_list_ru, prompt_beginnings, tokenizer_gpt):
    starting_sentences = [f'Вторым я услышал голос {author}: ',  f'...Затем зазвучал голос {author}: ',
                         f'Теперь я узнаю голос {author}: ', f'Теперь звучит фантазия {author}: '] # РЕДАКТИРОВАТЬ
    starting_sentence = random.choice(starting_sentences)
    generated_text = generate_text(model, num_beams, temperature, top_p, max_length, caption_ru, objects_str_ru, objects_list_ru, prompt_beginnings, tokenizer_gpt)
    second_author_text = starting_sentence + generated_text
    return second_author_text

def third_author(model, author, num_beams, temperature, top_p, max_length, caption_ru, objects_str_ru, objects_list_ru, prompt_beginnings, tokenizer_gpt):
    starting_sentences = [f'Наконец, посреди темноты сна искрит мысль {author}: ',  f'Третий голос - голос {author}: ',
                         f'В конце звучит слово {author}: ', f'Третьей звучит фантазия {author}: '] # РЕДАКТИРОВАТЬ
    starting_sentence = random.choice(starting_sentences)
    generated_text = generate_text(model, num_beams, temperature, top_p, max_length, caption_ru, objects_str_ru, objects_list_ru, prompt_beginnings, tokenizer_gpt)
    third_author_text = starting_sentence + generated_text
    return third_author_text