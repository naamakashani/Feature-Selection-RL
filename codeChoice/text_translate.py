# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 22:32:23 2021

@author: dafna
"""


# Imports the Google Cloud Translation library
# from google.cloud import translate
import six
from google.cloud import translate_v2 as translate
import pandas as pd

MIN_TEXT_LEN = 10

# # Initialize Translation client
# def translate_text(text="YOUR_TEXT_TO_TRANSLATE", project_id="translate-texts-26041508"):
#     """Translating Text."""

#     client = translate.TranslationServiceClient()

#     location = "global"

#     parent = f"projects/{project_id}/locations/{location}"

#     # Translate text from English to French
#     # Detail on supported types can be found here:
#     # https://cloud.google.com/translate/docs/supported-formats
#     response = client.translate_text(
#         request={
#             "parent": parent,
#             "contents": [text],
#             "mime_type": "text/plain",  # mime types: text/plain, text/html
#             "source_language_code": "iw", #hebrew
#             "target_language_code": "en-US",
#         }
#     )

#     # Display the translation for each input text provided
#     for translation in response.translations:
#         print("--")
#         print("Translated text: {}".format(translation.translated_text))
#     return response.translations



def translate_text(target, text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    

    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    # print(u"Text: {}".format(result["input"]))
    # print(u"Translation: {}".format(result["translatedText"]))
    # print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))
    return result["translatedText"]

# Text_with_translation = pd.read_pickle('data/text.pkl')
# Text = Text_with_translation['Free-text-all-exam-res']

# EngText = Text.copy()

# for i in range(len(Text)):
#     print(str(i) + " out of: " + str(len(Text)))
#     t = Text[i]
#     if len(t) >= MIN_TEXT_LEN:
#         nt = translate_text("en", t)
#         EngText[i] = nt
#     else:
#         EngText[i] = ""
    

# Text_with_translation['Free-text-all-exam-res-en-translated'] = EngText
# Text_with_translation.to_pickle('data/text_with_translation.pkl')

Text_with_translation = pd.read_pickle('data/text_with_translation.pkl')
EngText = Text_with_translation['Free-text-all-exam-res-en-translated']

i = 48201
while i <len(EngText):
    
    t = EngText[i]
    if len(t) >= MIN_TEXT_LEN:
        nt = translate_text("en", t)
        EngText[i] = nt
    else:
        EngText[i] = ""
    
    
    if (i%100)== 0:
        Text_with_translation['Free-text-all-exam-res-en-translated'] = EngText
        Text_with_translation.to_pickle('data/text_with_translation.pkl')
        print(str(i) + " out of: " + str(len(EngText)))
    i = i +1
    
    
Text_with_translation['Free-text-all-exam-res-en-translated'] = EngText
Text_with_translation.to_pickle('data/text_with_translation.pkl')
Text_with_translation.to_pickle('data/text_with_translation_23122021.pkl')
