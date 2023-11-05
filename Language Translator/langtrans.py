from googletrans import Translator, LANGUAGES

def get_supported_languages():
    return LANGUAGES

def translate_text(text, source_lang, target_lang):
    translator = Translator()
    translation = translator.translate(text, src=source_lang, dest=target_lang)
    return translation.text

def main():
    print("Supported Languages:")
    for code, language in get_supported_languages().items():
        print(f"{code}: {language}")

    source_lang = input("Enter the source language code: ")
    target_lang = input("Enter the target language code: ")

    if source_lang not in get_supported_languages() or target_lang not in get_supported_languages():
        print("Invalid language codes. Please use the language codes from the list above.")
        return

    text = input("Enter the text to translate: ")
    translated_text = translate_text(text, source_lang, target_lang)
    print(f"Translated text: {translated_text}")

if __name__ == "__main__":
    main()
