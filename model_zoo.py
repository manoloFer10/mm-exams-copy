INSTRUCTIONS_COT = {
    "en": "The following is a multiple-choice question. Think step by step and then provide your FINAL answer between the tags <ANSWER> X </ANSWER> where X is ONLY the correct letter of your choice. Do not write additional text between the tags.",
    "es": "Lo siguiente es una pregunta de opción múltiple. Piensa paso a paso y luego proporciona tu RESPUESTA FINAL entre las etiquetas <ANSWER> X </ANSWER>, donde X es ÚNICAMENTE la letra correcta de tu elección. No escribas texto adicional entre las etiquetas.",
    "hi": "निम्नलिखित एक बहुविकल्पीय प्रश्न है। चरणबद्ध सोचें और फिर <ANSWER> X </ANSWER> टैग के बीच अपना अंतिम उत्तर प्रदान करें, जहाँ X केवल आपके चयन का सही अक्षर है। टैग के बीच अतिरिक्त कोई पाठ न लिखें।",
    "hu": "A következő egy feleletválasztós kérdés. Gondolkodj lépésről lépésre, majd add meg a VÉGSŐ válaszodat a <ANSWER> X </ANSWER> címkék között, ahol X CSAK a választott helyes betű. Ne írj további szöveget a címkék közé.",
    "hr": "Sljedeće je pitanje s višestrukim izborom. Razmislite korak po korak, a zatim dajte svoj ZAVRŠNI odgovor između oznaka <ANSWER> X </ANSWER> gdje je X SAMO ispravno slovo vašeg izbora. Nemojte pisati dodatni tekst između oznaka.",
    "uk": "Наступне — це питання з множинним вибором. Думайте крок за кроком, а потім надайте вашу ОСТАННЮ відповідь між тегами <ANSWER> X </ANSWER>, де X — ЛИШЕ правильна літера за вашим вибором. Не пишіть додаткового тексту між тегами.",
    "pt": "A seguir, temos uma questão de múltipla escolha. Pense passo a passo e depois forneça sua RESPOSTA FINAL entre as tags <ANSWER> X </ANSWER>, onde X é SOMENTE a letra correta da sua escolha. Não escreva texto adicional entre as tags.",
    "bn": "নিম্নলিখিতটি একটি বহু-বিকল্প প্রশ্ন। ধাপে ধাপে চিন্তা করুন এবং তারপর <ANSWER> X </ANSWER> ট্যাগের মধ্যে আপনার চূড়ান্ত উত্তর প্রদান করুন, যেখানে X শুধুমাত্র আপনার পছন্দের সঠিক অক্ষর। ট্যাগগুলির মধ্যে অতিরিক্ত কোনো লেখা লিখবেন না।",
    "te": "కింద ఇచ్చినది ఒక బహుళ ఎంపిక ప్రశ్న. దశల వారీగా ఆలోచించి, <ANSWER> X </ANSWER> ట్యాగ్లలో మీ తుది సమాధానాన్ని ఇవ్వండి, ఇక్కడ X మీ ఎంపికలోని సరైన అక్షరం మాత్రమే. ట్యాగ్లలో అదనపు వచనం రాయవద్దు.",
    "ne": "तलको प्रश्न बहुविकल्पीय छ। चरणबद्ध सोच्नुहोस् र त्यसपछि <ANSWER> X </ANSWER> ट्यागहरूबीच आफ्नो अन्तिम उत्तर प्रदान गर्नुहोस्, जहाँ X केवल तपाईंको रोजाइको सही अक्षर हो। ट्यागहरूबीच अतिरिक्त पाठ नलेख्नुहोस्।",
    "sr": "Sledeće je pitanje sa višestrukim izborom. Razmislite korak po korak, a zatim dajte svoj KONAČNI odgovor između oznaka <ANSWER> X </ANSWER>, gde je X SAMO tačno slovo vašeg izbora. Nemojte pisati dodatni tekst između oznaka.",
    "nl": "Het volgende is een meerkeuzevraag. Denk stap voor stap na en geef dan je UITEINDLIJKE antwoord tussen de tags <ANSWER> X </ANSWER>, waarbij X ALLEEN de juiste letter van je keuze is. Schrijf geen extra tekst tussen de tags.",
    "ar": "التالي هو سؤال اختيار من متعدد. فكر خطوة بخطوة ثم قدم إجابتك النهائية بين الوسوم <ANSWER> X </ANSWER> حيث X هي الحرف الصحيح فقط من اختيارك. لا تكتب نصًا إضافيًا بين الوسوم.",
    "ru": "Следующее — это вопрос с выбором ответа. Думайте шаг за шагом, а затем предоставьте ваш ОКОНЧАТЕЛЬНЫЙ ответ между тегами <ANSWER> X </ANSWER>, где X — ТОЛЬКО правильная буква вашего выбора. Не пишите дополнительный текст между тегами.",
    "fr": "Ce qui suit est une question à choix multiple. Réfléchissez étape par étape, puis donnez votre RÉPONSE FINALE entre les balises <ANSWER> X </ANSWER>, où X est UNIQUEMENT la lettre correcte de votre choix. N'écrivez pas de texte supplémentaire entre les balises.",
    "fa": "متن زیر یک سوال چندگزینه‌ای است. مرحله به مرحله فکر کنید و سپس پاسخ نهایی خود را بین تگ‌های <ANSWER> X </ANSWER> قرار دهید، جایی که X تنها حرف صحیح انتخاب شماست. متن اضافی بین تگ‌ها ننویسید.",
    "de": "Im Folgenden ist eine Multiple-Choice-Frage. Denken Sie Schritt für Schritt nach und geben Sie dann Ihre ENDGÜLTIGE Antwort zwischen den Tags <ANSWER> X </ANSWER> an, wobei X NUR der korrekte Buchstabe Ihrer Wahl ist. Schreiben Sie keinen zusätzlichen Text zwischen den Tags.",
    "lt": "Toliau pateikiamas klausimas su keliomis pasirinkimo galimybėmis. Mąstykite žingsnis po žingsnio ir pateikite savo GALUTINĮ atsakymą tarp žymų <ANSWER> X </ANSWER>, kur X yra TIK teisinga jūsų pasirinkta raidė. Nerašykite jokio papildomo teksto tarp žymų.",
}

# Molmo

# Pangea


# Qwen2
def create_qwen2_prompt(question, method):
    content = []
    lang = question["language"]
    prompt = [INSTRUCTIONS_COT[lang]]
    if question["image"] is not None:
        content.append(
            {
                "type": "image",
                "image": question["image"],
                "resized_height": 256,
                "resized_width": 256,
            }
        )
        images = [question["image"]]
    else:
        images = None
    if method == "zero-shot":
        prompt.append(f"\n{question['question']} Options: \n")
        for t, option in enumerate(question["options"]):
            index = f"{chr(65+t)}. "
            prompt.append(f"{index}) {option}\n")
    content.append({"type": "text", "text": "".join(prompt)})
    message = [{"role": "user", "content": content}]
    return message, images


# GPT

# Gemini

# Claude
