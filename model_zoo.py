import base64
from PIL import Image
import io

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

system_message = {
    "en": "You are an expert at solving multiple-choice questions. Carefully analyze the question, think step by step, and provide your FINAL answer between the tags <ANSWER> X </ANSWER>, where X is ONLY the correct choice in latin script. Do not write any additional text between the tags.",
    "es": "Eres un experto en resolver preguntas de opción múltiple. Analiza cuidadosamente la pregunta, piensa paso a paso y proporciona tu respuesta FINAL entre las etiquetas <ANSWER> X </ANSWER>, donde X es ÚNICAMENTE la opción correcta en escritura latina. No escribas ningún texto adicional entre las etiquetas.",
    "hi": "आप बहुविकल्पीय प्रश्नों को हल करने में विशेषज्ञ हैं। प्रश्न का सावधानीपूर्वक विश्लेषण करें, कदम दर कदम सोचें, और अपना अंतिम उत्तर <ANSWER> X </ANSWER> टैग के बीच प्रदान करें, जहां X केवल लैटिन लिपि में सही विकल्प है। टैग के बीच कोई अतिरिक्त पाठ न लिखें।",
    "hu": "Ön szakértő a többszörös választásos kérdések megoldásában. Elemezze gondosan a kérdést, gondolkozzon lépésről lépésre, és adja meg VÉGLEGES válaszát a <ANSWER> X </ANSWER> címkék között, ahol X CSAK a helyes választás latin írásban. Ne írjon semmilyen további szöveget a címkék közé.",
    "hr": "Vi ste stručnjak u rješavanju pitanja s višestrukim izborom. Pažljivo analizirajte pitanje, razmišljajte korak po korak i navedite svoj KONAČNI odgovor između oznaka <ANSWER> X </ANSWER>, gdje je X SAMO točan odabir u latiničnom pismu. Ne pišite nikakav dodatni tekst između oznaka.",
    "uk": "Ви є експертом у вирішенні питань з багатовибірковою відповіддю. Уважно проаналізуйте питання, міркуйте крок за кроком і надайте свій ОСТАТОЧНИЙ відповідь між тегами <ANSWER> X </ANSWER>, де X — ЦЕ ЛИШЕ правильний вибір у латинській абетці. Не пишіть жодного додаткового тексту між тегами.",
    "pt": "Você é um especialista em resolver questões de múltipla escolha. Analise cuidadosamente a pergunta, pense passo a passo e forneça sua resposta FINAL entre as tags <ANSWER> X </ANSWER>, onde X é APENAS a escolha correta em script latino. Não escreva nenhum texto adicional entre as tags.",
    "bn": "আপনি বহু নির্বাচনী প্রশ্ন সমাধানে একজন বিশেষজ্ঞ। প্রশ্নটি সাবধানে বিশ্লেষণ করুন, ধাপে ধাপে চিন্তা করুন এবং আপনার চূড়ান্ত উত্তর <ANSWER> X </ANSWER> ট্যাগের মধ্যে প্রদান করুন, যেখানে X শুধুমাত্র লাতিন লিপিতে সঠিক পছন্দ। ট্যাগের মধ্যে কোনও অতিরিক্ত লেখা লিখবেন না।",
    "te": "మీరు బహుళైచ్ఛిక ప్రశ్నలను పరిష్కరించడంలో నిపుణులు. ప్రశ్నను జాగ్రత్తగా విశ్లేషించండి, దశలవారీగా ఆలోచించండి మరియు మీ అంతిమ సమాధానాన్ని <ANSWER> X </ANSWER> ట్యాగ్ల మధ్య అందించండి, ఇక్కడ X కేవలం లాటిన్ లిపిలో సరైన ఎంపిక. ట్యాగ్ల మధ్య ఏదైనా అదనపు వచనాన్ని వ్రాయవద్దు.",
    "ne": "तपाईं बहुविकल्पीय प्रश्नहरू समाधान गर्न विशेषज्ञ हुनुहुन्छ। प्रश्नलाई ध्यानपूर्वक विश्लेषण गर्नुहोस्, चरणबद्ध रूपमा सोच्नुहोस्, र तपाईंको अन्तिम उत्तर <ANSWER> X </ANSWER> ट्यागहरू बीचमा प्रदान गर्नुहोस्, जहाँ X केवल ल्याटिन लिपिमा सही छनौट हो। ट्यागहरू बीचमा कुनै अतिरिक्त पाठ नलेख्नुहोस्।",
    "sr": "Ви стручњак у решавању питања са вишеструким избором. Пажљиво анализирајте питање, размишљајте корак по корак и наведите свој КОНАЧНИ одговор између ознака <ANSWER> X </ANSWER>, где је X САМО тачан избор у латиничном писму. Не пишите никакав додатни текст између ознака.",
    "nl": "U bent een expert in het oplossen van meerkeuzevragen. Analyseer de vraag zorgvuldig, denk stap voor stap na en geef uw FINALE antwoord tussen de tags <ANSWER> X </ANSWER>, waarbij X ALLEEN de juiste keuze is in Latijns schrift. Schrijf geen extra tekst tussen de tags.",
    "ar": "أنت خبير في حل الأسئلة متعددة الخيارات. قم بتحليل السؤال بعناية، فكر خطوة بخطوة، وقدم إجابتك النهائية بين الوسوم <ANSWER> X </ANSWER>، حيث X هو الخيار الصحيح فقط بالحروف اللاتينية. لا تكتب أي نص إضافي بين الوسوم.",
    "ru": "Вы эксперт в решении вопросов с множественным выбором. Внимательно проанализируйте вопрос, думайте шаг за шагом и предоставьте свой ОКОНЧАТЕЛЬНЫЙ ответ между тегами <ANSWER> X </ANSWER>, где X — ТОЛЬКО правильный выбор в латинской графике. Не пишите никакого дополнительного текста между тегами.",
    "fr": "Vous êtes un expert en résolution de questions à choix multiples. Analysez soigneusement la question, réfléchissez étape par étape et fournissez votre réponse FINALE entre les balises <ANSWER> X </ANSWER>, où X est UNIQUEMENT le bon choix en script latin. N'écrivez aucun texte supplémentaire entre les balises.",
    "fa": "شما یک متخصص در حل سوالات چند گزینه ای هستید. سوال را به دقت تحلیل کنید، گام به گام فکر کنید و پاسخ نهایی خود را بین تگ های <ANSWER> X </ANSWER> ارائه دهید، جایی که X تنها گزینه صحیح به خط لاتین است. بین تگ ها هیچ متن اضافی ننویسید.",
    "de": "Sie sind ein Experte im Lösen von Multiple-Choice-Fragen. Analysieren Sie die Frage sorgfältig, denken Sie Schritt für Schritt nach und geben Sie Ihre ENDGÜLTIGE Antwort zwischen den Tags <ANSWER> X </ANSWER> an, wobei X NUR die richtige Wahl in lateinischer Schrift ist. Schreiben Sie keinen zusätzlichen Text zwischen den Tags.",
    "lt": "Jūs esate ekspertas sprendžiant daugiakartinius klausimus. Atidžiai išanalizuokite klausimą, galvokite žingsnis po žingsnio ir pateikite savo GALUTINĮ atsakymą tarp žymų <ANSWER> X </ANSWER>, kur X yra TIK teisingas pasirinkimas lotynišku raštu. Nerašykite jokio papildomo teksto tarp žymų.",
}

instruction = {
    "en": "Shortly think step by step and provide your final answer between the tags <ANSWER> X </ANSWER> to solve the following question",
    "es": "Piensa brevemente paso a paso y proporciona tu respuesta final entre las etiquetas <ANSWER> X </ANSWER> para resolver la siguiente pregunta",
    "hi": "संक्षेप में कदम दर कदम सोचें और अपने अंतिम उत्तर को टैग्स <ANSWER> X </ANSWER> के बीच प्रदान करें ताकि निम्नलिखित प्रश्न हल हो सके",
    "hu": "Gondolkodj röviden lépésről lépésre, és add meg a végső válaszodat a <ANSWER> X </ANSWER> címkék között a következő kérdés megoldásához",
    "hr": "Kratko razmišljaj korak po korak i pruži svoj konačan odgovor između oznaka <ANSWER> X </ANSWER> kako bi riješio sljedeće pitanje",
    "uk": "Коротко подумайте крок за кроком і надайте свою остаточну відповідь між тегами <ANSWER> X </ANSWER>, щоб вирішити наступне питання",
    "pt": "Pense brevemente passo a passo e forneça sua resposta final entre as tags <ANSWER> X </ANSWER> para resolver a seguinte questão",
    "bn": "সংক্ষেপে ধাপে ধাপে ভাবুন এবং নিম্নলিখিত প্রশ্নের সমাধানের জন্য আপনার চূড়ান্ত উত্তরটি <ANSWER> X </ANSWER> ট্যাগের মধ্যে প্রদান করুন",
    "te": "సంక్షిప్తంగా అడుగు అడుగుగా ఆలోచించి, క్రింది ప్రశ్నను పరిష్కరించడానికి మీ చివరి సమాధానాన్ని <ANSWER> X </ANSWER> ట్యాగ్‌ల మధ్య ఇవ్వండి",
    "ne": "छोटो सोच्दै चरणबद्ध रूपमा अगाडि बढ्नुहोस् र निम्न प्रश्न समाधान गर्न तपाईंको अन्तिम उत्तर <ANSWER> X </ANSWER> ट्यागहरू बीच प्रदान गर्नुहोस्",
    "sr": "Kratko razmislite korak po korak i navedite svoj konačan odgovor između oznaka <ANSWER> X </ANSWER> kako biste rešili sledeće pitanje",
    "nl": "Denk kort stap voor stap na en geef je uiteindelijke antwoord tussen de tags <ANSWER> X </ANSWER> om de volgende vraag op te lossen",
    "ar": "فكر باختصار خطوة بخطوة وقدم إجابتك النهائية بين الوسوم <ANSWER> X </ANSWER> لحل السؤال التالي",
    "ru": "Кратко подумайте шаг за шагом и укажите свой окончательный ответ между тегами <ANSWER> X </ANSWER> для решения следующего вопроса",
    "fr": "Réfléchissez brièvement étape par étape et fournissez votre réponse finale entre les balises <ANSWER> X </ANSWER> pour résoudre la question suivante",
    "fa": "به طور خلاصه گام به گام فکر کنید و پاسخ نهایی خود را بین برچسب‌های <ANSWER> X </ANSWER> قرار دهید تا سوال زیر حل شود",
    "de": "Denke kurz Schritt für Schritt nach und gib deine endgültige Antwort zwischen den Tags <ANSWER> X </ANSWER> an, um die folgende Frage zu lösen",
    "lt": "Trumpai apmąstykite žingsnis po žingsnio ir pateikite galutinį atsakymą tarp žymų <ANSWER> X </ANSWER>, kad išspręstumėte šį klausimą",
}

few_shot_instruction = {
    "en": "Here are three examples of how to solve similar questions:",
    "es": "Aquí tienes tres ejemplos de cómo resolver preguntas similares:",
    "hi": "यहाँ तीन उदाहरण दिए गए हैं कि समान प्रश्नों को कैसे हल किया जाए:",
    "hu": "Íme három példa arra, hogyan lehet hasonló kérdéseket megoldani:",
    "hr": "Ovdje su tri primjera kako riješiti slična pitanja:",
    "uk": "Ось три приклади того, як розв’язати подібні питання:",
    "pt": "Aqui estão três exemplos de como resolver questões semelhantes:",
    "bn": "এখানে তিনটি উদাহরণ দেওয়া হল কিভাবে অনুরূপ প্রশ্নগুলোর সমাধান করা যায়:",
    "te": "ఇలాంటి ప్రశ్నలను ఎలా పరిష్కరించాలో చూపించే మూడు ఉదాహరణలు ఇవి:",
    "ne": "यहाँ तीन उदाहरणहरू छन् जसले समान प्रश्नहरू कसरी समाधान गर्ने देखाउँछ:",
    "sr": "Ovde su tri primera kako rešiti slična pitanja:",
    "nl": "Hier zijn drie voorbeelden van hoe je soortgelijke vragen kunt oplossen:",
    "ar": "إليك ثلاثة أمثلة على كيفية حل أسئلة مشابهة:",
    "ru": "Вот три примера того, как решать похожие вопросы:",
    "fr": "Voici trois exemples de la manière de résoudre des questions similaires :",
    "fa": "در اینجا سه مثال آورده شده است که نشان می‌دهد چگونه می‌توان سؤالات مشابه را حل کرد:",
    "de": "Hier sind drei Beispiele dafür, wie ähnliche Fragen gelöst werden können:",
    "lt": "Štai trys pavyzdžiai, kaip išspręsti panašius klausimus:",
}

keywords = {
    "en": {"question": "Question", "options": "Options", "answer": "Answer"},
    "es": {"question": "Pregunta", "options": "Opciones", "answer": "Respuesta"},
    "hi": {"question": "प्रश्न", "options": "विकल्प", "answer": "उत्तर"},
    "hu": {"question": "Kérdés", "options": "Lehetőségek", "answer": "Válasz"},
    "hr": {"question": "Pitanje", "options": "Opcije", "answer": "Odgovor"},
    "uk": {"question": "Питання", "options": "Варіанти", "answer": "Відповідь"},
    "pt": {"question": "Pergunta", "options": "Opções", "answer": "Resposta"},
    "bn": {"question": "প্রশ্ন", "options": "বিকল্প", "answer": "উত্তর"},
    "te": {"question": "ప్రశ్న", "options": "ఎంపికలు", "answer": "సమాధానం"},
    "ne": {"question": "प्रश्न", "options": "विकल्पहरू", "answer": "उत्तर"},
    "sr": {"question": "Pitanje", "options": "Opcije", "answer": "Odgovor"},
    "nl": {"question": "Vraag", "options": "Opties", "answer": "Antwoord"},
    "ar": {"question": "السؤال", "options": "الخيارات", "answer": "الإجابة"},
    "ru": {"question": "Вопрос", "options": "Варианты", "answer": "Ответ"},
    "fr": {"question": "Question", "options": "Options", "answer": "Réponse"},
    "fa": {"question": "سؤال", "options": "گزینه‌ها", "answer": "پاسخ"},
    "de": {"question": "Frage", "options": "Optionen", "answer": "Antwort"},
    "lt": {"question": "Klausimas", "options": "Pasirinkimai", "answer": "Atsakymas"},
}

SYS_MESSAGE = 'You are a helpful assistant who answers multiple-choice questions. For each question, output your final answer in JSON format with the following structure:\n\n{"choice": "The correct option (e.g., A, B, C, or D)"}\n\ONLY output this format exactly. Do not include any additional text or explanations outside the JSON structure.'
INSTRUCTION = "Output your choice in the specified JSON format."


# Molmo
def create_molmo_prompt_vllm2(question, method, few_shot_samples):
    lang = question["language"]
    lang_keyword = keywords[lang]
    prompt = [system_message[lang]]
    if question["image"] is not None:
        images = [question["image"]]
    else:
        images = None
    prompt.append(
        f"\n{lang_keyword['question']}: {question['question'].replace('<image>', '')} \n{lang_keyword['options']}: \n"
    )
    for t, option in enumerate(question["options"]):
        index = f"{chr(65+t)}. "
        prompt.append(f"{index}) {option.replace('<image>', '')}\n")
    prompt.append(f"\n{lang_keyword['answer']}:")
    prompt = "".join(prompt)

    prompt = f"<|im_start|>user <image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return prompt, images


def create_molmo_prompt_vllm(question, method, few_shot_samples):
    lang = question["language"]
    lang_keyword = keywords[lang]
    prompt = [SYS_MESSAGE]
    prompt.append(INSTRUCTION)
    if question["image"] is not None:
        images = [question["image"]]
    else:
        images = None
    prompt.append(
        f"\n{lang_keyword['question']}: {question['question'].replace('<image>', '')} \n{lang_keyword['options']}: \n"
    )
    for t, option in enumerate(question["options"]):
        index = f"{chr(65+t)}. "
        prompt.append(f"{index}) {option.replace('<image>', '')}\n")
    prompt.append(f"\nANSWER:")
    prompt = "".join(prompt)

    prompt = f"<|im_start|>user <image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return prompt, images


# Pangea
def create_pangea_prompt2(question, method, few_shot_samples):
    lang = question["language"]
    lang_keyword = keywords[lang]
    prompt = [
        f"<|im_start|>system\n{system_message[lang]}<|im_end|>\n<|im_start|>user\n"
    ]
    if question["image"] is not None:
        prompt.append("<image>\n")
        images = question["image"]
    else:
        images = None

    if method == "few-shot":
        few_shot = few_shot_samples.get(lang, [])
        if len(few_shot) != 0:
            prompt.append(f"\n{few_shot_instruction[lang]}\n")
            for q in few_shot:
                prompt.append(
                    f"\n{lang_keyword['question']}: {q['question'].replace('<image>', '')} \nOptions: \n"
                )
                for t, option in enumerate(q["options"]):
                    index = f"{chr(65+t)}. "
                    prompt.append(f"{index}) {option.replace('<image>', '')}\n")
                prompt.append(
                    f"\n{lang_keyword['answer']}: <ANSWER>{chr(65+q['answer'])}</ANSWER>\n"
                )
                prompt.append("\n---\n")

    prompt.append(f"\n{instruction[lang]}\n")
    prompt.append(
        f"\n{lang_keyword['question']}: {question['question'].replace('<image>', '')} \n{lang_keyword['options']}: \n"
    )
    for t, option in enumerate(question["options"]):
        index = f"{chr(65+t)}. "
        prompt.append(f"{index}) {option.replace('<image>', '')}\n")
    prompt.append(f"\n{lang_keyword['answer']}:")
    prompt.append("<|im_end|>\n<|im_start|>assistant\n")
    message = "".join(prompt)
    return message, images


def create_pangea_prompt(question, method, few_shot_samples):
    lang = question["language"]
    lang_keyword = keywords[lang]
    prompt = [f"<|im_start|>system\n{SYS_MESSAGE}<|im_end|>\n<|im_start|>user\n"]
    if question["image"] is not None:
        prompt.append("<image>\n")
        images = question["image"]
    else:
        images = None

    prompt.append(f"\n{INSTRUCTION}\n")
    prompt.append(
        f"\n{lang_keyword['question']}: {question['question'].replace('<image>', '')} \n{lang_keyword['options']}: \n"
    )
    for t, option in enumerate(question["options"]):
        index = f"{chr(65+t)}. "
        prompt.append(f"{index}) {option.replace('<image>', '')}\n")
    # prompt.append(f"\n{lang_keyword['answer']}:")
    prompt.append("<|im_end|>\n<|im_start|>assistant\n")
    message = "".join(prompt)
    return message, images


# Qwen2
def create_qwen_prompt(question, method, few_shot_samples):
    content = []
    lang = question["language"]
    prompt = []
    lang_keyword = keywords[lang]
    if question["image"] is not None:
        content.append(
            {"type": f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"}
        )
        images = [question["image"]]
    else:
        images = None

    if method == "few-shot":
        few_shot = few_shot_samples.get(lang, [])
        if len(few_shot) != 0:
            prompt.append(f"\n{few_shot_instruction[lang]}\n")
            for q in few_shot:
                prompt.append(
                    f"\n{lang_keyword['question']}: {q['question']} \n{lang_keyword['options']}: \n"
                )
                for t, option in enumerate(q["options"]):
                    index = f"{chr(65+t)}. "
                    prompt.append(f"{index}) {option}\n")
                prompt.append(
                    f"\n{lang_keyword['answer']}: <ANSWER> {chr(65+q['answer'])} </ANSWER>\n"
                )
                prompt.append("\n---\n")

    prompt.append(f"\n{instruction[lang]}\n")
    prompt.append(
        f"\n{lang_keyword['question']}: {question['question']} \n{lang_keyword['options']}: \n"
    )
    for t, option in enumerate(question["options"]):
        index = f"{chr(65+t)}. "
        prompt.append(f"{index}) {option}\n")
    prompt.append(f"\n{lang_keyword['answer']}:")
    content.append({"type": "text", "text": "".join(prompt)})
    message = [
        {"role": "system", "content": system_message[lang]},
        {"role": "user", "content": content},
    ]
    return message, images


def create_qwen_prompt_vllm2(question, method, few_shot_samples):
    # Determine the placeholder for images
    lang = question["language"]
    prompt = ""

    # Add few-shot examples if applicable
    if method == "few-shot":
        few_shot = few_shot_samples.get(lang, [])
        if few_shot:
            prompt += f"\n{few_shot_instruction[lang]}\n"
            for q in few_shot:
                prompt += (
                    f"\n{keywords[lang]['question']}: {q['question']}\n"
                    f"{keywords[lang]['options']}:\n"
                )
                for t, option in enumerate(q["options"]):
                    prompt += f"{chr(65 + t)}. {option}\n"
                prompt += (
                    f"\n{keywords[lang]['answer']}: <ANSWER> {chr(65 + q['answer'])} </ANSWER>\n"
                    "\n---\n"
                )

    # Add the main question and options
    prompt += (
        f"\n{instruction[lang]}\n"
        f"\n{keywords[lang]['question']}: {question['question']}\n"
        f"{keywords[lang]['options']}:\n"
    )
    for t, option in enumerate(question["options"]):
        prompt += f"{chr(65 + t)}. {option}\n"
    prompt += f"\n{keywords[lang]['answer']}:"

    # Construct the final message
    if question["image"] is not None:
        images = [question["image"]]
        message = (
            f"<|im_start|>system\n{system_message[lang]}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        message = (
            f"<|im_start|>system\n{system_message[lang]}<|im_end|>\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        images = None

    return message, images


def create_qwen_prompt_vllm(question, method, few_shot_samples):
    # Determine the placeholder for images
    lang = question["language"]
    prompt = ""
    # Add the main question and options
    prompt += (
        f"\n{INSTRUCTION}\n"
        f"\n{keywords[lang]['question']}: {question['question']}\n"
        f"{keywords[lang]['options']}:\n"
    )
    for t, option in enumerate(question["options"]):
        prompt += f"{chr(65 + t)}. {option}\n"
    prompt += f"\nANSWER:"

    # Construct the final message
    if question["image"] is not None:
        images = [question["image"]]
        message = (
            f"<|im_start|>system\n{SYS_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        message = (
            f"<|im_start|>system\n{SYS_MESSAGE}<|im_end|>\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        images = None

    return message, images


# Deep-seek
def create_deepseek_prompt(question, method, few_shot_samples):
    content = []
    lang = question["language"]
    lang_keyword = keywords[lang]
    prompt = []
    if question["image"] is not None:
        prompt.append("<image>\n")
        images = [question["image"]]
    else:
        images = None

    if method == "few-shot":
        few_shot = few_shot_samples.get(lang, [])
        prompt.append(f"\n{few_shot_instruction[lang]}\n")
        for q in few_shot:
            prompt.append(
                f"\n{lang_keyword['question']}: {q['question'].replace('<image>', '')} \n{lang_keyword['options']}: \n"
            )
            for t, option in enumerate(q["options"]):
                index = f"{chr(65+t)}. "
                prompt.append(f"{index}) {option.replace('<image>', '')}\n")
            prompt.append(
                f"\n{lang_keyword['answer']}: <ANSWER> {chr(65+q['answer'])} </ANSWER>\n"
            )
            prompt.append("\n---\n")

    prompt.append(f"\n{instruction[lang]}\n")
    prompt.append(
        f"\n{lang_keyword['question']}: {question['question'].replace('<image>', '')} \n{lang_keyword['options']}: \n"
    )
    for t, option in enumerate(question["options"]):
        index = f"{chr(65+t)}. "
        prompt.append(f"{index}) {option.replace('<image>', '')}\n")
    prompt.append(f"\n{lang_keyword['answer']}:")
    message = [
        {"role": "<|User|>", "content": "".join(prompt), "images": images},
        {"role": "<|Assistant|>", "content": ""},
    ]
    return [system_message[lang], message], None


# GPT

# Gemini

# Claude


# Aya-Vision
def create_aya_prompt2(question, method, few_shot_samples):
    lang = question["language"]
    lang_keyword = keywords[lang]
    prompt = []
    content = []
    # zero shot
    if question["image"] is not None:
        with open(question["image"], "rb") as img_file:
            img = Image.open(img_file).resize((512, 512))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")

            base64_image_url = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": base64_image_url},
            },
        )
    prompt.append(f"\n{instruction[lang]}\n")
    prompt.append(f"\n{keywords[lang]['question']}: {question['question']}\n")
    prompt.append(f"{keywords[lang]['options']}:\n")
    for t, option in enumerate(question["options"]):
        index = f"{chr(65+t)}. "
        prompt.append(f"{index}) {option}\n")
    prompt.append(f"\n{lang_keyword['answer']}:")
    prompt = "".join(prompt)
    content.append({"type": "text", "text": prompt})
    message = [
        {"role": "system", "content": system_message[lang]},
        {"role": "user", "content": content},
    ]
    return message, None


def create_aya_prompt(question, method, few_shot_samples):
    lang = question["language"]
    lang_keyword = keywords[lang]
    prompt = []
    content = []
    # zero shot
    if question["image"] is not None:
        with open(question["image"], "rb") as img_file:
            img = Image.open(img_file).resize((512, 512))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")

            base64_image_url = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": base64_image_url},
            },
        )
    prompt.append(f"\n{INSTRUCTION}\n")
    prompt.append(f"\n{keywords[lang]['question']}: {question['question']}\n")
    prompt.append(f"{keywords[lang]['options']}:\n")
    for t, option in enumerate(question["options"]):
        index = f"{chr(65+t)}. "
        prompt.append(f"{index}) {option}\n")
    # prompt.append(f"\n{lang_keyword['answer']}:")
    prompt = "".join(prompt)
    content.append({"type": "text", "text": prompt})
    message = [
        {"role": "system", "content": SYS_MESSAGE},
        {"role": "user", "content": content},
    ]
    return message, None
