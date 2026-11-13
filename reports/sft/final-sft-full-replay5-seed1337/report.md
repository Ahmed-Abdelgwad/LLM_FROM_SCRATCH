# Fine-tuning evaluation: runs/sft-full-replay5-seed1337/merged.pt

| Variety | Test pairs | Response bits/char | Stopped | Right language | Empty | Looping | Mean new tokens |
|---|---|---|---|---|---|---|---|
| msa | 940 | 1.690 | 52% | 99% | 0% | 59% | 139 |
| egyptian | 26 | 1.837 | 58% | 96% | 0% | 46% | 119 |
| english | 405 | 1.186 | 69% | 100% | 0% | 55% | 120 |

## Samples (greedy)

### msa
- **Prompt:** قم بإنشاء قصيدة عتاب من بحر الطويل.
  - **Reference:** وَلَمّا بَدا لي أَنَّ ما كُنتُ أَرتَجي
مِنَ الأَمرِ وَلّى بَعدَما قُلتُ أَقبَلا
تَلَوَّمتُ بَينَ اللَومِ وَالعُذرِ ساعَةً
كَذي الوِردِ يُرمى قَبلَ أَن يَتَبَدَّلا
فَلَمّا رَأَيتُ الحِلمَ قَد طارَ طَير
  - **Reply:** وَأَخْذَ أَمْكَ أَرْفُ
وَأَخْذَ أَرْفُ
وَكَذَ أَرْفُ
وَكَذَ أَرْفُ
وَكَذَ أَرْفُ
وَكَذَ أَرْفُ
وَكَذَ أَرْفُ
وَكَذَ أَرْفُ
وَكَذَ أَرْفُ
وَكَذَ أَرْفُ
وَكَذَ أَرْفُ
وَكَذَ أَرْفُ
وَكَذَذَ أَرْفُ
وَكَذَذَ أَرْفُ
وَكَذَذَ أَرْفُ
وَكَذَذَ أَرْفُ
وَكَذَذَ أَرْفُ
وَكَذَذَ الشُّقُ
وَكَذَذَ الشُّقُ
وَكَذَذ [cut]
- **Prompt:** أريد إنشاء قصيدة حزينة من بحر المنسرح. هل يمكنك مساعدتي؟
  - **Reference:** خُذها سُطوراً إِلَيكَ قَد بُعِثَت
تَروم للنَفس ما يُعللها
في طَي بيضا ظَلَت مِن وَلَهٍ
فيكَ بِأَيدي اللحاظ أَصقلها
أَكتُبها وَالدُموع تَنقطها
بِعبرة لا أَزال أُهملها
لَو كانَ ظَني إِذا بصرت بِها
نِياب
  - **Reply:** أَنتَ في قِبلِ الأَخْوِ
وَأَنتَ في قِبلِ الأَخْوِ
وَتَنتَ في قِبلِ الأَخْوِ
وَتَنتَ في قِبلِ الأَخْوِ
وَتَنتَ في قِبلِ الأَخْوِ
وَتَنتَ في قِبلِ الأَخْوِ
وَتَنتَ في قِبلِ الأَخْوِ
وَتَنتَ في قِبلِ الأَخْوِ
وَتَنتَ في قِبلِ الأَخْوِ
وَتَنتَ في قِبلِ الأَخْوِ
وَتَنتَ في قِبلِ الأَخْوِ
وَتَنتَ في قِبلِ [cut]
- **Prompt:** ما أثر ذلك على الأغذية والإنتاج الغذائي في البلد المتضرر؟
  - **Reference:** إن أثر النشاط الإشعاعي على الأغذية والإنتاج الغذائي في بلد متضرر من طارئة إشعاعية أو نووية سيتوقف على أنواع النويدات المشعة ومقدار الإشعاع المترسب أو المنتشر في مكان إنتاج الأغذية أو حصادها. وعلى الرغ
  - **Reply:** الزراعة هي مصدر رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئيسي رئ
- **Prompt:** ما هي الألياف البصرية؟
  - **Reference:** تعرف الألياف البصرية او الضوئية بأنها التقّنية المرّتبطة بنقل البيانات باستخدام نبضات ضوئيّة تنتقل جنبًا إلى جنب مع ألياف طويلة مصنوعة عادةً من البلاستيك أو الزجاج، وقد تمّ تصميمها بِحيّث تُسهِل انتشا
  - **Reply:** الألياف البصرية هي الألياف البصرية.
- **Prompt:** كمل الفقرة:

على أن الخليفة عثمان بن عفان (ت 35هـ/656م) لم يكن أقل حذرًا من عمر في ركوب البحر؛ فحين أرسل إليه واليه على الشام معاوية بن أبي سفيان (ت 60هـ/681م) يستأذنه في غزو قبرص سنة 27هـ/649م ويهوّن
  - **Reference:** فعل مع الخليفة عمر الذي رفض طلبه- أجابه عثمان بشروط قائلا: "فإن ركبتَ البحرَ ومعكَ امرأتك فاركبه مأذونًا لك، وإلا فلا..؛ خيّرهم (= جنوده) فمن اختار الغزو طائعًا فاحمله وأعنْه"؛ كما يروي البلاذري (ت 27
  - **Reply:** أحد هذه العبارة:

1. أبو بكر بن عفان (ت 48ه/656م)
2. أبو بكر بن عفان (ت 48ه/656م)
3. أبو بكر بن عفان (ت 48ه/656م)
4. أبو بكر بن عفان (ت 48ه/656م)
5. أبو بكر بن عفان (ت 48ه/656م)
6. أبو بكر بن عفان (ت 48ه/656م)
7. أبو بكر بن عفان (ت 48ه/656م)
8. أبو بكر بن عفان (ت 48ه/656م)
9. أبو بكر بن عفان (ت 48ه/ [cut]
- **Prompt:** اكمل الجملة التالية:

وهو ما قد يُسبب حالة من الشعور بالذنب، لا تخلو من تداعيات جسدية ونفسية لدى الأشخاص
  - **Reference:** المشغولين لدرجة لا تُمكّنهم من ممارسة الرياضة بشكل منتظم، وبالكاد يستطيعون ممارستها في عطلة نهاية الأسبوع فقط.
  - **Reply:** بالتأكيد، لا تلبي أي أعراض، لا تلبي أي أعراض، لا تلبي أي أعراض.

### egyptian
- **Prompt:** ايه العلاقة بين الجملتين دول، توافق - تناقض - مفيش علاقة:

الجملة الأولى: رياض بيلبس بسرعة عشان ينزل جرى و يلحق الأتوبيس.
الجملة التانية: رياض متأخر عن ميعاد ما.
  - **Reference:** خلينا نشوف كده مع بعض:

- لو رياض مكنش وراه ميعاد متأخر عليه مكنش هينزل جرى عشان يلحق الأتوبيس.
- الهرولة عامة سمة المتأخر.

عشان كده ممكن نقول ان الجملة التانية علاقتها علاقة توافق بالجملة الأولى.
  - **Reply:** الجملة الأولى: رياض متأخر عن ميعاد ما هي الجملة الأولى.

اللى بتتغير في الجملة الأولى.

اللى بتتغير في الجملة الأولى.

اللى بتتغير في الجملة الأولى.

اللى بتتغير في الجملة الأولى.

اللى بتتغير في الجملة الأولى.

اللى بتتغير في الجملة الأولى.

اللى بتتغير في الجملة الأولى.

اللى بتتغير في الجملة الأو [cut]
- **Prompt:** ممكن تقولي أيه الفرق بين بروس ويليس و بروس واين.
  - **Reference:** بروس ويليس ممثل أمريكى مشهور من مواليد 19 مارس 1955 في مدينة ايدار-اوبرشتاين. حصل على جايزة الإيمي مرتين كأفضل ممثل رئيسي في مسلسل درامي سنة 1987 عن دوره في مسلسل ضوء القمر، وسنة 2000 جايزة ضيف الشرف 
  - **Reply:** بالتأكيد. في حاجة تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية  [cut]
- **Prompt:** اقرأ الكام جملة دول و بعدين جاوبلي على السؤال الي جاي:

شوية أرانب كانوا بيعبروا الطريق السريع. مصطفى كان سايق براحته و أخر حاجة كان ممكن يتوقعها هى أرنب على الطريق. يا دوبك قدر يفرمل على أخر أرنب فيه
  - **Reference:** أنا يسعدنى أجاوبك على الأسئلة دى.

- بالنسبة لأو حته فالسؤال، فجملة "بص له" بتتكلم علي بصة الأرنب لمصطفى، علشان كده "له" هنا بتتكلم على مصطفى.
- نيجى دلوقتى للحتة التانية من السؤال، للأسف مقدرش أقولك 
  - **Reply:** اللى كان سايق براحته و كان سايق براحته و كان سايق براحته و كان سايق براحته و كان سايق براحته و كان سايق براحته و كان سايق براحته و كان سايق براحته و كان سايق براحته و كان سايق براحته و كان سايق براحته و كان سايق براحته و كان سايق براحته و كان سايق براحته و كان سايق براحته و كان سايق براحته و كان ساي [cut]
- **Prompt:** اتلانتيك ليه؟ وبالنظر للسؤال السابق، اكتب سياق يحتوي على الإجابة. ممكن يكون من 1 - 20 جملة. السياق:
  - **Reference:** خلال منتصف العصر الايوسيني، بيعتقد إن حوض الصرف الصحي في الأمازون تم تقسيمه على طول وسط القارة بواسطة قوس بوروس. المياه في الجانب الشرقي تتدفق باتجاه المحيط الأطلسي، في حين تتدفق المياه في الغرب باتجا
  - **Reply:** بالتأكيد. الإجابة هي:

1. ما هي الجملة التي تثير الجدل؟
2. ما هي الجملة التي تثير الجدل؟
3. ما هي الجملة التي تثير الجدل؟
4. ما هي الجملة التي تثير الجدل؟
5. ما هي الجملة التي تثير الجدل؟
6. ما هي الجملة التي تثير الجدل؟
7. ما هي الجملة التي تثير الجدل؟
8. ما هي الجملة التي تثير الجدل؟
9. ما هي الجم [cut]
- **Prompt:** ممكن تقولي أيه الفرق بين بروس ويليس و بروس واين.
  - **Reference:** بروس ويليس ممثل أمريكى مشهور من مواليد 19 مارس 1955 فى ايدار-اوبرشتاين. حصل على جايزة الإيمي مرتين كأفضل ممثل رئيسي في مسلسل درامي سنة 1987 عن دوره في مسلسل ضوء القمر، وسنة 2000 جايزة ضيف الشرف البارز
  - **Reply:** بالتأكيد. في حاجة تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية تانية  [cut]
- **Prompt:** اديني مثال على معلومة في الفئة دي: ألعاب الفيديو
  - **Reference:** فى حوالى 8 خصوم بيحاربوا ميجامان فى كل لعبة.
  - **Reply:** الاجابة هي: لعبة ألعاب الفيديو.

### english
- **Prompt:** Predict the dependency parsing of the following statement:
"I love apples"

  - **Reference:** In this dependency parse tree, "love" is the root of the sentence, indicating that it is the main verb. "I" is the subject of the sentence, and "apples" is the direct object of the verb "love." 
  - **Reply:** I love apples, like the parsing of the world's first-ever individual peaceوم.
- **Prompt:** Question: What will happen if all the deers are killed in the given food chain?

Grass — Deer — Lion


a. The population of Lion increases
b. The population of grass decreases
c. The population of Lio
  - **Reference:** Answer: (c)

If all the deer are killed in the given food chain, there won’t be any food available for the lions. Hence, the lions will starve to death and their population will decrease. On the other
  - **Reply:** Answer: (c)

The population of Lion decreases and grass is diverse. The population of lions is about 1.5 million. The population of lions is about 1.5 million. The population of Lion decreases and grass is about 1.5 million. The population of Lion decreases and grass is about 1.5 million. The popula [cut]
- **Prompt:** Explain the types of projects in COCOMO 81.
  - **Reference:** COCOMO 81 distinguishes among Organic (small teams), Semi-Detached (mixed experience), and Embedded (tight constraints) projects.
  - **Reply:** COCOMO 81 is a city that is a key component of the project. It is the largest city in the world, and the largest city in the world's population.

COCOMOO 81 is the largest city in the world. It is the largest city in the world, and the largest city in the world's population. It is the world's larges
- **Prompt:** How does DevOps contribute to the software development process?


  - **Reference:** DevOps combines development and operations, promoting collaboration and automation to achieve faster development cycles, continuous integration, and continuous deployment.
  - **Reply:** The DevOps is a process that involves automating and maintaining and maintaining and maintaining and maintaining and maintaining and maintaining and maintaining and maintaining and maintaining and maintaining.
- **Prompt:** How is Buddhism practiced in Sri Lanka, and what are some important Buddhist sites?
  - **Reference:** Buddhism plays a central role in Sri Lankan culture and daily life. Theravada Buddhism is the predominant practice, influencing rituals, customs, and festivals. The Temple of the Tooth Relic in Kandy 
  - **Reply:** Buddhist sites are used to create a positive impact on the environment, including the need for forskilled spaces, and the need for forskilled spaces.
- **Prompt:** The "Technical Skills" secition is is not included in the top of the resume.

is there any grammatical error?
  - **Reference:** Yes, there is a minor grammatical error in your sentence. It should be:

"The 'Technical Skills' section is not included at the top of the resume."

In this corrected sentence, I removed the double "i
  - **Reply:** Your sentence is a grammatical error that is used to ensure that the resume is not included in the next level.

