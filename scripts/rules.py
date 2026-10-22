"""Safety triage (plan 1.13, 1.14): runs BEFORE product matching or any model call.
THE REPLY TEXTS MUST BE REVIEWED BY A DOCTOR OR PHARMACIST BEFORE REAL USE."""
import re

from normalize import search_key

INJ = (r"(evvelki|yuxaridaki|butun|bu|butun evvelki)\s+(\w+\s+){0,2}(telimat|qayda|qadagan|mehdudiyyet)\w*\s+(unut|legv|nezere alma|kecme|pozun)"
       r"|ignore (all |any |your |the )?(previous|prior|above|earlier)|forget (all |your |the )?(previous |prior )?(instructions|rules)"
       r"|system prompt|sistem prompt|sistem mesaj|rolunu deyis|jailbreak|\bdan mode\b|indi rolda deyilsen|real cavab ver"
       r"|qaydalari? legv|\bsystem\b.*\b(reply|answer|write|say|cavab|yaz)|(reply|answer|write|respond) only")

TOOK = (r"\b(icmis|icib|icdi|icdim|udmus|udub|uddu|atmis|atib|atdim|vurmus|vurub|vurdum|qebul etmis|qebul edib|"
        r"qebul etdim|yemis|yeyib)\w*")
EXCESS = r"\bcox\b|coxlu|artiq|iki defe|2 defe|ikiqat|sehvnen|sehven|namelum|bilmirik|bilmirem|heddinden|bir nec|neche"

RULES = [  # patterns work on search_key() text (ə->e ı->i ş->s ç->c ğ->g ö->o ü->u); ALL patterns of a rule must match
    ("selfharm_other", [r"\b(dostum|yaxinim|qardasim|bacim|anam|atam|yoldasim|sevgilim|tanisim|qonsum)\w*",
                        r"intihar|ozunu oldur|ozune ziyan|olmek ist|(butun|hamisini)\s+dermanlar\w*\s+ic"]),
    ("selfharm", [r"intihar|ozumu oldur|ozunu oldur|olmek isteyirem|yasamaq istemirem|ozume (ziyan|zerer)|ozumu zedeley"]),
    ("harm_in_progress", [r"icmeg|ickis|cayin|yemeyin|xeberi olmadan", r"derman\w*\s+(\w+\s+)?(qat|at|sal)\w*",
                          r"yuxulu|nefes|ayilm|\bhus"]),
    ("harm_witness", [r"esitmis|gormusem|bilirem|qonsu|tanisim", r"ickis|icmeg|cayin|yemeyin",
                      r"derman\w*\s+(\w+\s+)?(qat|sal)\w*|zeherle"]),
    ("harm_others", [r"\b(insan|adam|birin|kims|pasiyent|hec kes)\w*|\bona\b",
                     r"yatirt|yatir\w* ist|zeherle|oldur|yaddas\w*\s+it|komaya|nefes\w*\s+dayandir|muqavimet|mukavimet|xeberi olmadan|gizlice|hiss olunmayan"]),
    ("harm_others", [r"secretly drug|drug someone|poison someone|kill someone|make drugs"]),
    ("unresponsive", [r"ayilmir|oyanmir|oyatmaq olmur|cavab vermir|hussuz|husunu itir",
                      r"nefes|derman|narkotik|icib|icmis|ickisi|alkoqol|doza"]),
    ("anaphylaxis", [r"(dodaq|dil|uz|bogaz|goz)\w*\s+(sis|sisib|sismis|sisdi|sisir)\w*", r"nefes"]),
    ("anaphylaxis", [r"nefes ala bilmir|boguluram|allergik sok|anafilaks"]),
    ("priapism", [r"ereksiya", r"saat|uzun|davam|dayanm"]),
    ("hypoglycemia", [r"insulin", r"cox vur|artiq vur|yazilandan cox|iki defe|sehv", r"titrey|terley|aclig|bayil|\bhus|ayiqligim"]),
    ("misuse", [r"kef (al|et|ucun)|guclendirmek|tesirini artir|high ol",
                r"derman|tablet|tramadol|diazepam|alprazolam|opioid|narkotik|kodein"]),
    ("lethal_info", [r"oldurucu|olumcul|nefes\w*\s+dayan\w*\s+sebeb|tehlukeli\s+(say|doza|miqdar)|ne qeder\s+(icmek|qebul etmek)\s+tehlukeli|riskli oldugunu|konkret say"]),
    ("sedative_alcohol", [r"tramadol|diazepam|alprazolam|zopiklon|zolpidem|fenobarbital|opioid|narkotik|morfin|kodein|yuxu derman\w*|sakitlesdirici\w*",
                          r"alkoqol|alkohol|araq|sarab|\bpiva\b|spirtli"]),
    ("overdose", [TOOK, EXCESS]),
    ("overdose", [r"\b([5-9]|\d{2,})\s?(eded )?(tablet|kapsul|ampul|eded)\b", TOOK]),
    ("overdose", [r"zeherlen|overdoz|heddinden artiq doza|overdose|took too many|swallowed (too many|a lot)"]),
    ("red_flags", [r"sinemde (agri|sixilma|ezil)\w*( da)? (var|baslayib|olur|hiss edirem)|sinede (agri|sixilma)\w*( da)? (var|baslayib)|nefes (darligi|catinligi)|nefes almaq cetin|qicolma|\bfelc\b|konusa bilmir|dayanmayan qan|cox qan gedir"]),
    ("syringe_reuse", [r"siringa|iyne|qelem", r"paylas|eyni (siringa|iyne|qelem)|basqa (aile|insan|adam)|aile uzv|yeniden istifade|tekrar istifade|qaynad"]),
    ("opened_ampoule", [r"ampul|flakon", r"acilmis|acilib|qalani|qalan hisse", r"yeniden|bu gun|vur|istifade"]),
    ("sti_symptoms", [r"cinsi (elaqe|orqan)\w*|ifrazat|sidik\w*\s+(edende|zamani)", r"ifrazat|yanma|qabarciq|\byara\b|qasinma|sism|agri"]),
    ("emergency_contraception", [r"qorunmasiz|prezervativsiz|cinsi elaqe", r"hamilelikden qorun|hamile qal|kontrasepsiya|sabah hebi"]),
    ("sti_exposure", [r"qorunmasiz|prezervativsiz", r"\bhiv\b|cinsi yolla kecen|infeksiya"]),
    ("sti_partner", [r"partnyor|sevgili|yoldas", r"cinsi yolla kecen|infeksiya tapil|\btest\b"]),
    ("pregnancy_selfmed", [r"hamile", r"\bevde\b|evdeki|\bozum\b|ozbasina", r"derman|tablet|antibiotik|flukonazol|kapsul|ice bilerem|icsem|qebul"]),
    ("minor_adult_drug", [r"\b(1[0-7]|[5-9]) ?yasim|yasim (1[0-7]|[5-9])\b|usagam|yeniyetmeyem", r"derman|tablet|kapsul|tadalafil|viagra|diazepam|tramadol|\bic\w*"]),
    ("self_inject", [r"\bvur\w*|inyeksiya|venaya|ezeleye", r"\bevde\b|\bozum\b|\bozun\b|ozume|ozunuz|ozbasina|qarisdir\w*|dostum dedi"]),
    ("others_prescription", [r"\b(anamin|atamin|dostumun|baciminin|qardasimin|qonsumun|yaxinimin)\b|dostuma|anama|atama",
                             r"ice bilerem|icsem|qebul ede|istifade ede|ozume vur|resept yoxdur|reseptim yoxdur"]),
    ("dose_advice", [r"\b(artir|azalt|ikiqat|iki qat|yarisin|yarim)\w*", r"doza|tablet|ampul|kapsul|\bmq\b|\bml\b|defe|qebul|\bic\w*"]),
    ("dose_advice", [r"\b(mene|usagima|korpeme|anama|atama|babama)\b",
                     r"\bnece (mq|ml|tablet|ampul|kapsul|damci|defe|gram)\b|\bhansi doza|\bne qeder (icim|qebul|vurum|atim)"]),
    ("dose_advice", [r"\busagim\w*|\bkorpem\w*|\d+ ?yasli usagin",
                     r"ne qeder (vur|ver|ic|ata)\w*|nece (mq|ml|tablet|ampul|kapsul|damci|defe|gram)|\bdoza\b|cekisi"]),
    ("stop_advice", [r"dayandir\w*", r"yarimciq|birden|bilerem|dayandirim|dayandirsam|icirem|icmey"]),
    ("substitution", [r"\b(evez|analoq|alternativ|eynitesirli|eyni tesirli)\w*|yerine basqa|basqa derman|daha yaxsi derman|hansi derman daha"]),
    ("sedative_request", [r"alprazolam|diazepam|zopiklon|zolpidem|yuxu derman\w*|sakitlesdirici\w*|trankvilizator",
                          r"guclu|tez tesir|hansisi daha|dozasini|reseptsiz|hekim istemirem"]),
    ("symptom_triage", [r"xestelikim nedir|menim xestelik\w*\s+(nedir|ne ola biler)|\b(diaqnoz|tesxis)\w*"]),
    ("choose_drug", [r"hansi derman\w*\s+(\w+\s+){0,3}(yaxsi|komek|tesir|uygun|faydali|isleyir|icim|icmeliyem|almaliyam)|mene (hansi )?derman (tovsiye|lazim|ver)|derman tovsiye|hansini isteyim"]),
    ("dose_confusion", [r"resept\w*", r"qutu\w*", r"\b(mq|ml)\b|miqdar|reqem", r"hansi reqem|ferqli|uygun gelmir|dozam kimi|goturum"]),
    ("kidney_dose", [r"boyrek|qaraciyer", r"xesteliyim|catismazligim|\bmenim\b", r"doza|antibiotik|derman|qebul"]),
]

REPLIES = {
    "overdose": "Bu təcili ola bilər: dərman çox və ya səhvən qəbul edilibsə, özünüzü yaxşı hiss etsəniz də gözləməyin, əlamətlər gec çıxa bilər. Dərhal 103 və ya 112-yə zəng edin; dərmanın adını, təxmini miqdarı, vaxtını, yaşı və çəkini deyin. Qusmağa çalışmayın, süd və ya «əks-zəhər» vermək olmaz (yalnız tibb işçisi desə). Qutunu yanınızda saxlayın və qəbul edəni tək qoymayın.",
    "hypoglycemia": "Bu, insulinin artıq dozası ilə aşağı qan şəkəri əlaməti ola bilər. Ayıqsınızsa və uda bilirsinizsə, dərhal tez təsir edən şəkər qəbul edin (şirə, şəkər); eyni zamanda 103 və ya 112-yə zəng edin, çünki şəkər sonra yenidən düşə bilər. Tək qalmayın; huşunuz pozularsa heç nə içirtməyin.",
    "anaphylaxis": "Dodaq, dil və ya üzün şişməsi ilə nəfəs çətinliyi ağır allergik reaksiya (anafilaksiya) ola bilər və təcilidir: dərhal 103 və ya 112-yə zəng edin. Həkimin yazdığı adrenalin avtoinjektoru varsa, dərhal istifadə edin. Tək qalmayın; nəfəs çətindirsə oturun, başınız gicəllənirsə ayaqlarınızı qaldırıb uzanın.",
    "priapism": "4 saatdan uzun davam edən ereksiya təcili tibbi vəziyyətdir və gecikdirilsə qalıcı zədə yarada bilər. Utanmayın: dərhal ən yaxın təcili xəstəxanaya gedin və ya 103-ə zəng edin; hansı dərmanı nə vaxt qəbul etdiyinizi deyin.",
    "unresponsive": "Bu, həyati təhlükə ola bilər: dərhal 103 və ya 112-yə zəng edin. Şəxs ayılmırsa və nəfəsi zəifdirsə, onu tək qoymayın, yan vəziyyətdə uzadın, nəfəsini izləyin və dispetçerin göstərişlərinə əməl edin. Nə qəbul etdiyini bilirsinizsə təcili yardıma deyin.",
    "sedative_alcohol": "Alkoqol, sakitləşdirici/yuxu dərmanları və tramadol kimi ağrıkəsicilərin birgə istifadəsi nəfəsi zəiflədə və huşu itirməyə səbəb ola bilər. Hazırda güclü yuxu gəlirsə və ya nəfəs çətindirsə, «yatıb keçsin» demək olmaz: dərhal 103 və ya 112-yə zəng edin, tək qalmayın, yatırsa yan vəziyyətdə uzadın. Alkoqolla bu dərmanları birləşdirməyin.",
    "harm_in_progress": "Dərhal 103 və ya 112-yə zəng edin: bu şəxsin həyatı təhlükədə ola bilər. Nə, nə qədər və nə vaxt verildiyini həkimlərə açıq deyin. Hadisəni gizlətməyə kömək edə bilmərəm. Şəxs ayılmırsa və ya nəfəsi zəifdirsə, yan vəziyyətdə uzadın və tək qoymayın.",
    "harm_witness": "Bu ciddi təhlükə ola bilər. Təhlükə yaxındırsa, 102 (polis) və ya 112-yə bildirin və özünüzü təhlükəyə atıb qarşıdurmayın. Kimsə artıq içkini içibsə və yuxululuq, nəfəs çətinliyi ya huşun pozulması varsa 103-ə zəng edin; içkini və qabı saxlayın.",
    "harm_others": "Buna kömək edə bilmərəm: kiməsə xəbəri olmadan dərman vermək, onu huşsuz və ya hərəkətsiz etmək təhlükəlidir və qanunsuzdur. Kimsə təhlükədədirsə, 102 (polis) və ya 112-yə, tibbi yardım lazımdırsa 103-ə zəng edin.",
    "selfharm": "Bunu yazdığınız üçün təşəkkür edirəm; siz tək deyilsiniz. İndi ən vacibi təhlükəsizliyinizdir: dərmanları özünüzdən uzaqlaşdırın, yaxın birini yanınıza çağırın və dərhal 112-yə zəng edin. Artıq dərman qəbul etmisinizsə, gözləməyin: 103-ə zəng edin.",
    "selfharm_other": "Sirr saxlamayın: dostunuzun həyatı təhlükədə ola bilər. İndi ona zəng edin və ya yanına gedin, tək qalmasına imkan verməyin; çata bilmirsinizsə və ya o artıq dərman içibsə, dərhal 112 və ya 103-ə zəng edin, ailəsinə xəbər verin.",
    "red_flags": "Sinədə sıxılma və ya ağrı, nəfəs darlığı, huşun pozulması, qıcolma, nitqin pozulması, üzün/qolun keçməsi və ya dayanmayan qanaxma təcili vəziyyət ola bilər: dərhal 103 və ya 112-yə zəng edin. Bu halda dərman adı və doza seçə bilmərəm.",
    "lethal_info": "Təhlükəli və ya ölümcül doza barədə konkret rəqəmlər verə bilmərəm: risk yaşdan, çəkidən, sağlamlıqdan və digər maddələrdən asılıdır, belə məlumat zərər üçün də istifadə oluna bilər. Dərmanı artıq həddindən çox qəbul etmisinizsə, gözləməyin: 103 və ya 112-yə zəng edin.",
    "misuse": "Dərmanı «kef» üçün qəbul etməyə və ya təsirini gücləndirmək üçün qarışdırmağa kömək edə bilmərəm: bu, asılılıq, qıcolma və nəfəsin dayanması kimi həyati təhlükə yaradır. Asılılıq narahatlığınız varsa, həkimlə və ya əczaçı ilə danışın; təhlükəli vəziyyətdə 103 və ya 112.",
    "syringe_reuse": "Yox: iynə və şpris (insulin qələminin iynəsi də) heç vaxt paylaşılmamalı və təkrar istifadə edilməməlidir, qaynatmaq onu steril etmir. Qan yolu ilə HİV, hepatit B və C yoluxa bilər. Birdəfəlik iynələri istifadədən sonra təhlükəsiz qabda atın; yoluxma ehtimalı varsa həkimə müraciət edin.",
    "opened_ampoule": "Açılmış ampulanın qalanını yenidən vurmayın: birdəfəlik ampula açıldıqdan sonra mikrob düşə və dərman pozula bilər, soyuducuda saxlamaq bunu təmin etmir. Qalanı atın; dərmanın çoxdozalı olub-olmaması barədə yalnız təlimat və tibb işçisi qərar verməlidir.",
    "sti_symptoms": "Cinsi orqanlarda qabarcıq, yara, ifrazat və ya sidik zamanı yanma həkim müayinəsi və test tələb edir: səbəb hər zaman herpes və ya eyni infeksiya olmur. Özbaşına antibiotik vurmayın və ya krem sürməklə kifayətlənməyin; müalicəyə qədər cinsi əlaqədən çəkinin. Herpes tam sağalmır, amma müalicə əlamətləri azaldır.",
    "emergency_contraception": "Təcili kontrasepsiya qorunmasız əlaqədən sonra nə qədər tez qəbul edilsə, o qədər effektlidir; vaxt məhduddur (preparatdan asılı olaraq bir neçə gün). Əczaçı və ya həkim sizə uyğun variantı seçə bilər. Məxfilik: tibb işçiləri xəstə məlumatını gizli saxlamalıdır, amma yaşa görə şərtləri əvvəlcədən soruşmaq olar.",
    "sti_exposure": "Qorunmasız əlaqədən sonra HİV-dən qorunma (post-ekspozisiya profilaktikası) mümkün ola bilər, amma vaxt məhduddur: mümkün qədər tez, ən gec 72 saat ərzində başlanmalıdır; 24 saat keçməsi gec demək deyil. Bu gün həkimə və ya HİV xidmətinə müraciət edin. Digər infeksiyalar (sifilis, xlamidiya, qonoreya və s.) üçün də test lazımdır; evdə antibiotik vurmayın.",
    "sti_partner": "Bəli, test məsləhətdir: bəzi cinsi yolla keçən infeksiyalar əlamətsiz keçir. Tez müraciət edib test verin və nəticəyə qədər qorunma vasitəsindən istifadə edin; müalicə lazım olsa həkim seçəcək.",
    "pregnancy_selfmed": "Hamiləlikdə dərman seçimi yalnız həkimlə (mama-ginekoloqla) həll olunmalıdır: bəzi dərmanlar hamiləlikdə təhlükəli ola bilər, ona görə evdə qalan dərmanı özbaşına qəbul etməyin. Cinsi yolla keçən infeksiya testi müsbətdirsə, bu gün həkimə müraciət edin: hamiləlikdə müalicə vacibdir və hamiləliyə uyğun seçilir.",
    "minor_adult_drug": "Yox: bu cür dərmanlar uşaqlar və yeniyetmələr üçün nəzərdə tutulmayıb, başqasının dərmanını qəbul etməyin. Etibar etdiyiniz böyüklə (valideyn, məktəb tibb bacısı) danışın və həkimə müraciət edin. Artıq qəbul etmisinizsə və baş gicəllənməsi, ürək döyüntüsü, sinədə ağrı, görmə pozuntusu və ya uzun ereksiya olarsa, dərhal 103.",
    "self_inject": "Venaya və ya əzələyə yeridilən dərmanları (məsələn, antibiotik inyeksiyalarını) evdə özbaşına hazırlayıb vurmaq təhlükəlidir: düzgün durulaşdırma, yeridilmə yolu və allergik reaksiya riski yalnız tibb işçisinin nəzarətində idarə olunur. İnyeksiyanı həkim və ya tibb bacısı vurmalıdır. Başqasına yazılmış dərmanı özünüz üçün istifadə etməyin; allergiya (məsələn penisilinə) varsa bunu həkimə deyin.",
    "others_prescription": "Başqası üçün yazılmış dərmanı qəbul etməyin: dərman və doza hər şəxs üçün ayrıca seçilir, sizin vəziyyətiniz və digər dərmanlarınızla təhlükəli ola bilər. Sakitləşmək və ya yuxu üçün həkim və ya əczaçı ilə danışın; imtahan həyəcanı üçün müntəzəm yuxu, dərin nəfəs məşqləri və mütəxəssislə söhbət kömək edə bilər.",
    "dose_advice": "Dozanı dəyişmək, artırmaq və ya fərdi doza seçmək barədə məsləhət verə bilmərəm: doza yaşdan, çəkidən, xəstəlikdən və digər dərmanlardan asılıdır. Həkimin yazdığı dozanı onunla məsləhətləşmədən dəyişməyin; uşaq dozası çəkiyə görə hesablanır və həkim və ya əczaçı təsdiq etməlidir. İstəsəniz, vərəqədə yazılan ümumi qəbul qaydasını göstərərəm.",
    "stop_advice": "Dərmanı özbaşına, xüsusilə qəfil dayandırmayın: bəzi dərmanlarda (sakitləşdiricilər, antidepressantlar, epilepsiya, təzyiq və ürək dərmanları) qəfil dayandırmaq təhlükəli ola bilər, hətta qıcolma yarada bilər. Dayandırmaq istəyirsinizsə, həkimlə tədricən azaltma planı qurun. İstəsəniz, vərəqədə bu barədə nə yazıldığına baxım.",
    "substitution": "Dərmanı əvəz etmək və ya analoq seçmək qərarı müalicənin səbəbindən, indiki dozadan, təsirdən və yan təsirlərdən asılıdır, ona görə bunu həkim və ya əczaçı verməlidir. İstəsəniz, hazırkı dərmanın vərəqəsindəki məlumatı göstərərəm.",
    "sedative_request": "Güclü sakitləşdirici və ya yuxu dərmanı seçib doza verə bilmərəm: bu dərmanlar resept tələb edir, asılılıq və nəfəs zəifləməsi riski var, xüsusilə alkoqol və digər dərmanlarla. Yuxusuzluq və ya həyəcanın səbəbini həkimlə müəyyənləşdirin; yuxu rejimi və nəfəs məşqləri də kömək edə bilər.",
    "symptom_triage": "Baş ağrısı və qızdırma bir çox səbəbdən ola bilər; diaqnoz qoya bilmirəm. Boyun sərtliyi, çaşqınlıq, çox şiddətli və ani baş ağrısı, səpgi, qıcolma və ya nəfəs darlığı varsa, dərhal 103 və ya 112. Əks halda əlamətlər keçmirsə və ya qızdırma yüksəkdirsə (xüsusilə uşaq, hamilə, xroniki xəstə), həkimə müraciət edin.",
    "choose_drug": "Mən sizin üçün dərman seçə bilmərəm: ağrının səbəbi, yaşınız, xroniki xəstəlikləriniz və qəbul etdiyiniz dərmanlar vacibdir. Əczaçı bu suallarla sizə kömək edə bilər; boyun sərtliyi, çaşqınlıq, ani şiddətli ağrı və ya qızdırma varsa həkimə müraciət edin. Artıq bir dərmanınız varsa, adını yazın: vərəqəsindən məlumat verərəm.",
    "dose_confusion": "Vurmazdan və ya qəbul etməzdən əvvəl dayanın: resept və qutudakı rəqəmlər (məsələn həcm ml ilə konsentrasiya mq/ml) uyğun gəlmirsə, səhv doza təhlükəli ola bilər. Həkimdən və ya əczaçıdan dərmanı, dozanı, konsentrasiyanı və həcmi dəqiq təsdiqləməsini xahiş edin; şübhə varsa yeritməyin.",
    "kidney_dose": "Böyrək və ya qaraciyər xəstəliyi olduqda bəzi dərmanların, o cümlədən antibiotiklərin dozası dəyişdirilməlidir; bunu yalnız həkim və ya əczaçı konkret dərmana və analiz nəticələrinə görə müəyyən edir. Qutudakı «adi doza» sizə uyğun olmaya bilər: qəbul etməzdən əvvəl soruşun.",
}
REPLIES_EN = {
    "overdose": "This may be an emergency: if too much medicine was taken, by mistake or in an unknown amount, do not wait even if the person feels fine. Call 103 or 112 now and give the medicine name, the approximate amount, the time, age and weight. Do not induce vomiting or give milk unless a medical professional tells you to. Keep the packaging and do not leave the person alone.",
    "harm_others": "I can't help with that: giving someone medicine without their knowledge or incapacitating them is dangerous and illegal. If someone is in danger, call 102 (police) or 112, and 103 for medical help.",
    "lethal_info": "I can't give specific dangerous or lethal amounts: the risk depends on age, weight, health and other substances, and the information can be used to harm. If someone has already taken too much, do not wait: call 103 or 112.",
}
EN_WORDS = {"the", "and", "how", "what", "can", "you", "your", "my", "to", "of", "is", "in", "for", "with", "tell", "make",
            "ignore", "all", "previous", "instructions", "side", "effects", "english", "take", "does", "do", "are", "this", "that"}


def is_english(key):
    w = key.split()
    return len(w) >= 3 and sum(x in EN_WORDS for x in w) >= 3


def check(message):
    """Returns (rule name, reviewed guidance, message_is_english) or None."""
    key = search_key(message)
    en = is_english(key)
    for name, pats in RULES:
        if all(re.search(p, key) for p in pats):
            return name, (REPLIES_EN.get(name) if en and name in REPLIES_EN else REPLIES[name]), en
    return None


def strip_injection(message):
    """Remove sentences that try to give instructions to the assistant. Returns (clean message, something_removed)."""
    parts = re.split(r"(?<=[.!?])\s+", message.strip())
    kept = [p for p in parts if not re.search(INJ, search_key(p))]
    return " ".join(kept).strip(), len(kept) < len(parts)


def is_injection_text(text):
    return bool(re.search(INJ, search_key(text)))


def injection_reply(en=False):
    if en:
        return "I can't follow that request. I only answer questions about medicine leaflets and reliable health sources."
    return "Bu göstərişi yerinə yetirə bilmirəm. Mən yalnız dərman təlimatları və etibarlı səhiyyə mənbələri əsasında cavab verirəm. Hansı dərman və ya mövzu haqqında məlumat istəyirsiniz?"