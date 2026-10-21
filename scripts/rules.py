"""Deterministic safety rules (plan 1.13, 1.14). They run BEFORE the language model."""
import re

from normalize import search_key

# Patterns are written in folded form: ə->e ı->i ş->s ç->c ğ->g ö->o ü->u. All patterns of a rule must match.
RULES = [
    ("injection", [r"(evvelki|yuxaridaki|butun|bu) (telimat|qayda)\w* (unut|nezere alma|kecme)|ignore (previous|all|above|your)|system prompt|sistem prompt|sistem mesaj|rolunu deyis|jailbreak"]),
    ("selfharm", [r"intihar|ozumu oldur|olmek isteyirem|ozume zerer"]),
    ("emergency", [r"zeherlen|overdoz|heddinden artiq|nefes ala bilmir|husunu itir|boguluram|allergik sok|cox (icmisem|atmisam|qebul etmisem)"]),
    ("emergency", [r"\b([5-9]|\d{2,})\s?(eded )?(tablet|kapsul|ampul|eded)", r"icmisem|atmisam|qebul etmisem|vurmusam"]),
    ("dose_advice", [r"\b(artir|azalt|ikiqat|iki qat|yarisin|yarim)\w*", r"doza|tablet|ampul|kapsul|\bmq\b|\bml\b|defe|qebul|\bic\w*"]),
    ("dose_advice", [r"\b(mene|usagima|korpeme|anama|atama|babama)\b",
                     r"\bnece (mq|ml|tablet|ampul|kapsul|damci|defe|gram)\b|\bhansi doza|\bne qeder (icim|qebul|vurum|atim)"]),
    ("substitution", [r"\b(evez|analoq|alternativ|eynitesirli|eyni tesirli)\w*|yerine basqa|basqa derman|daha yaxsi derman|hansi derman daha"]),
    ("diagnosis", [r"\b(diaqnoz|tesxis)\w*|hansi derman\w*\s+(\w+\s+){0,3}(yaxsi|komek|tesir|uygun|faydali|isleyir|icim|icmeliyem|almaliyam)|mene (hansi )?derman (tovsiye|lazim|ver)|menim (xestelik|simptom|sikayet)\w* (nedir|ne ola biler)"]),
    ("stop_advice", [r"\b(dayandir\w*|kesmek|kese bilerem|buraxa bilerem)", r"derman|tablet|kapsul|ampul|icmey|qebul"]),
]

REPLIES = {
    "injection": "Bu sorğunu yerinə yetirə bilmirəm. Mən yalnız dərmanların təlimat vərəqələri və kataloq məlumatları əsasında suallara cavab verirəm. Hansı dərman haqqında məlumat istəyirsiniz?",
    "selfharm": "Bunu yazdığınız üçün təşəkkür edirəm, təhlükəsizliyiniz çox vacibdir. Əgər özünüzə zərər vermək fikri varsa, dərhal 112-yə zəng edin və yaxın bir insanın yanına gedin. Tək qalmayın.",
    "emergency": "Təsvir etdiyiniz vəziyyət təcili ola bilər. Gözləməyin: dərhal 103 (təcili yardım) və ya 112 nömrəsinə zəng edin və ya ən yaxın xəstəxanaya müraciət edin. Qəbul etdiyiniz dərmanın adını və miqdarını həkimə deyin.",
    "dose_advice": "Dozanı dəyişmək və ya fərdi doza seçmək barədə məsləhət verə bilmərəm: doza yaşdan, çəkidən, xəstəlikdən və digər dərmanlardan asılıdır, bunu yalnız həkim və ya əczaçı müəyyən edə bilər. İstəsəniz, təlimat vərəqəsində yazılmış ümumi dozalanma məlumatını göstərə bilərəm.",
    "substitution": "Dərmanı başqası ilə əvəz etmək və ya analoq seçmək barədə məsləhət verə bilmərəm. Bu qərarı həkim və ya əczaçı verməlidir. Seçdiyiniz dərmanın təlimat vərəqəsindəki məlumatı göstərə bilərəm.",
    "diagnosis": "Diaqnoz qoymaq və ya sizin üçün dərman seçmək mənim işim deyil, bunu həkim edir. Konkret bir dərman haqqında təlimat vərəqəsi və kataloq məlumatı verə bilərəm.",
    "stop_advice": "Dərmanı özbaşına dayandırmaq, kəsmək və ya fasilə vermək barədə məsləhət verə bilmərəm. Bunu mütləq həkiminizlə danışın, çünki bəzi dərmanları qəfil dayandırmaq təhlükəli ola bilər. İstəsəniz, təlimat vərəqəsində bu barədə nə yazıldığına baxa bilərəm.",

}


def check(message):
    key = search_key(message)
    for name, pats in RULES:
        if all(re.search(p, key) for p in pats):
            return name, REPLIES[name]
    return None