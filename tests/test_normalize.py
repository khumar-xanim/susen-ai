from normalize import search_key


def test_capital_dotted_i():
    assert search_key("EXİNASEYA KOMP.") == "exinaseya komp"


def test_typed_without_special_letters():
    assert search_key("dərman məhlul") == search_key("derman mehlul")


def test_dotless_i():
    assert search_key("Işıq") == "isiq"

def test_symbols_removed():
    assert search_key("Abizol® 5 mq") == "abizol 5 mq"
    
def test_cyrillic_lookalikes():
    assert search_key("İstifаdə qаydаsı və dоzаsı") == search_key("Istifade qaydasi ve dozasi")