from retrieval import detect_intents, wanted_categories
from rules import check


def test_child_question():
    i = detect_intents("bronxovent 1 yasli usaga vermek olar?")
    assert "pediatric" in i and "contraindications" in wanted_categories(i)


def test_route_question():
    assert "dosage" in wanted_categories(detect_intents("atropin icirler yoxsa inyeksiya ile vururlar?"))


def test_price_variants():
    assert "price" in detect_intents("adeliv neceyedir?")


def test_drug_recommendation_refused():
    assert check("hansi derman bas agrima yaxsi tesir eder?")[0] == "diagnosis"