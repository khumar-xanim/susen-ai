from catalogue_names import reconcile

NAME = "ATROPIN SULFAT, İnyeksiya üçün məhlul, Ampul 1 ml, №10 (ATROPINE SULFATE 1 MQ/ML, ZDOROVYE ƏCZAÇILIQ ŞIRKƏTI MMC, Ukrayna)"


def test_wrong_columns_are_replaced_by_the_product_string():
    r = reconcile(NAME, "Tablet", "DAPOXETINE + TADALAFIL", "(60+20) MQ", "SYDLER REMEDIES PVT LTD", "Hindistan", "Blister", "№10")
    assert r["cols_match_name"] is False
    assert r["form_clean"] == "İnyeksiya üçün məhlul"
    assert r["country_clean"] == "Ukrayna"


def test_consistent_columns_are_kept():
    name = "ABIZOL, Tablet, Blister, №28 (ARIPIPRAZOLE 15 MQ, NOBEL ILAÇ SAN VE TIC AŞ, Türkiyə)"
    r = reconcile(name, "Tablet", "ARIPIPRAZOLE", "15 MQ", "NOBEL ILAÇ SAN VE TIC AŞ", "Türkiyə", "Blister", "№28")
    assert r["cols_match_name"] is True