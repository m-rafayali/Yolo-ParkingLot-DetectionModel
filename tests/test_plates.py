from parkwatch.plates.formats import PlateFormat, best_match, sg_checksum, sg_valid
from parkwatch.plates.readers import parse_llm_reply


def test_sg_checksum_known_plates():
    # the two plates from the measured run on real footage
    assert sg_valid("SHD1476M")
    assert sg_valid("SLN9145R")
    assert sg_checksum("SBS", "3229") == "P"
    assert not sg_valid("SHD1476N")


def test_sg_repair_confusable_characters():
    fmt = PlateFormat("sg")
    assert fmt.validate("5HD1476M") == ("SHD1476M", True)   # 5 read instead of S in the letter prefix
    assert fmt.validate("SLN9I45R") == ("SLN9145R", True)   # I read instead of 1 in the digits
    assert fmt.validate("shd 1476 m") == ("SHD1476M", True)  # spaces / case
    plate, ok = fmt.validate("SHD1476Q")
    assert plate == "SHD1476Q" and not ok


def test_generic_format():
    fmt = PlateFormat("generic")
    assert fmt.validate("abc-123") == ("ABC123", True)
    assert fmt.validate("") == (None, False)


def test_fuzzy_match_is_unambiguous():
    assert best_match("SHD1476N", ["SHD1476M", "SLN9145R"]) == "SHD1476M"
    assert best_match("XXXXXXXX", ["SHD1476M"]) is None
    assert best_match("AB12", ["AB13", "AB14"]) is None  # tie -> refuse to guess


def test_parse_llm_reply():
    assert parse_llm_reply('```json\n{"plate": "SBA1234K", "confidence": 0.93}\n```') == ("SBA1234K", 0.93)
    assert parse_llm_reply('{"plate": null, "confidence": 0}') == (None, 0.0)
    assert parse_llm_reply("The plate reads SBA1234K.")[0] == "SBA1234K"
