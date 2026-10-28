from scratchllm.data.cleaning import PII_SPAM_MATCHES, scrub_pii
from scratchllm.data.pii import EMAIL_PLACEHOLDER, PHONE_PLACEHOLDER, find_matches, mask


def kinds(text: str) -> list[str]:
    return [m.kind for m in find_matches(text)]


def test_positions_point_at_the_matched_text():
    text = "اتصل على 01102106655 أو راسل a.b@example.com الآن"
    phone, email = find_matches(text)
    assert text[phone.start : phone.end] == "01102106655" and phone.kind == "phone"
    assert text[email.start : email.end] == "a.b@example.com" and email.kind == "email"


def test_arabic_indic_digits_are_found_at_their_original_position():
    text = "رقمي ٠١١٠٢١٠٦٦٥٥ للتواصل"
    (m,) = find_matches(text)
    assert text[m.start : m.end] == "٠١١٠٢١٠٦٦٥٥"
    assert mask(text, [m]) == f"رقمي {PHONE_PLACEHOLDER} للتواصل"


def test_dates_and_year_ranges_are_not_phones():
    assert find_matches("نشر في 2019-05-12 وفي 12/05/2019 بين 2002-2003") == []


def test_a_date_next_to_a_phone_does_not_hide_it_or_move_it():
    text = "2019-05-12 01102106655"
    (m,) = find_matches(text)
    assert text[m.start : m.end] == "01102106655"


def test_mask_replaces_each_match_and_keeps_the_text_around_it():
    text = "a 01102106655 b x@y.com c"
    assert mask(text, find_matches(text)) == f"a {PHONE_PLACEHOLDER} b {EMAIL_PLACEHOLDER} c"


def test_an_email_whose_digits_look_like_a_phone_becomes_one_placeholder():
    text = "راسلني 01102106655@mail.com"
    assert sorted(kinds(text)) == ["email", "phone"]  # both counted by the EDA check
    assert mask(text, find_matches(text)) == f"راسلني {EMAIL_PLACEHOLDER}"


def test_text_without_matches_is_returned_unchanged():
    assert mask("لا شيء هنا", []) == "لا شيء هنا"


# ---------- the rule ----------


def test_scrub_masks_a_page_with_a_contact_number_and_keeps_the_sentence():
    assert scrub_pii("للحجز اتصل 01102106655 الآن") == f"للحجز اتصل {PHONE_PLACEHOLDER} الآن"


def test_scrub_leaves_a_clean_page_untouched():
    assert scrub_pii("صفحة عادية بلا أرقام") == "صفحة عادية بلا أرقام"


def test_four_matches_are_masked_and_five_drop_the_page():
    numbers = [f"0110210665{i}" for i in range(PII_SPAM_MATCHES)]
    four = " ".join(numbers[: PII_SPAM_MATCHES - 1])
    assert scrub_pii(four) == " ".join([PHONE_PLACEHOLDER] * (PII_SPAM_MATCHES - 1))
    assert scrub_pii(" ".join(numbers)) is None


def test_after_masking_a_second_scan_finds_nothing():
    text = "اتصل 01102106655 أو 01002106655 أو x@y.com"
    assert find_matches(scrub_pii(text) or "") == []


# ---------- placeholder emails left by the data source ----------


def test_reserved_example_domains_are_flagged_as_placeholders():
    text = "a email@example.com b firstname.lastname@example.org c x@Example.NET d real@gmail.com"
    flags = [(text[m.start : m.end], m.placeholder) for m in find_matches(text)]
    assert flags == [
        ("email@example.com", True),
        ("firstname.lastname@example.org", True),
        ("x@Example.NET", True),
        ("real@gmail.com", False),
    ]


def test_a_domain_that_only_contains_example_is_not_a_placeholder():
    (m,) = find_matches("ali@myexample.com")
    assert not m.placeholder


def test_placeholders_are_masked_like_other_emails():
    assert scrub_pii("اكتب إلى email@example.com") == f"اكتب إلى {EMAIL_PLACEHOLDER}"


def test_placeholders_do_not_count_towards_the_spam_limit():
    many = " ".join(["email@example.com"] * (PII_SPAM_MATCHES + 3))
    assert scrub_pii(many) == " ".join([EMAIL_PLACEHOLDER] * (PII_SPAM_MATCHES + 3))
    real = " ".join(f"0110210665{i}" for i in range(PII_SPAM_MATCHES - 1))
    assert scrub_pii(real + " email@example.com") is not None  # four real + one placeholder
