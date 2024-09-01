from zeta.tokenizers.tokenmonster import TokenMonster


def test_token_monster_initialization():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")

    assert isinstance(tokenizer, TokenMonster)
    assert tokenizer.vocab is not None


def test_token_monster_set_local_directory():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    tokenizer.set_local_directory(
        "/path/to/your/directory"
    )  # replace with your actual directory

    # There's no direct way to assert the effect of this method as it doesn't return anything
    # and it doesn't change any accessible state of the TokenMonster object.
    # You might need to check manually if the directory is set correctly.


def test_token_monster_load():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    tokenizer.load("englishcode-32000-consistent-v1")

    assert tokenizer.vocab is not None


def test_token_monster_load_multiprocess_safe():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    tokenizer.load_multiprocess_safe("englishcode-32000-consistent-v1")

    assert tokenizer.vocab is not None


def test_token_monster_new():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    yaml = """
    tokens:
      - token: " "
        score: 0
      - token: "e"
        score: 1
      - token: "t"
        score: 2
    """
    tokenizer.new(yaml)

    assert tokenizer.vocab is not None


def test_token_monster_export_yaml():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    yaml = tokenizer.export_yaml()

    assert isinstance(yaml, bytes)


def test_token_monster_tokenize():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    tokens = tokenizer.tokenize("Hello world!")

    assert isinstance(tokens, list)


def test_token_monster_tokenize_count():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    count = tokenizer.tokenize_count("Hello world!")

    assert isinstance(count, int)


def test_token_monster_decode():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    tokens = tokenizer.tokenize("Hello world!")
    text = tokenizer.decode(tokens)

    assert isinstance(text, str)
    assert text == "Hello world!"


def test_token_monster_decoder():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    decoder = tokenizer.decoder()

    assert decoder is not None


def test_token_monster_get_dictionary():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    dictionary = tokenizer.get_dictionary()

    assert isinstance(dictionary, list)


def test_token_monster_charset():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    charset = tokenizer.charset()

    assert isinstance(charset, str)


def test_token_monster_normalization():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    normalization = tokenizer.normalization()

    assert isinstance(normalization, str)


def test_token_monster_capcode():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    capcode = tokenizer.capcode()

    assert isinstance(capcode, int)


def test_token_monster_mode():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    mode = tokenizer.mode()

    assert isinstance(mode, int)


def test_token_monster_id_to_token():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    token = tokenizer.id_to_token(1)

    assert isinstance(token, str)


def test_token_monster_id_to_token_decoded():
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    token = tokenizer.id_to_token_decoded(1)

    assert isinstance(token, str)
