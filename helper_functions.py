def preprocess(s: str) -> str:
    """Remove notes in double parentheses, @, punctuation, redundant
    whitespace. Lowercases

    :param str s: Input string
    :return str: Output string
    """
    import re

    s = re.sub(r"\(\([^)]*\)\)", "", s)
    s = s.replace("@", "")
    from string import punctuation

    for p in punctuation:
        s = s.replace(p, "")
    s = " ".join(s.split())
    s = s.casefold()
    return s
