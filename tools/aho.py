import ahocorasick

def build_automaton(keywords):
    A = ahocorasick.Automaton()
    for keyword in keywords:
        A.add_word(keyword, keyword)  # (key, value)
    A.make_automaton()
    return A

def matches_keyword_aho(input_str, keyword_config, automaton):
    # Check required keywords
    if keyword_config["required"]:
        required_matches = set()
        for _, keyword in automaton.iter(input_str):
            if keyword in keyword_config["required"]:
                required_matches.add(keyword)
        if not required_matches:
            return False

    # Check exclude keywords
    for _, keyword in automaton.iter(input_str):
        if keyword in keyword_config["exclude"]:
            return False

    # Check include keywords
    if keyword_config["include"]:
        include_matches = False
        for _, keyword in automaton.iter(input_str):
            if keyword in keyword_config["include"]:
                include_matches = True
                break
        if not include_matches:
            return False

    return True