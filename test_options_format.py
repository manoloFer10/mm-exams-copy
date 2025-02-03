import re

def check_options_format(question):
    # Match any Unicode letter as a standalone word followed by punctuation and/or whitespace
    pattern = r"^\s*\b[^\W\d_]\b[^\w\s]*\s*"
    
    question["options"] = [
        re.sub(pattern, "", option, flags=re.UNICODE) if re.match(pattern, option, flags=re.UNICODE)
        else option
        for option in question["options"]
    ]

    return question



