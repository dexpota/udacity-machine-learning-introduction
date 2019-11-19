def from_string_to_list(x):
    return [] if not x.strip('][') else x.strip('][').split(",")