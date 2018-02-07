import os, codecs


def load_text(path, encoding = "utf8"):
    paths = path.split("/")
    with codecs.open(os.path.join(paths[0], *paths[1:]),encoding=encoding) as f:
        text = f.read()
    return text

def inverse_dict(d):
    return {e[1]:e[0] for e in d.items()}
