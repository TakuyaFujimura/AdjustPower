import soundfile as sf


def wavread(fn):
    data, sr = sf.read(fn)
    f = sf.info(fn)
    return data, sr, f.subtype


def wavwrite(fn, data, sr, subtype):
    sf.write(fn, data, sr, subtype)
