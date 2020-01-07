import re
from collections import Counter

def split_text(text):
    return [t for t in re.split('\W', text) if len(t) > 1 or t == 'i']

def get_features(text, patterns, extra_patterns):
    _count = Counter(split_text(text))
    features = [_count[p] for p in patterns]
    for extra_pattern in extra_patterns:
        pass
    return features