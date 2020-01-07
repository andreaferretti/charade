import torch


def tagged_tokens(path):
    last_is_eos = False

    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            pairs = []
            yield_line = False
            if line != '':
                tokens = line.split(' ')
                pairs.append(('<SOS>', '<SOS>'))
                for token in tokens:
                    if token != '':
                        chunks = token.split('|')
                        word = chunks[0]
                        tag = chunks[-1]
                        pairs.append((word, tag))
                        if tag != 'O':
                            yield_line = True
                pairs.append(('<EOS>', '<EOS>'))

            if yield_line:
                for pair in pairs:
                    yield pair



def generate(path):
    words_ = set()
    tags_ = set()

    count = 0
    for word, tag in tagged_tokens(path):
        count += 1
        words_.add(word)
        tags_.add(tag)

    words = { w: i for i, w in enumerate(words_) }
    tags = { t: i for i, t in enumerate(tags_) }

    with open(f'{path}-words.index', 'w') as f:
        for word in words:
            f.write(word + '\n')

    with open(f'{path}-tags.index', 'w') as f:
        for p in tags:
            f.write(p + '\n')

    train_count = count * 5 // 6
    test_count = count - train_count

    train_word_tensor = torch.LongTensor(train_count)
    train_tag_tensor = torch.LongTensor(train_count)
    test_word_tensor = torch.LongTensor(test_count)
    test_tag_tensor = torch.LongTensor(test_count)

    i = 0
    word_tensor = train_word_tensor
    tag_tensor = train_tag_tensor

    for word, tag in tagged_tokens(path):
        word_tensor[i] = words[word]
        tag_tensor[i] = tags[tag]
        i += 1
        if i == train_count:
            word_tensor = test_word_tensor
            tag_tensor = test_tag_tensor
            i = 0

    torch.save(train_word_tensor, path + '-train-words.pth')
    torch.save(train_tag_tensor, path + '-train-tags.pth')
    torch.save(test_word_tensor, path + '-test-words.pth')
    torch.save(test_tag_tensor, path + '-test-tags.pth')


if __name__ == '__main__':
    import sys
    generate(sys.argv[1])