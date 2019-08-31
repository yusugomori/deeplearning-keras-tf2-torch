class Vocab(object):
    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.special_chars = ['<pad>', '<s>', '</s>', '<unk>']
        self.bos_char = self.special_chars[1]
        self.eos_char = self.special_chars[2]
        self.oov_char = self.special_chars[3]

    def fit(self, path):
        self._words = set()

        with open(path, 'r') as f:
            sentences = f.read().splitlines()

        for sentence in sentences:
            self._words.update(sentence.split())

        self.w2i = {w: (i + len(self.special_chars))
                    for i, w in enumerate(self._words)}

        for i, w in enumerate(self.special_chars):
            self.w2i[w] = i

        self.i2w = {i: w for w, i in self.w2i.items()}

    def transform(self, path, bos=False, eos=False):
        output = []

        with open(path, 'r') as f:
            sentences = f.read().splitlines()

        for sentence in sentences:
            sentence = sentence.split()
            if bos:
                sentence = [self.bos_char] + sentence
            if eos:
                sentence = sentence + [self.eos_char]
            output.append(self.encode(sentence))

        return output

    def encode(self, sentence):
        output = []

        for w in sentence:
            if w not in self.w2i:
                idx = self.w2i[self.oov_char]
            else:
                idx = self.w2i[w]
            output.append(idx)

        return output

    def decode(self, sentence):
        return [self.i2w[id] for id in sentence]
