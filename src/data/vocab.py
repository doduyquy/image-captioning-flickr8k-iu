#cầu nối text <-> số
#model không hiểu "a dog is running" -> nó hiểu "[45,12,9,233]"
#itos: index to string, stoi: string to index
import collections

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold=freq_threshold #-> giữ từ phổ bién ->  model học tốt hơn
        self.itos={
            0:"<pad>", # ví batch cần dùng lenght
            1:"<start>", # cho decoder biết bắt đàu sinh từ đâu
            2:"<end>", # cho biết khi nào dừng
            3:"<unk>" # nếu không có
        }
        self.stoi={v: k for k,v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, sentence_list):
        frequincies=collections.Counter()
        idx=4 # tạo index kế tiếp, không lặp lại 4 tk trên kia
        for sentence in sentence_list:
            for word in sentence.split():
                frequincies[word]+=1
                if frequincies[word]==self.freq_threshold: 
                    self.stoi[word]=idx
                    self.itos[idx]=word
                    idx+=1

    def numericalize(self, text):
        tokenized=text.split()
        return [
            self.stoi.get(token, self.stoi["<unk>"])
            for token in tokenized
        ]

if __name__ == "__main__":
    # Test nhanh Vocabulary
    sentences = [
        "a dog is running",
        "a dog is barking",
        "a cat is sleeping",
        "the dog is running"
    ]
    
    # freq_threshold = 2 để lọc ra các từ xuất hiện >= 2 lần
    vocab = Vocabulary(freq_threshold=2)
    vocab.build_vocab(sentences)
    
    print("Vocab size:", len(vocab))
    print("itos:", vocab.itos)
    print("stoi:", vocab.stoi)
    
    test_text = "a dog is running fast"
    print(f"\nCâu test: '{test_text}'")
    print("Numericalize:", vocab.numericalize(test_text))
    
    """
    python -m src.data.vocab                                                                                                                           ─╯
        Vocab size: 8
        itos: {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>', 4: 'a', 5: 'dog', 6: 'is', 7: 'running'}
        stoi: {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3, 'a': 4, 'dog': 5, 'is': 6, 'running': 7}

        Câu test: 'a dog is running fast'
        Numericalize: [4, 5, 6, 7, 3]
    """