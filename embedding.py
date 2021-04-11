import fasttext
if __name__ == '__main__':
    model = fasttext.load_model('/home/thanhlc/thanhlc/Data/c_pretrained.bin')
    vect = model.get_sentence_vector("int Num() {")
    print(vect)

    