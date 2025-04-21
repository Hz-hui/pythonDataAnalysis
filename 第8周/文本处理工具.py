class AnalyzerDecorator:
    def __init__(self, func):
        self.func = func


    def __call__(self, text):
        def wrapper(text):
            text = text.lower()
            text = text.strip()
            return self.func(text)
        return wrapper(text)
        


@AnalyzerDecorator
def count_words(text):
    word_count = 1
    for i in text:
        word_count += 1 if i == ' ' else 0

    text = text.replace('bad', '***')
    text = text.replace('evil', '****')
    text = text.replace('ugly', '****')

    #lis = [word_count, text]
    return word_count, text



if __name__ == "__main__":
    input_text = input()
    word_count, filtered_text1 = count_words(input_text)
    print(f"单词数量:{word_count}")
    print(filtered_text1) 