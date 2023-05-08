from version1 import model



MAXLEN = 120 #681
PADDING = 'post'
path = '../../saved_model/my_model'
tokenizer = 'tokenizer.pickle'
sentence = 'بازی بدی نبود اونقدر هم ارزش نداره ک بخای. بری رفتار پرسنل اصلن خوب نیست بازی باگ نور داره اکتور زعیف واقعن باید تلاشتونو بیشتر کنید'


output = model(sentence, tokenizer, PADDING, MAXLEN, path)
print(output)