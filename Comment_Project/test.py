from version1 import model
import pandas as pd


MAXLEN = 120 #681
PADDING = 'post'
path = '../../saved_model/my_model'
tokenizer = 'tokenizer.pickle'
sentence = 'بازی بدی نبود اونقدر هم ارزش نداره ک بخای. بری رفتار پرسنل اصلن خوب نیست بازی باگ نور داره اکتور زعیف واقعن باید تلاشتونو بیشتر کنید'

df = pd.read_excel('/Users/niloufar/Desktop/DeepLearning/tf_specialization/comment/spam_or_not1.xlsx')
sentence = "رفتارشون خوب بود اما معماها بی معنا و الکی بودن، در کل چنتا اتاق تاریک بدون سناریوی جالب. جای لوکیشن هم اصلا مناسب نبود. در کل پیشنهاد نمیکنم."
output = model(sentence, tokenizer, PADDING, MAXLEN, path)
print(output)
# print(sentence)