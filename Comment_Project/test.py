from version1 import model
import pandas as pd
from padded_seq import seq_and_pad
import pickle
MAXLEN = 250 #681
PADDING = 'post'
path = '../../saved_model/my_model4'
tokenizer = 'tokenizer.pickle'
# sentence = 'بازی بدی نبود اونقدر هم ارزش نداره ک بخای. بری رفتار پرسنل اصلن خوب نیست بازی باگ نور داره اکتور زعیف واقعن باید تلاشتونو بیشتر کنید'
# sentence = 'این اتاق فرار استاندارد نبود تاچ زیاد میکردن اما تاچ نرمال نه درکل مناسب خانم ها نی داخل بازی از لفظ های قشنگی استفاده نمی'
# sentence = "رفتارشون خوب بود اما معماها بی معنا و الکی بودن، در کل چنتا اتاق تاریک بدون سناریوی جالب. جای لوکیشن هم اصلا مناسب نبود. در کل پیشنهاد نمیکنم."
# sentence = 'بی ناموساااا'
# sentence = 'اگه می خواید پولتونو تو جوب بریزید برید اینجا'
sentence = 'افتتتتتتضااااااااااااااااااححححح'
# sentence = ' خرابی بد بچه های مردم راازرابدر مکنید لااقل یه شماره تلفن بدین ازتون شکایت بشه خرابی بد بچه های مردم راازرابدر مکنید لااقل یه شماره تلفن بدین ازتون شکایت بشه خرابی بد بچه های مردم راازرابدر مکنید لااقل یه شماره تلفن بدین ازتون شکایت بشه'
output = model(sentence, tokenizer, PADDING, MAXLEN, path)
print(output)
# print(sentence)
# with open(tokenizer, 'rb') as handle:
#     tokenizer = pickle.load(handle)

# print(seq_and_pad(tokenizer, [sentence], PADDING, MAXLEN)[0])