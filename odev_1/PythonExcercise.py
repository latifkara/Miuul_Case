

# Görev 1: Verilen değerlerin veri yapılarını inceleyiniz.
x = 8
print(type(x))

y = 3.2
print(type(y))

z = 8j + 18
print(type(z))

a = "Hello World"
print(type(a))

b = True
print(type(b))

c = 23 < 22
print(type(c))

l = [1, 2, 3, 4, 5]
print(type(l))

d = {"Name": "Jake",
     "Age": 27,
     "Address": "Downtown"
     }
print(type(d))

t = ("Machine Learning", "Data Science")
print(type(t))

s = {"Python", "Machine Learning", "Data Science"}
print(type(s))

# Görev 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz,
# kelime kelime ayırınız.

text = "The goal is to turn data into information, and information into insight"
arr = [string.upper() for string in text.split()]

# Görev 3: Verilen listeye aşağıdaki adımları uygulayınız.
lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Adım 1: Verilen listenin eleman sayısına bakınız.
print(len(lst))

# Adım 2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
print(lst[0])
print(lst[10])

# Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
new_list = lst[0:4]

# Adım 4: Sekizinci indeksteki elemanı siliniz.
lst.remove(lst[8])

# Adım 5: Yeni bir eleman ekleyiniz.
lst.append("!")

# Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.
lst.insert(8, "N")

# Görev 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.

dictionary = {'Christian': ["America", 18],
              'Daisy': ["England", 12],
              "Antonio": ["Spain", 22],
              "Dante": ["Italy", 25]
              }

# Adım 1: Key değerlerine erişiniz.
print(dictionary.keys())

# Adım 2: Value'lara erişiniz.
print(dictionary.values())

# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dictionary['Daisy'] = ["England", 13]

# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
dictionary['Ahmet'] = ["Turkey", 24]

# Adım 5: Antonio'yu dictionary'den siliniz.
dictionary.pop('Antonio')


# Görev 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu listeleri
# return eden fonksiyon yazınız.

l = [2, 13, 18, 93, 22]

def func(arr):
     odd_list = []
     even_list = []
     for num in arr:
          if num % 2 == 0:
               even_list.append(num)
          else:
               odd_list.append(num)
     return even_list, odd_list

even_list, odd_list = func(l)


# Görev 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri
# bulunmaktadır. Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de
# tıp fakültesi öğrenci sırasına aittir. Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

def show_student(arr, faculty):
    for index, student in enumerate(arr, 1):
        print(f"{faculty} {index}. öğrenci: {student}")

show_student(ogrenciler[0:3], "Mühendislik Fakültesi")
show_student(ogrenciler[3:], "Tıp Fakültesi")


# Görev 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer
# almaktadır. Zip kullanarak ders bilgilerini bastırınız.

ders_kodu = ["CMP1005", "PYS1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

for k, ders, kont in list(zip(kredi, ders_kodu, kontenjan)):
    print(f"Kredisi {k} olan {ders} kodlu dersin kontenjanı {kont} kişidir.")


# Görev 8: Aşağıda 2 adet set verilmiştir. Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını
# eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.

def func_set(set_data1, set_data2):
    if set_data2.issubset(set_data1):
        print(set_data1.intersection(set_data2))
    else:
        print(set_data2.difference(set_data1))

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

func_set(kume1, kume2)

