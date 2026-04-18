📌 Opis projektu

Projekt został zrealizowany w ramach laboratorium z analizy obrazu.
Celem było stworzenie aplikacji w Pythonie umożliwiającej:

wykrywanie charakterystycznych punktów obrazu,
wyznaczanie deskryptorów cech lokalnych,
dopasowanie cech pomiędzy obrazami,
lokalizację obiektu na podstawie dopasowań,
śledzenie obiektu w sekwencji wideo.

Aplikacja wykorzystuje bibliotekę OpenCV oraz algorytmy detekcji cech takie jak ORB, SIFT lub BRISK.

⚙️ Wykorzystane technologie
Python 3
OpenCV
NumPy
🧠 Zastosowane metody
🔹 Detekcja cech

W projekcie zastosowano algorytmy:

ORB (domyślny)
SIFT
BRISK

Do wykrywania punktów kluczowych oraz obliczania deskryptorów wykorzystano funkcję:

detectAndCompute()
🔹 Dopasowanie deskryptorów

Do porównania deskryptorów użyto:

BFMatcher (Brute Force Matcher)
metody k-NN matching (k=2)

Zastosowano również test Lowe’a (ratio test):

m.distance < 0.75 * n.distance

Pozwala to odrzucić słabe i niepewne dopasowania.

🔹 Lokalizacja obiektu

Do określenia położenia obiektu wykorzystano:

homografię (cv2.findHomography)
metodę RANSAC

Dzięki temu możliwe jest:

wykrycie obiektu mimo obrotu,
zmiany skali,
perspektywy.
🔹 Wizualizacja

Program wizualizuje:

dopasowane punkty pomiędzy obrazami,
obszar wykrytego obiektu (wielokąt),
liczbę dopasowań,
status wykrycia.
🎥 Śledzenie wideo

W trybie wideo program:

analizuje każdą klatkę,
wykrywa cechy,
dopasowuje je do wzorca,
lokalizuje obiekt,
zaznacza go tylko wtedy, gdy został wiarygodnie wykryty.
🚀 Uruchomienie
▶️ Obraz:
python main.py --reference saw1.jpg --image saw2.jpg
🎥 Wideo:
python main.py --reference saw1.jpg --video sawmovie.mp4
📊 Wyniki

Program poprawnie:

wykrywa obiekt na innych obrazach,
lokalizuje jego położenie,
śledzi obiekt w materiale wideo w czasie rzeczywistym.
⚠️ Uwagi
W przypadku niewystarczającej liczby dopasowań obiekt nie jest zaznaczany.
Stabilność detekcji zależy od jakości obrazu oraz liczby wykrytych cech.
Dla lepszych rezultatów można użyć algorytmu SIFT.
📁 Struktura projektu
main.py
saw1.jpg
saw2.jpg
sawmovie.mp4
