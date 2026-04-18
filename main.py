#!/usr/bin/env python3
"""
LAB3: Deskryptory
Rozwiązanie

Cele:
1. Wyznaczyć deskryptory dla obrazu wzorcowego.
2. Znaleźć obiekt na innym obrazie przez porównanie z wzorcem.
3. Śledzić obiekt w kolejnych klatkach filmu.
"""

import argparse
import sys
from typing import Optional, Tuple

import cv2
import numpy as np


# ============================================================
# TODO 1: Utworzenie detektora i matchera
# ============================================================

def utworz_detektor(nazwa_metody: str = "ORB"):
    if nazwa_metody.upper() == "ORB":
        return cv2.ORB_create(nfeatures=1000)
    elif nazwa_metody.upper() == "BRISK":
        return cv2.BRISK_create()
    elif nazwa_metody.upper() == "SIFT":
        return cv2.SIFT_create()
    else:
        raise ValueError("Nieznana metoda")


def utworz_matcher(nazwa_metody: str = "ORB"):
    if nazwa_metody.upper() in ["ORB", "BRISK"]:
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif nazwa_metody.upper() == "SIFT":
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        raise ValueError("Nieznana metoda")


# ============================================================
# TODO 2: Wyznaczanie punktów kluczowych i deskryptorów
# ============================================================

def wyznacz_cechy(obraz_szary, detektor):
    kp, des = detektor.detectAndCompute(obraz_szary, None)
    return kp, des


# ============================================================
# TODO 3: Dopasowanie deskryptorów
# ============================================================

def dopasuj_deskryptory(
        deskryptory_wzorca: np.ndarray,
        deskryptory_obrazu: np.ndarray,
        matcher
):
    """
    Dopasowuje deskryptory pomiędzy wzorcem i obrazem analizowanym.
    """
    if deskryptory_wzorca is None or deskryptory_obrazu is None:
        return []

    if len(deskryptory_wzorca) < 2 or len(deskryptory_obrazu) < 2:
        return []

    # Używamy knnMatch do znalezienia 2 najbliższych sąsiadów dla każdego deskryptora
    matches = matcher.knnMatch(deskryptory_wzorca, deskryptory_obrazu, k=2)

    dobre_dopasowania = []
    # Test Lowe'a (Lowe's ratio test)
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                dobre_dopasowania.append(m)

    return dobre_dopasowania


# ============================================================
# TODO 4: Lokalizacja obiektu
# ============================================================

def zlokalizuj_obiekt(
        punkty_wzorca,
        punkty_obrazu,
        dopasowania,
        ksztalt_wzorca
):
    """
    Lokalizuje obiekt na podstawie dopasowanych punktów.
    """
    MIN_MATCH_COUNT = 5  # Minimalna liczba dopasowań by uznać, że obiekt jest na zdjęciu

    if len(dopasowania) >= MIN_MATCH_COUNT:
        # Pobranie współrzędnych dopasowanych punktów
        src_pts = np.float32([punkty_wzorca[m.queryIdx].pt for m in dopasowania]).reshape(-1, 1, 2)
        dst_pts = np.float32([punkty_obrazu[m.trainIdx].pt for m in dopasowania]).reshape(-1, 1, 2)

        # Obliczenie macierzy homografii za pomocą RANSAC
        M, maska_inlierow = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Jeśli homografia została poprawnie wyznaczona
        if M is not None:
            h, w = ksztalt_wzorca[:2]
            # Narożniki wzorca
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            # Przekształcenie narożników wzorca na układ współrzędnych analizowanego obrazu
            narozniki = cv2.perspectiveTransform(pts, M)
            return narozniki, maska_inlierow

    # Gdy za mało dopasowań lub nie wyznaczono M
    return None, None


# ============================================================
# TODO 5: Wizualizacja wyniku
# ============================================================

def narysuj_obiekt(obraz_bgr: np.ndarray, narozniki) -> np.ndarray:
    """
    Rysuje obszar wykrytego obiektu.
    """
    obraz_wynikowy = obraz_bgr.copy()

    if narozniki is not None:
        cv2.polylines(obraz_wynikowy, [np.int32(narozniki)], True, (0, 255, 0), 3, cv2.LINE_AA)

    return obraz_wynikowy


def narysuj_dopasowania(
        obraz_wzorca,
        punkty_wzorca,
        obraz_testowy,
        punkty_testowe,
        dopasowania,
        maks_liczba: int = 50
):
    """
    Tworzy wizualizację dopasowanych punktów.
    """
    # Posortuj dopasowania po dystansie, by rysować tylko najlepsze
    dopasowania = sorted(dopasowania, key=lambda x: x.distance)
    najlepsze_dopasowania = dopasowania[:maks_liczba]

    obraz_wizualizacji = cv2.drawMatches(
        obraz_wzorca, punkty_wzorca,
        obraz_testowy, punkty_testowe,
        najlepsze_dopasowania, None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return obraz_wizualizacji


# ============================================================
# Część 1 i 2: obraz wzorcowy + obraz testowy
# ============================================================

def przetworz_obraz(referencja: str, obraz_testowy: str, metoda: str):
    obraz_wzorca_bgr = cv2.imread(referencja)
    obraz_testowy_bgr = cv2.imread(obraz_testowy)

    if obraz_wzorca_bgr is None:
        print(f"BŁĄD: Nie można otworzyć obrazu wzorcowego: {referencja}", file=sys.stderr)
        return

    if obraz_testowy_bgr is None:
        print(f"BŁĄD: Nie można otworzyć obrazu testowego: {obraz_testowy}", file=sys.stderr)
        return

    obraz_wzorca_szary = cv2.cvtColor(obraz_wzorca_bgr, cv2.COLOR_BGR2GRAY)
    obraz_testowy_szary = cv2.cvtColor(obraz_testowy_bgr, cv2.COLOR_BGR2GRAY)

    detektor = utworz_detektor(metoda)
    matcher = utworz_matcher(metoda)

    punkty_wzorca, deskryptory_wzorca = wyznacz_cechy(obraz_wzorca_szary, detektor)
    punkty_testowe, deskryptory_testowe = wyznacz_cechy(obraz_testowy_szary, detektor)

    dobre_dopasowania = dopasuj_deskryptory(deskryptory_wzorca, deskryptory_testowe, matcher)

    narozniki, maska_inlierow = zlokalizuj_obiekt(
        punkty_wzorca,
        punkty_testowe,
        dobre_dopasowania,
        obraz_wzorca_szary.shape
    )

    obraz_wynikowy = narysuj_obiekt(obraz_testowy_bgr, narozniki)
    obraz_dopasowan = narysuj_dopasowania(
        obraz_wzorca_bgr,
        punkty_wzorca,
        obraz_testowy_bgr,
        punkty_testowe,
        dobre_dopasowania
    )

    # TODO 6: Dodawanie informacji tekstowych
    cv2.putText(obraz_wynikowy, f"Zgodne cechy: {len(dobre_dopasowania)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    if narozniki is not None:
        cv2.putText(obraz_wynikowy, "STATUS: OBIEKT ZNALEZIONY", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(obraz_wynikowy, "STATUS: NIE ZNALEZIONO OBIEKTU", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    okna = [
        ("Obraz wzorcowy", obraz_wzorca_bgr),
        ("Wykryty obiekt", obraz_wynikowy),
        ("Dopasowane punkty", obraz_dopasowan)
    ]

    for nazwa_okna, obraz in okna:
        cv2.namedWindow(nazwa_okna, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nazwa_okna, 800, 600)
        cv2.imshow(nazwa_okna, obraz)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ============================================================
# Część 3: śledzenie wideo
# ============================================================

def przetworz_wideo(referencja: str, plik_wideo: str, metoda: str):
    obraz_wzorca_bgr = cv2.imread(referencja)
    if obraz_wzorca_bgr is None:
        print(f"BŁĄD: Nie można otworzyć obrazu wzorcowego: {referencja}", file=sys.stderr)
        return

    cap = cv2.VideoCapture(plik_wideo)
    if not cap.isOpened():
        print(f"BŁĄD: Nie można otworzyć pliku wideo: {plik_wideo}", file=sys.stderr)
        return

    obraz_wzorca_szary = cv2.cvtColor(obraz_wzorca_bgr, cv2.COLOR_BGR2GRAY)

    detektor = utworz_detektor(metoda)
    matcher = utworz_matcher(metoda)

    punkty_wzorca, deskryptory_wzorca = wyznacz_cechy(obraz_wzorca_szary, detektor)

    fps = cap.get(cv2.CAP_PROP_FPS)
    opoznienie = int(1000 / fps) if fps and fps > 1 else 20

    while True:
        poprawnie, klatka = cap.read()
        if not poprawnie:
            break

        klatka_szara = cv2.cvtColor(klatka, cv2.COLOR_BGR2GRAY)

        punkty_klatki, deskryptory_klatki = wyznacz_cechy(klatka_szara, detektor)
        dobre_dopasowania = dopasuj_deskryptory(deskryptory_wzorca, deskryptory_klatki, matcher)

        narozniki, maska_inlierow = zlokalizuj_obiekt(
            punkty_wzorca,
            punkty_klatki,
            dobre_dopasowania,
            obraz_wzorca_szary.shape
        )

        wynik = narysuj_obiekt(klatka, narozniki)

        # TODO 7: Dodawanie informacji tekstowych
        cv2.putText(wynik, f"Zgodne cechy: {len(dobre_dopasowania)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if narozniki is not None:
            cv2.putText(wynik, "STATUS: OBIEKT WIDOCZNY", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(wynik, "STATUS: BRAK OBIEKTU", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Sledzenie obiektu w wideo", wynik)

        klawisz = cv2.waitKey(opoznienie) & 0xFF
        if klawisz in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Funkcja główna
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LAB3 - wersja ćwiczeniowa: deskryptory")
    parser.add_argument("--reference", required=True, help="Ścieżka do obrazu wzorcowego, np. saw1.jpg")
    parser.add_argument("--image", help="Ścieżka do obrazu testowego")
    parser.add_argument("--video", help="Ścieżka do pliku wideo, np. sawmovie.mp4")
    parser.add_argument("--method", default="ORB", help="Metoda deskryptorów: ORB, BRISK, SIFT")

    args = parser.parse_args()

    if args.image is None and args.video is None:
        print("BŁĄD: Podaj --image lub --video", file=sys.stderr)
        return 1

    if args.image is not None:
        przetworz_obraz(args.reference, args.image, args.method)

    if args.video is not None:
        przetworz_wideo(args.reference, args.video, args.method)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())