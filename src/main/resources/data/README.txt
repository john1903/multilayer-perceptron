LOKALIZACJA UWB

Dane zawierają wyniki pomiarów lokalizacji robota wewnątrz budynku uzyskane za
pomocą technologii UWB. Każdy wynik pomiaru obejmuje 2 wartości: współrzędnej x
oraz współrzędnej y, wyrażone w milimetrach w ustalonym układzie odniesienia.

Pomiary zostały przeprowadzone w dwóch różnych częściach budynku określonych
jako "pod salą F8" i "pod aulą F10", a ich wyniki zostały umieszczone w
osobnych katalogach - odpowiednio 'f8' i 'f10'. W obu przypadkach użyto takiego
samego układu odniesienia, zaś wykonane pomiary były dwojakiego rodzaju: część
stanowiły tzw. pomiary statyczne, zaś pozostałą część - tzw. pomiary
dynamiczne. Podczas pomiarów statycznych robot umieszczany był po kolei w
każdym spośród 225 wyznaczonych punktów i w każdym takim punkcie dokonywana
była seria niezależnych pomiarów. Z kolei podczas pomiarów dynamicznych robot
poruszał się wzdłuż wyznaczonego toru, zaś pomiary odbywały się w trakcie jego
ruchu.

Rozmieszczenie wszystkich punktów statycznych, jak i nałożonego na nie toru
ruchu robota pokazane jest w pliku 'stat_punkty.png'. Ponadto w katalogu 'f8'
oraz w katalogu 'f10' znajduje się plik, który zawiera animację wyników
wszystkich pomiarów statycznych uzyskanych w każdym z wyznaczonych punktów w
danej części budynku - nazywa się on odpowiednio 'f8_stat_pomiary.gif' lub
'f10_stat_pomiary.gif'.

Wyniki pomiarów zostały zapisane w osobnych plikach, które znalazły się w
odpowiednich podkatalogach katalogów 'f8' i 'f10': pliki z wynikami pomiarów
statycznych zostały umieszczone w podkatalogu 'stat', natomiast pliki z
wynikami pomiarów dynamicznych - w podkatalogu 'dyn'.

Nazwa każdego pliku z wynikami pomiarów statycznych składa się z 3 członów
oddzielonych znakiem podkreślenia, które określają odpowiednio:
1. część budynku ('f8' lub 'f10');
2. rodzaj pomiaru ('stat');
3. numer punktu.

Nazwa każdego pliku z wynikami pomiarów dynamicznych również składa się z 3
członów oddzielonych znakiem podkreślenia, które określają odpowiednio:
1. część budynku ('f8' lub 'f10');
2. rodzaj pomiaru ('dyn');
3. numer przejazdu robota wzdłuż toru wraz z określeniem kierunku ruchu ('p' -
   przeciwnie do ruchu wskazówek zegara, 'z' - zgodnie z ruchem wskazówek
   zegara).

Pojedynczy plik z wynikami pomiarów to plik w formacie CSV, w którym każdy
wiersz dotyczy osobnego pomiaru i który składa się z 4 kolumn oddzielonych
przecinkami. Poszczególne kolumny zawierają odpowiednio:
1. zmierzone wartości współrzędnej x [mm];
2. zmierzone wartości współrzędnej y [mm];
3. rzeczywiste wartości współrzędnej x [mm];
4. rzeczywiste wartości współrzędnej y [mm].

Dodatkowo wyznaczone zostały dystrybuanty błędów wyników pomiarów dynamicznych.
Wykres dystrybuanty błędów wszystkich pomiarów dynamicznych zawarty jest w
pliku 'dyn_bledy.png', natomiast w katalogu 'f8' oraz w katalogu 'f10' znajduje
się plik, który zawiera wykres dystrybuanty błędów pomiarów dynamicznych w
danej części budynku - nazywa się on odpowiednio 'f8_dyn_bledy.png' lub
'f10_dyn_bledy.png'.
