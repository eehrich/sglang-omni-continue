# Continuation: zwei Effekte, nicht einer

## Ausgangspunkt

Acht zusammenhaengende Segmente aus einer echten Szene (51-69 Woerter, Soft-Cap
der Pipeline ist 75), identischer Text, Seed 46. Dieselben Basis-Codes, dieselbe
Kette, einmal ueber `transformers.generate()` und einmal ueber sglang.

Leseurteil (1-10):

    transformers, continuation, 1.7B          8   fast perfekt
    sglang,       continuation, 1.7B          2   fremde Stimmen im Durchlauf
    sglang,       continuation, 8B            2   dito
    sglang,       clone-foermiges Fenster     7   wie bisher, mit dem Restdrift

Referenzlaenge (13 s gegen 51,7 s) macht am Urteil nichts.

## Was die Messung findet

Zwei getrennte Effekte, die sich im Hoereindruck addieren.

**Effekt A -- Prefixlaenge, beide Engines.** Je kuerzer das Prefix, desto hoeher
die Stimme. Gleicher Text, gleiche Seeds, nur das Prefix auf seine letzten N
Frames gekuerzt, F0-Median aus zwei Seeds:

    Prefix   sglang    transformers   Abstand
     40      135,6       123,1         +12,5
     80      123,5       111,7         +11,8
    163       96,8        81,9         +14,9

Das ist eine Modelleigenschaft, keine sglang-Sache: die Referenz steigt genauso.
Bei 40 Frames (3,2 s) liegen beide rund 40 Hz ueber dem Ziel. Deckt sich mit dem,
was fuer gefensterte Referenzen auf dem Delay-Modell notiert ist (738ce79,
1e9f1f8).

**Effekt B -- konstanter Versatz, nur sglang.** Der Abstand zwischen den Engines
bleibt ueber alle drei Prefixlaengen bei rund 13 Hz. Er skaliert nicht, er sitzt
additiv obendrauf.

Der Versatz ist gesichert, nicht gewuerfelt. Segment eins allein, sechs Seeds je
Engine:

    sglang   Median 96,8   Spanne 94,9-109,1   Streuung 4,9
    HF       Median 82,8   Spanne 76,9- 87,6   Streuung 3,5

Die Verteilungen ueberlappen nicht -- sglangs kleinster Wert liegt ueber HFs
groesstem. Unter der Annahme gleicher Verteilung ist das p ~ 0,002. Das war
noetig zu pruefen: die Take-zu-Take-Streuung dieses Modells ist mit 89-113 Hz
breiter als der Abstand, eine Ziehung je Seite haette nichts entschieden.

## Was als Ursache ausscheidet

Alles gemessen, nicht gelesen:

**Prompt-Konstruktion.** Greedy (`top_k=1`) ueber sechs Texte: drei davon im
ersten erzeugten Frame in allen zwoelf Codebooks bitgleich, die anderen weichen
nur in den hinteren Kanaelen ab (11 dreimal, 10 und 7 je einmal). Der Local
Transformer erzeugt die Kanaele nacheinander, jeder auf den vorherigen bedingt,
also kippen bei knappen Entscheidungen die hinteren zuerst -- unterschiedliche
Attention-Kernel reichen dafuer. Ein struktureller Fehler wuerde keine exakten
Treffer zulassen. Die Naht ist in Ordnung.

**Vocoder.** Dieselben Codes zweimal dekodiert, einmal von sglang, einmal von der
Referenz: Korrelation 0,9969 bis 0,9990, RMS und Bandverteilung identisch.
sglangs Audio gibt genau das wieder, was in seinen Codes steht.

**Wiederholungsstrafe.** Kette mit 1,05 und mit 1,0: 103,9 gegen 104,8 Hz. Keine
Bewegung. Die Vermutung lag nahe, weil die Gruppierung auf dem Delay-Modell
genau so aussah (844b933, 115,5 -> 99,0 Hz) und `moss_tts_local` die Strafe
strikt pro Kanal fuehrt -- sie traegt hier trotzdem nicht.

**Sampler.** Temperatur 1,7 gegen 0,5: Abstand +15,5 gegen +12,8 Hz, die Luecke
schliesst sich nicht. Entropie der gesampelten Codes je Codebook: Mittel -0,02
bit Unterschied, kanalweise hoechstens 0,08. sglang sampelt genauso vielfaeltig
wie die Referenz.

**Treiber und Parameter.** Beide Ketten bauen `prefix = concat(Basis,
Vorgaenger)`, konkatenieren den Text gleich und senden `token_count` als
Gesamtwert. Sampling-Werte identisch (1,7 / 0,80 / 17 / 1,05). `token_count`
fliesst in sglang ausschliesslich in `build_user_message`, steuert also keine
Generierung.

**Audio-History fuer die Strafe.** Wird bei Continuation nicht aus dem Prompt
vorbelegt -- aber `audio_token_presence` dient nur der Wiederholungsstrafe
(`state_pool.py:227`), kann also keine fremde Stimme erzeugen.

## Ein Befund, der noch keine Erklaerung hat

Greedy gegen gesampelt, dasselbe Segment:

    greedy      HF 94,5   sglang 98,4   Abstand  3,9
    gesampelt   HF 82,8   sglang 96,8   Abstand 14,0

Unter Greedy landen beide hoch und dicht beieinander. Sampling zieht die
Referenz um 12 Hz nach unten, sglang nur um 1,5 -- bei nachweislich gleicher
Entropie. Beide Engines explorieren also gleich weit, aber sglangs Ziehungen
bleiben in einer hoeheren Region. Das ist der schaerfste Hinweis, den die
Messung hergibt, und die naechste Suche sollte dort ansetzen: nicht wie breit
gesampelt wird, sondern wohin.

## Was das fuer die Pipeline heisst

Der Restdrift, den wir loswerden wollen, kommt aus dem clone-foermigen Fenster,
das die Produktion faehrt (`assistant_slot_continuation: false`). Continuation
waere der Weg dahin -- der Referenzpfad zeigt mit 81,9 Hz bei 163 Frames Prefix,
dass es geht.

Zwei Bedingungen dafuer:

1. **Das Prefix muss lang sein.** Effekt A ist groesser als Effekt B: bei 40
   Frames liegt selbst die Referenz 40 Hz zu hoch. Der Modus
   `sliding_window_ref_anchor` (Basisaufnahme + letzte Segmente) erfuellt das
   von sich aus, `sliding_window` mit nur dem letzten Segment nicht zwingend.

2. **Effekt B muss weg**, sonst bleibt sglang rund 13 Hz ueber der Referenz --
   anderthalb Halbtoene, hoerbar als andere Stimme.

## Serverzustand

`sglang-moss.service` wird von allen Messkripten gestoppt und per `trap` wieder
gestartet. Nach einem harten Abbruch pruefen, ob er laeuft -- sonst hat die
Pipeline kein TTS-Backend.
