"""LLM prompt templates for taxonomy generation, refinement, review, classification, and new-topic proposal."""

from langchain_core.prompts import ChatPromptTemplate

TAXONOMY_GENERATION_PROMPT = ChatPromptTemplate.from_template(
    """Esti un asistent care analizeaza mentiuni web (articole, postari, comentarii)
despre un brand/proiect si identifica principalele subiecte (topicuri) discutate.

Scop: {use_case}

Mentiuni de analizat:
{documents}

Genereaza o taxonomie de topicuri relevante, care sa acopere cat mai bine subiectele
discutate in aceste mentiuni. Foloseste denumiri SCURTE (2-3 cuvinte, sub 30 de
caractere) si clare, in limba romana - un nume de topic nu e o propozitie sau o
descriere, e o eticheta scurta. Evita topicuri prea specifice (legate de un
singur eveniment) sau prea generale (ex. "Diverse", "Altele").

Un topic descrie UN ASPECT sau O TEMA a discutiei (ex. "Probleme tehnice",
"Preturi si abonamente"), NU o entitate specifica mentionata in text:
- NU crea topicuri separate per nume de platforma/produs/concurent (ex. NU
  "Spotify", "Tidal", "Deezer" ca topicuri distincte) - grupeaza toate
  mentiunile de acest tip sub UN SINGUR topic tematic (ex. "Comparatie cu
  concurenti"), indiferent care platforma anume e numita.
- NU crea un topic cu numele brandului/proiectului monitorizat el insusi -
  daca observi ca un nume apare in aproape toate mentiunile, acela e brandul
  monitorizat, nu un topic (toate mentiunile sunt deja despre el, deci un
  topic cu numele lui nu distinge nimic).

Exemplu (proiect fictiv, doar pentru format):
Mentiuni: "X lanseaza un produs nou cu functii asteptate de clienti.",
"Clientii se plang de intarzieri la livrarea comenzilor X.",
"X a semnat un parteneriat cu o firma de logistica.",
"Recenzie: X e mai ieftin decat Y si Z, dar Y are mai multe functii."
Raspuns BUN: <topic>Lansari produse</topic> <topic>Probleme livrare</topic>
<topic>Parteneriate</topic> <topic>Comparatie cu concurenti</topic>
Raspuns DE EVITAT: <topic>Lansarea de produse noi si reactiile clientilor la
functionalitatile prezentate</topic> (prea lung, e o descriere, nu o eticheta);
<topic>X</topic> (X e brandul monitorizat, nu un topic); <topic>Y</topic>,
<topic>Z</topic> (nume de concurenti in loc de un topic tematic unificat)

Inainte de raspunsul final, scrie 1-2 propozitii (text simplu, fara XML) despre
ce teme majore ai identificat in mentiuni - te ajuta sa alegi denumiri mai
precise. Apoi raspunde STRICT in acest format XML, fara alt text dupa el:
<topics>
    <topic>Nume topic 1</topic>
    <topic>Nume topic 2</topic>
</topics>

Nu depasi {max_num_clusters} topicuri in total.
"""
)

TAXONOMY_REVIEW_PROMPT = ChatPromptTemplate.from_template(
    """Ai deja o taxonomie de topicuri, generata pe un esantion mic de mentiuni.
Acum primesti un esantion nou, mai mare, de mentiuni din acelasi proiect.

Taxonomia curenta:
{current_taxonomy}

Mentiuni noi de verificat:
{documents}

Verifica daca taxonomia curenta acopera bine subiectele din acest esantion nou.
Poti: pastra topicurile care se potrivesc bine, elimina/fuziona topicuri
redundante sau prea specifice, si adauga topicuri noi DOAR daca subiecte
importante lipsesc clar din taxonomia curenta.

Doua topicuri sunt REDUNDANTE cand descriu practic acelasi lucru cu alte
cuvinte (ex. "Migrare playlist-uri" si "Transfer muzica intre servicii" ar
trebui fuzionate intr-unul singur; la fel, "Preturi", "Modele de pret" si
"Promotii si reduceri" descriu cu totii latura financiara si ar trebui sa
fie UN SINGUR topic, nu trei) - nu tine ambele/toate doar pentru ca
formularea difera putin.

Un topic descrie UN ASPECT/O TEMA, NU o entitate specifica mentionata in
text:
- Daca taxonomia contine topicuri separate per nume de platforma/produs/
  concurent (ex. "Spotify", "Tidal", "Deezer"), FUZIONEAZA-le intr-un singur
  topic tematic (ex. "Comparatie cu concurenti") - nu tine unul per nume.
- Daca taxonomia contine un topic cu numele brandului/proiectului monitorizat
  el insusi, ELIMINA-l - toate mentiunile sunt deja despre el, nu distinge
  nimic.

Daca gasesti topicuri cu denumiri lungi (propozitii sau descrieri, nu etichete),
REFORMULEAZA-le mai scurt (2-3 cuvinte, sub 30 de caractere) in taxonomia finala -
nu le pastra asa cum sunt doar pentru ca deja existau.

Inainte de raspunsul final, scrie 1-2 propozitii (text simplu, fara XML) despre
ce ai pastrat, fuzionat, eliminat sau adaugat si de ce. Apoi raspunde STRICT in
acest format XML, cu taxonomia FINALA (nu doar diferentele), fara alt text dupa:
<topics>
    <topic>Nume topic 1</topic>
    <topic>Nume topic 2</topic>
</topics>

Nu depasi {max_num_clusters} topicuri in total.
"""
)

TAXONOMY_UPDATE_PROMPT = ChatPromptTemplate.from_template(
    """Ai deja o taxonomie de topicuri, construita pana acum din mentiuni
anterioare ale aceluiasi proiect:

{current_taxonomy}

Primesti un nou mini-batch de mentiuni. Verifica daca taxonomia curenta
acopera bine subiectele din acest batch. Poti pastra, fuziona sau elimina
topicuri redundante, si adauga topicuri noi DOAR daca subiecte importante
lipsesc clar.

Doua topicuri sunt REDUNDANTE cand descriu practic acelasi lucru cu alte
cuvinte (ex. "Migrare playlist-uri" si "Transfer muzica intre servicii" ar
trebui fuzionate intr-unul singur; la fel, "Preturi", "Modele de pret" si
"Promotii si reduceri" descriu cu totii latura financiara si ar trebui sa
fie UN SINGUR topic, nu trei) - nu tine ambele/toate doar pentru ca
formularea difera putin.

Un topic descrie UN ASPECT/O TEMA, NU o entitate specifica mentionata in
text:
- Daca taxonomia contine topicuri separate per nume de platforma/produs/
  concurent (ex. "Spotify", "Tidal", "Deezer"), FUZIONEAZA-le intr-un singur
  topic tematic (ex. "Comparatie cu concurenti") - nu tine unul per nume.
- Daca taxonomia contine un topic cu numele brandului/proiectului monitorizat
  el insusi, ELIMINA-l - toate mentiunile sunt deja despre el, nu distinge
  nimic.

Topicurile trebuie sa fie etichete SCURTE (2-3 cuvinte, sub 30 de caractere),
nu propozitii sau descrieri - daca gasesti unul prea lung, reformuleaza-l
mai scurt in taxonomia finala.

Mentiuni noi:
{documents}

Inainte de raspunsul final, scrie 1-2 propozitii (text simplu, fara XML):
1) acopera taxonomia curenta bine acest batch? 2) ce ai schimbat si de ce.
Apoi raspunde STRICT in acest format XML, cu taxonomia FINALA (nu doar
diferentele), fara alt text dupa:
<topics>
    <topic>Nume topic 1</topic>
    <topic>Nume topic 2</topic>
</topics>

Nu depasi {max_num_clusters} topicuri in total.
"""
)

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_template(
    """Ai urmatoarea taxonomie de topicuri, deja existenta pentru acest proiect:
{taxonomy}

Clasifica fiecare mentiune de mai jos, folosind DOAR topicurile din taxonomia
de mai sus. O mentiune poate avea unul sau mai multe topicuri aplicabile.

Fiecare mentiune este precedata de identificatorul ei, in formatul [id=XXX].
COPIAZA EXACT acest identificator (fara "id=", doar valoarea) in campul
"id" din raspunsul tau - este critic sa se potriveasca exact, altfel
clasificarea nu poate fi asociata cu mentiunea corecta.

Pentru fiecare mentiune, da un scor de incredere (0.0 - 1.0), calibrat astfel:
- 0.9 - 1.0: mentiunea e clar despre topicul ales, fara ambiguitate.
- 0.7 - 0.89: se potriveste bine, dar nu e evident/central in mentiune.
- 0.5 - 0.69: potrivire partiala sau incerta - topicul e doar tangential.
- sub 0.5: mentiunea pare sa vorbeasca despre altceva, neacoperit de taxonomia
  curenta.

NU forta o mentiune intr-un topic doar ca sa dai un raspuns "sigur" - un scor
mic e un raspuns valid si util (arata ca taxonomia trebuie extinsa), mult mai
util decat o incadrare fortata la un topic care nu se potriveste cu adevarat.

Mentiuni de clasificat:
{documents}

Raspunde STRICT ca JSON, o lista de obiecte, fara alt text inainte sau dupa,
cu EXACT atatea obiecte cate mentiuni sunt mai sus:
[
  {{
    "id": "valoarea exacta din [id=XXX] al mentiunii",
    "primary_category": "topicul principal",
    "categories": ["topic1", "topic2"],
    "confidence": 0.0,
    "explanation": "motiv scurt, maxim 10-12 cuvinte"
  }}
]
"""
)

TAXONOMY_EVALUATION_PROMPT = ChatPromptTemplate.from_template(
    """Esti un evaluator care compara doua variante ale unei taxonomii de
topicuri, ca sa decizi care se potriveste mai bine cu un esantion de
mentiuni si cu cerintele de mai jos.

Scop: {use_case}
Numar maxim de topicuri permis: {max_num_clusters}

Varianta 1:
{taxonomy_a}

Varianta 2:
{taxonomy_b}

Esantion de mentiuni pentru evaluare:
{documents}

Compara cele doua variante dupa:
1. Cat de bine acopera subiectele din esantion, fara suprapuneri sau
   contradictii intre topicuri.
2. Daca respecta numarul maxim de topicuri.
3. Daca denumirile sunt scurte (2-3 cuvinte), clare si fara ambiguitate.

Alege varianta mai buna, chiar daca diferenta e mica. Scrie mai intai o
singura propozitie (text simplu) cu motivul alegerii, apoi raspunde STRICT
in acest format XML, fara alt text dupa el:
<evaluation>
    <better_variant>1 sau 2</better_variant>
</evaluation>
"""
)

NEW_TOPIC_PROPOSAL_PROMPT = ChatPromptTemplate.from_template(
    """Taxonomia existenta a acestui proiect este:
{taxonomy}

Urmatoarele mentiuni NU s-au potrivit bine cu niciun topic din taxonomia
existenta (scor de incredere scazut la clasificare):

{documents}

Analizeaza daca aceste mentiuni au in comun unul sau mai multe subiecte
noi, coerente, care lipsesc din taxonomia existenta. Propune topicuri noi
DOAR daca exista un tipar clar, sustinut de mai multe mentiuni - ignora
cazurile izolate sau ambigue.

Exemplu: daca 8 din 20 de mentiuni vorbesc despre acelasi subiect nou
(ex. o problema tehnica recurenta), e un tipar clar - propune un topic. Daca
doar 1-2 mentiuni ating un subiect, fara legatura clara intre ele, NU propune
nimic pentru atat de putine cazuri izolate.

Topicurile noi trebuie sa fie etichete SCURTE (2-3 cuvinte, sub 30 de
caractere), nu propozitii sau descrieri.

Un topic nou descrie UN ASPECT/O TEMA, NU o entitate specifica. NU propune un
topic care e doar numele unei platforme/produs/concurent mentionat in
mentiuni (daca observi asta, tiparul e mai degraba "comparatie cu concurenti"
sau similar - un topic tematic unic, nu unul per nume). NU propune niciodata
un topic cu numele brandului/proiectului monitorizat el insusi.

Daca NU gasesti niciun subiect nou coerent, raspunde cu o lista goala.

Scrie mai intai 1-2 propozitii (text simplu) despre tiparul gasit (sau lipsa
lui), apoi raspunde STRICT in acest format XML, fara alt text dupa el:
<topics>
    <topic>Nume topic nou 1</topic>
</topics>
"""
)
