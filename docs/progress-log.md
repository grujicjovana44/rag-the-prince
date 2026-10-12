# Progress log

## RAGAS evaluacija: četiri metrike

- Cilj: dodati odvojenu evaluaciju stvarnih Prince RAG odgovora sa četiri tražene RAGAS metrike: Faithfulness, Answer Relevancy, Context Precision i Context Recall.
- Kreirani fajlovi: `src/evaluation.py`, `scripts/evaluate_ragas.py`, `data/ragas_eval.json`.
- Izmenjeni fajlovi: `requirements.txt` (dodate direktne runtime zavisnosti i kompatibilnosni pin), `README.md` (komanda, značenje metrika i upozorenje o Gemini API pozivima), `docs/homework-checklist.md` (status implementacije i neizvršene evaluacije), `docs/progress-log.md` (ovaj unos).
- Dataset sadrži četiri podržana pitanja i ručno napisane referentne odgovore. Evaluator pokreće postojeći Prince pipeline, šalje odgovore i dobavljene tekstove u RAGAS i zapisuje stvarne pojedinačne/prosečne rezultate i korišćene dokaze u `evidence/ragas-metrics.json`. Nema unapred upisanih ili izmišljenih skorova.
- Izabrane metrike: Faithfulness (utemeljenost odgovora u kontekstu), Answer Relevancy (relevantnost odgovora za pitanje), Context Precision (rangiranje relevantnih chunk-ova prema referenci) i Context Recall (pokrivenost činjenica iz reference kontekstom). Answer Relevancy koristi lokalni `all-MiniLM-L6-v2`; ostale metrike koriste RAGAS evaluator preko Google GenAI SDK-a.
- Provera zavisnosti: RAGAS 0.4.3 navodi Python >=3.9; ciljna Python 3.12 verzija je kompatibilna. Aktuelni RAGAS collections API i Google GenAI integracija korišćeni su po dokumentaciji. Import RAGAS 0.4.3 je inicijalno padao uz `langchain-community==0.4.2` zbog uklonjene Vertex AI putanje; okruženje je zato pinovano na 0.4.1, a kompatibilni import je potom uspeo. Pin je zabeležen u `requirements.txt`.
- `.venv/bin/python -m pip install ragas==0.4.3` je prijavio da je RAGAS 0.4.3 već instaliran. `.venv/bin/python -m pip install langchain-community==0.4.1` je izvršena radi kompatibilnosti.
- Stvarni izlaz naknadne API/import introspekcije:

  ```text
  ragas=0.4.3
  langchain-community=0.4.1
  ragas-import=ok
  llm_factory=(model: 'str', provider: 'str' = 'openai', client: 't.Optional[t.Any]' = None, adapter: 'str' = 'auto', cache: 't.Optional[CacheInterface]' = None, **kwargs: 't.Any') -> 'InstructorBaseRagasLLM'
  HuggingFaceEmbeddings=(model: str, use_api: bool = False, api_key: Optional[str] = None, device: Optional[str] = None, normalize_embeddings: bool = True, batch_size: int = 32, cache: Optional[ragas.cache.CacheInterface] = None, **model_kwargs: Any)
  Faithfulness=(llm: 'InstructorBaseRagasLLM', name: str = 'faithfulness', **kwargs)
  AnswerRelevancy=(llm: 'InstructorBaseRagasLLM', embeddings: 'BaseRagasEmbedding', name: str = 'answer_relevancy', strictness: int = 3, **kwargs)
  ContextPrecision=(llm: 'InstructorBaseRagasLLM', **kwargs)
  ContextRecall=(llm: 'InstructorBaseRagasLLM', name: str = 'context_recall', **kwargs)
  ```

- RAGAS evaluacija nije pokrenuta; pokretanje bi slalo više Gemini poziva i stvaralo stvarne skorove. Nisu pokretani testovi i nisu dodavani testovi u ovom koraku. Checklist zato ostavlja neoznačeno hvatanje stvarnog RAGAS izveštaja.
- Izvori: [RAGAS 0.4.3 na PyPI](https://pypi.org/project/ragas/), [RAGAS metrike](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/), [RAGAS Gemini integracija](https://docs.ragas.io/en/stable/howtos/integrations/gemini/), [RAGAS upstream import problem](https://github.com/vibrantlabsai/ragas/issues/2753).

## Završna provera i stvarna RAGAS evaluacija

- Komanda: `.venv/bin/python -m pytest -q -rs`
- Stvarni izlaz:
  ```
  ................................................                         [100%]
  48 passed in 2.59s
  ```
- Lokalna provera zavisnosti i tokenizer-a: RAGAS 0.4.3, langchain-community 0.4.1, sentence-transformers 6.1.0, faiss-cpu 1.15.1, flashrank 0.2.10, google-genai 2.29.0, python-dotenv 1.2.4; uspešno uvezeni FAISS, FlashRank, Google GenAI i RAGAS. Korpus: 27 dokumenata, 265 chunk-ova; najveća dužina 192 tokena sa specijalnim tokenima, 0 chunk-ova preko 220.
- Evaluator je inicijalno padao jer instalirani `instructor` 1.3.2 nema `from_genai`, iako ga RAGAS 0.4.3 adapter očekuje. Povezan je asinhroni OpenAI klijent sa Gemini OpenAI-kompatibilnim endpoint-om; main RAG generation i dalje koristi zvanični Google GenAI SDK. Dodat je direktan pin `openai==1.109.1`.
- Komanda evaluacije: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/evaluate_ragas.py`
- Sažeti stvarni izlaz evaluacije (SDK HTTP logovi i retry poruke su izostavljeni):
  ```
  Generated RAG answer for mercenary_arms.
  Generated RAG answer for hereditary_principalities.
  Not scored fear_and_hatred: Generated citations reference chunks outside retrieved evidence: prince-chapter-xvii-chunk-4, prince-chapter-xix-chunk-0.
  Not scored fortune_as_river: Generated citations reference chunks outside retrieved evidence: prince-chapter-xxv-chunk-1, prince-chapter-xxv-chunk-2.
  RAGAS 0.4.3 evaluated 4 real answers.
  faithfulness: 0.9167
  answer_relevancy: 0.7631
  context_precision: 0.7500
  context_recall: 1.0000
  Results saved to evidence/ragas-metrics.json
  ```
- Dva od četiri odgovora su ocenjena. Preostala dva su sačuvana sa retrieval dokazima i greškom citiranja, ali nisu prosleđena RAGAS metriki. Izveštaj beleži broj ocenjenih/neocenjenih slučajeva i stvarne rezultate po slučaju; skorovi su procene LLM sudije, ne determinističke garancije kvaliteta.
- Puna evaluacija je vratila HTTP 429 za neke sudijske zahteve; SDK ih je ponovio, a četiri metrike su završene. Izveštaj je napravljen bez API ključa u njemu.

## Ispravka validnosti citata u Gemini odgovorima

- Uzrok: Gemini je u slobodnom tekstu ponekad navodio validne chunk ID-jeve iz knjige, ali van konkretnih dobavljenih dokaza. Pipeline je takve odgovore ispravno odbijao radi očuvanja grounding-a.
- Izmenjen `src/generation.py`: odgovor se traži kao JSON lista tvrdnji; Google GenAI `response_json_schema` ograničava `citation_ids` na IDs prosleđenih chunk-ova. Svaka tvrdnja mora da ima citat; adapter dodatno proverava strukturu i ID-jeve pre formatiranja, a postojeća pipeline validacija ostaje.
- Izmenjen `tests/test_generation_pipeline.py`: test proverava da schema sadrži samo dobavljeni ID i da se ID van allowlist-a odbacuje.
- Izmenjen `README.md`: opisuje ograničeno strukturisano generisanje citata.
- Offline provera: `.venv/bin/python -m pytest -q` → `49 passed in 2.60s`.
- Stvarna proba sa četiri RAGAS pitanja ispisala je `Generated RAG answer for ...` za sva četiri: `mercenary_arms`, `hereditary_principalities`, `fear_and_hatred`, `fortune_as_river`. To znači da su odgovori prošli pipeline validaciju citata.
- RAGAS ocenjivanje nakon generisanja nije završeno: Google API je vratio HTTP 429 zbog free-tier limita od 15 zahteva/minut za `gemini-3.1-flash-lite`. Nije generisan nov metric report; `evidence/ragas-metrics.json` ostaje prethodni izveštaj sa 2/4 ocenjena odgovora i ne predstavlja rezultate nove structured-citation probe.

## Jednostavan Gradio interfejs

- Kreiran `app.py`: jednostavan baby-pink UI za Prince RAG, pitanje/odgovor, citati, dobavljeni chunk-ovi i rerank skorovi. Koristi isti pipeline i lokalni cache kao CLI; učitava `.env`; vezan je za `127.0.0.1`, `share=False`.
- Izmenjen `requirements.txt`: dodat `gradio==6.30.0` (Gradio nije bio instaliran u `.venv`; verzija je instalirana nakon neuspešnog sandbox mrežnog pokušaja uz eskalaciju).
- Izmenjen `README.md`: dodat lokalni start command `.venv/bin/python app.py`.
- Gradio konstrukcija proverena stub pipeline-om, bez pokretanja Gemini-ja. Pravi server je startovao i vratio HTTP 200 za `GET /gradio_api/startup-events` i `HEAD /`; server je zaustavljen posle provere.
- Provera posle izmene: `.venv/bin/python -m pytest -q` → `49 passed in 2.71s`.
- Moje odluke i napomene:


## Završna Gemini provera i evidence

- Cilj: upotrebiti Gemini ključ već sačuvan u lokalnom `.env` fajlu da bi se zabeležio podržani odgovor sa citatima, zatim ponoviti ceo test skup i osvežiti checklist.
- Izmenjeni fajlovi: `evidence.md` (stvarni supported Gemini odgovor i izlazi), `docs/homework-checklist.md` (označeni završeni zahtevi Outcome 3 i podržani deo Outcome 5) i `docs/progress-log.md` (ovaj zapis). `.env`, kod, testovi i korpus nisu menjani; vrednost ključa nije ispisana niti upisana u evidence.
- Dependency review: `.venv` je Python 3.12.15; `requirements.txt` sadrži `pytest==8.3.3`, a `google-genai` ranije nije bio instaliran. Zvanični PyPI zapis navodi Python >=3.9, pa je kompatibilan sa Python 3.12. Instalacija `google-genai` odobrena je i izvršena u `.venv`; potvrđen je `google-genai==2.29.0`. Paket nije dodat u `requirements.txt`.
- Provera tajne u `.env` ispisala je samo `GEMINI_API_KEY_in_env_file=True`; vrednost nije prikazana.
- Komanda za testove nakon instalacije SDK-a: `.venv/bin/python -m pytest -q -rs`.
- Stvarni izlaz:

  ```text
  ................................................                         [100%]
  48 passed in 2.22s
  ```

- Evidence komanda je izvršena preko `.venv/bin/python`, uz ključa iz `.env` prosleđenog samo procesu i `--allow-gemini --offline`. Prvi pokušaj iz sandbox-a završio se stvarnom greškom `[Errno 8] nodename nor servname provided, or not known`; ponovljen je uz odobren pristup mreži.
- Stvarni izlaz uspešnog izvršavanja evidence generatora:

  ```text
  Generated answer captured from Gemini.
  ```

- U captured CLI izlazu u `evidence.md` Gemini zahtev za `gemini-3.1-flash-lite` vratio je `HTTP/1.1 200 OK`. Generisani odgovor je prirodno-jezička sinteza, a citat `prince-chapter-ii-chunk-0` pripada dobavljenom kontekstu i prikazan je u validiranom polju `Citations`. Isti evidence run beleži i odbijanje nepodržanog pitanja.
- Evidence test snapshot:

  ```text
  ................................................                         [100%]
  48 passed in 2.18s
  ```

- Izvor kompatibilnosti: [google-genai na PyPI](https://pypi.org/project/google-genai/).
- Moje odluke i napomene:


## Ponovna provera nakon ručne instalacije

- Cilj: ponovo proveriti celu offline testnu bazu i dostupnost Gemini preduslova pre nastavka.
- Fajlovi izmenjeni: `docs/progress-log.md` (ovaj zapis) i `evidence.md` (regenerisano stvarnim offline izvršavanjem skripte). Nisu menjani kod, testovi, korpus niti zavisnosti.
- Komanda: `.venv/bin/python -m pytest -q -rs`.
- Stvarni izlaz:

  ```text
  ................................................                         [100%]
  48 passed in 2.20s
  ```

- Provera Gemini preduslova preko `.venv/bin/python -c`:

  ```text
  google-genai=not importable
  GEMINI_API_KEY_set=False
  GEMINI_MODEL=<default>
  ```

- Nije pokretana instalacija niti Gemini API poziv. Gemini SDK i ključ nisu dostupni, pa podržani primer još nema stvarno generisan odgovor ni validirane citate. Offline lažni LLM testovi prolaze u okviru svih 48 testova.
- Komanda za osvežavanje evidence-a: `.venv/bin/python scripts/generate_evidence.py --offline`.
- Stvarni izlaz:

  ```text
  Evidence written to /Users/jovanagrujic/Downloads/rag-week-6-homework-starter/evidence.md
  INCOMPLETE: the supported query did not produce a Gemini answer. The captured output records the real generation error and retrieved evidence; no answer has been fabricated.
  ```

- Test snapshot koji je skripta zabeležila u `evidence.md`:

  ```text
  ................................................                         [100%]
  48 passed in 2.25s
  ```
- Moje odluke i napomene:


## Korak A: specs/specweek06/PLAN.md

- Cilj: zabeležiti ciljnu RAG arhitekturu, dogovorene odluke i sedam faza sa kriterijumima za završetak.
- Kreiran fajl: `specs/specweek06/PLAN.md`.
- Komande: nijedna u ovoj sesiji. Fajl je kreiran u ranijem koraku; ovde nije ponovo kreiran niti izmenjen. Nema izlaza komande za ovaj unos.
- Moje odluke i napomene:


## Faza 2: chapter-aware chunking i poređenje

- Cilj: sačuvati starter chunker `140/30` i dodati book chunker `800/120` sa granicama reči, overlap-om, granicama poglavlja, stabilnim ID-jevima i metapodacima. Offline sigurnosna granica je 880 karaktera kao približna zaštita za cilj od 220 tokena. Dodata je i provera stvarnog broja tokena preko prosleđenog token-counter interfejsa.
- Izmenjeni/kreirani fajlovi:
  - `src/chunking.py` — named konfiguracije, word-aware split, book chunk builder, metapodaci chunk-a i provera token limita.
  - `tests/test_chunking.py` — offline testovi za overlap, granice reči, ID-jeve, metapodatke, ograničenje dužine i poređenje retrieval-a.
  - `tests/test_chunking_integration.py` — označena integraciona provera stvarnim MiniLM tokenizer-om, samo iz lokalnog cache-a.
  - `pytest.ini` — registracija oznake `integration`.
  - `docs/progress-log.md` — ovaj unos.
- Nisu menjani `data/book.txt`, `src/corpus.py`, CLI ni postojeći test slučajevi; `tests/test_chunking.py` je dopunjen novim testovima. Nisu instalirani paketi.

### Testovi

- Komanda: `.venv/bin/python -m pytest -q`.
- Stvarni izlaz:

  ```text
  ........s................                                                [100%]
  24 passed, 1 skipped in 0.06s
  ```

- Komanda za prikaz razloga preskakanja: `.venv/bin/python -m pytest -q -rs`.
- Stvarni izlaz:

  ```text
  ........s................                                                [100%]
  =========================== short test summary info ============================
  SKIPPED [1] tests/test_chunking_integration.py:9: optional tokenizer dependency is not installed
  24 passed, 1 skipped in 0.05s
  ```

- Provera dostupnosti opcionog tokenizer-a preko `.venv/bin/python -c` dala je:

  ```text
  {'transformers': False, 'sentence_transformers': False}
  {'/Users/jovanagrujic/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2': True, '/Users/jovanagrujic/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L12-v2': False}
  ```

### Outcome 1: stvarno poređenje retrieval-a

- Isti upit: `What qualities are praised or blamed in men and princes?` nad poglavljem XV.
- Komanda: `.venv/bin/python -c` koja izgradi oba chunk skupa i pozove `retrieve_top_k(..., k=2)`.
- Stvarni izlaz:

  ```text
  CONFIG=starter-140-30 chunks=27
    rank=1 id=prince-chapter-xv-chunk-0 score=0.307692 chars=140 text='CONCERNING THINGS FOR WHICH MEN, AND ESPECIALLY PRINCES, ARE PRAISED OR BLAMED It remains now to see what ought to be the rules of conduct f'
    rank=2 id=prince-chapter-xv-chunk-11 score=0.222222 chars=140 text='ich are real, I say that all men when they are spoken of, and chiefly princes for being more highly placed, are remarkable for some of those'
  CONFIG=book-800-120 chunks=5
    rank=1 id=prince-chapter-xv-chunk-0 score=0.114865 chars=796 text='CONCERNING THINGS FOR WHICH MEN, AND ESPECIALLY PRINCES, ARE PRAISED OR BLAMED It remains now to see what ought to be the rules of conduct for a prince towards subject and friends.'
    rank=2 id=prince-chapter-xv-chunk-1 score=0.099338 chars=800 text='how one lives is so far distant from how one ought to live, that he who neglects what is done for what ought to be done, sooner effects his ruin than his preservation; for a man wh'
  ```

- Napomena: stvarna provera tokenizer-a nije izvršena jer `transformers` nije instaliran. Integracioni test je lokalni i označen; ne preuzima model niti poziva mrežu.
- Završna provera posle izmene loga, komanda `.venv/bin/python -m pytest -q`, dala je:

  ```text
  ........s................                                                [100%]
  24 passed, 1 skipped in 0.05s
  ```

- Moje odluke i napomene:

## Ponovna provera Faze 2 nakon instaliranja transformers

- Cilj: ponovo pokrenuti ceo skup testova sa dostupnim `transformers` i proveriti veličine chunk-ova stvarnim tokenizerom `sentence-transformers/all-MiniLM-L6-v2`.
- Fajlovi izmenjeni u ovom unosu: `docs/progress-log.md`.
- Fajlovi pročitani/provereni: `src/corpus.py`, `src/chunking.py`, testovi Faze 2 i `data/book.txt`. `data/book.txt`, `src/` i `tests/` nisu menjani.
- Nije pokretana instalacija paketa.

### Testovi

- Komanda: `.venv/bin/python -m pytest -q -rs`
- Stvarni izlaz:

  ```text
  .........................                                                [100%]
  25 passed in 1.36s
  ```

Tokenizer integracioni test se izvršio i prošao; ovaj put nije bio preskočen.

### Provera stvarnog tokenizer limita

- Komanda: `.venv/bin/python -c 'from transformers import AutoTokenizer; from src.corpus import load_prince_corpus; from src.chunking import build_book_chunks; t=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True); docs=load_prince_corpus(); chunks=build_book_chunks(docs); vals=[(len(t.encode(c.text, add_special_tokens=True)), c) for c in chunks]; worst=max(vals, key=lambda x:x[0]); bad=[(n,c.chunk_id) for n,c in vals if n>220]; print(f"documents={len(docs)} chunks={len(chunks)}"); print(f"max_chars={max(len(c.text) for c in chunks)}"); print(f"max_tokens_with_special_tokens={worst[0]} chunk_id={worst[1].chunk_id}"); print(f"over_220_tokens={len(bad)}"); print(f"over_limit={bad}")'`
- Stvarni izlaz:

  ```text
  [transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
  documents=27 chunks=265
  max_chars=800
  max_tokens_with_special_tokens=192 chunk_id=prince-chapter-iii-chunk-14
  over_220_tokens=0
  over_limit=[]
  ```

- Moje odluke i napomene:

## Faza 1: corpus loader

- Cilj: učitati posvetu i poglavlja I–XXVI iz `data/book.txt`, isključiti Gutenberg okvir, sadržaj, Marriottov uvod i dodatke, i prijaviti neočekivanu strukturu.
- Fajlovi iz ranije implementacije: izmenjen `src/corpus.py`; kreiran `tests/test_prince_corpus.py`. `data/book.txt` nije menjan.
- Izvršenje implementacije: nije izvršeno u ovoj sesiji; fajlovi su izmenjeni u ranijem koraku.
- Komanda za proveru izvršena u ovoj sesiji:

  ```text
  .venv/bin/python -m pytest -q
  ```

- Stvarni izlaz:

  ```text
  .............                                                            [100%]
  13 passed in 0.02s
  ```

- Moje odluke i napomene:

## Faza 1b: uklanjanje fusnota

- Cilj: ukloniti definicije fusnota do prve prazne linije i izolovane inline oznake `[n]`, uz očuvanje Machiavellijevog teksta, poglavlja, numerisanih stavki i završne atribucije XXVI.
- Izmenjeni fajlovi:
  - `src/corpus.py` — uklanjanje definicija i inline oznaka, beleženje broja uklanjanja i greška ako se brojevi definicija i inline oznaka ne poklapaju.
  - `tests/test_prince_corpus.py` — testovi za uklonjene oznake, parnost, poznate fusnote, očuvanje sadržaja, XVIII i završetke poglavlja.
  - `docs/progress-log.md` — ovaj unos.
- Nisu menjani: `data/book.txt`, postojeći profili, CLI i Faza 2 nije započeta.

### Dijagnostika sirovih fusnota

- Komanda: `.venv/bin/python -c` nad `data/book.txt`; ispisane su tražene fizičke linije i nastavci do prve prazne linije.
- Stvarni izlaz:

  ```text
  645:  [1] Duke Lodovico was Lodovico Moro, a son of Francesco Sforza, who
  646:  married Beatrice d’Este. He ruled over Milan from 1494 to 1500, and
  647:  died in 1510.
  648:
  1651:  [1] Nabis, tyrant of Sparta, conquered by the Romans under Flamininus
  1652:  in 195 B.C.; killed 192 B.C.
  1653:
  1654:  [2] Messer Giorgio Scali. This event is to be found in Machiavelli’s
  1655:  “Florentine History,” Book III.
  1656:
  1970:  [4] Giovanni Acuto. An English knight whose name is Sir John
  1971:  Hawkwood. He fought in the English wars in France, and was knighted by
  1972:  Edward III; afterwards he collected a body of troops and went into
  1973:  Italy. These became the famous “White Company.” He took part in many
  1974:  wars, and died in Florence in 1394. He was born about 1320 at Sible
  1975:  Hedingham, a village in Essex. He married Domnia, a daughter of
  1976:  Bernabo Visconti.
  1977:
  2153:  [5] “Many speakers to the House the other night in the debate on the
  2154:  reduction of armaments seemed to show a most lamentable ignorance of
  2155:  the conditions under which the British Empire maintains its existence.
  2156:  When Mr Balfour replied to the allegations that the Roman Empire sank
  2157:  under the weight of its military obligations, he said that this was
  2158:  ‘wholly unhistorical.’ He might well have added that the Roman power
  2159:  was at its zenith when every citizen acknowledged his liability to
  2160:  fight for the State, but that it began to decline as soon as this
  2161:  obligation was no longer recognised.”—_Pall Mall Gazette_, 15th May
  2162:  1906.
  2163:
  2432:  [2] . . . against my will, my fate
  2433: A throne unsettled, and an infant state,
  2434: Bid me defend my realms with all my pow’rs,
  2435: And guard with these severities my shores.
  2436:
  2514:  [1] “The present chapter has given greater offence than any other
  2515:  portion of Machiavelli’s writings.” Burd, “Il Principe,” p. 297.
  2516:
  3110:  [1] Catherine Sforza, a daughter of Galeazzo Sforza and Lucrezia
  3111:  Landriani, born 1463, died 1509. It was to the Countess of Forli that
  3112:  Machiavelli was sent as envoy on 1499. A letter from Fortunati to the
  3113:  countess announces the appointment: “I have been with the signori,”
  3114:  wrote Fortunati, “to learn whom they would send and when. They tell me
  3115:  that Nicolo Machiavelli, a learned young Florentine noble, secretary
  3116:  to my Lords of the Ten, is to leave with me at once.” _Cf_. “Catherine
  3117:  Sforza,” by Count Pasolini, translated by P. Sylvester, 1898.
  3118:
  ```

### Testovi

- Komanda: `.venv/bin/python -m pytest -q` (prva proba nakon izmene testa).
- Stvarni izlaz:

  ```text
  ...........F......                                                     [100%]
  =================================== FAILURES ===================================
  ______ test_prince_corpus_removes_known_footnotes_but_preserves_book_text ______
  
      def test_prince_corpus_removes_known_footnotes_but_preserves_book_text() -> None:
          documents = load_prince_corpus(BOOK_PATH)
          by_label = {document.chapter_label: document for document in documents}
  
          assert "Hiero II, born about 307 B.C." not in by_label["VI"].text
          assert "Duke Lodovico to raise insurrections" in by_label["III"].text
  >       assert "passed into Italy, this country was" in by_label["XII"].text
  E       AssertionError: assert 'passed into Italy, this country was' in 'HOW MANY KINDS OF SOLDIERY THERE ARE, AND CONCERNING MERCENARIES\n\nHaving discoursed particularly on the characteris...ised by them to avoid, as I have said, both fatigue and dangers; thus they have brought Italy to slavery and contempt.'
  E        +  where 'HOW MANY KINDS OF SOLDIERY THERE ARE, AND CONCERNING MERCENARIES\n\nHaving discoursed particularly on the characteris...ised by them to avoid, as I have said, both fatigue and dangers; thus they have brought Italy to slavery and contempt.' = Document(doc_id='prince-chapter-xii', title='The Prince — Chapter XII: HOW MANY KINDS OF SOLDIERY THERE ARE, AND CONCE...DIERY THERE ARE, AND CONCERNING MERCENARIES', part_number=12, footnote_definitions_removed=8, inline_markers_removed=8).text
  
  tests/test_prince_corpus.py:87: AssertionError
  =========================== short test summary info ============================
  FAILED tests/test_prince_corpus.py::test_prince_corpus_removes_known_footnotes_but_preserves_book_text
  1 failed, 17 passed in 0.05s
  ```

- Uzrok i ispravka: konkretna referenca `Italy,[1]` nalazi se u XI, ne XII poglavlju. Test je ispravljen da proverava XI.
- Komanda: `.venv/bin/python -m pytest -q` (ponovljena puna proba).
- Stvarni izlaz:

  ```text
  ..................                                                       [100%]
  18 passed in 0.04s
  ```

- Završna provera posle izmene ovog loga:

  ```text
  Komanda: .venv/bin/python -m pytest -q
  Izlaz:
  ..................                                                       [100%]
  18 passed in 0.03s
  ```

### Jednokratno poređenje iz `/tmp`

- Komanda: jednokratna provera preko `/Users/jovanagrujic/Downloads/rag-week-6-homework-starter/.venv/bin/python -c` iz `/tmp`. Očišćeni loader je korišćen za rezultat posle čišćenja; u memoriji je privremeno očuvan sirovi ulaz radi dužine pre čišćenja. Nijedan fajl nije kreiran tom proverom.
- Stvarni izlaz:

  ```text
  prince-dedication	before=2520	after=2520	definitions=0	inline=0	last120='turn your eyes to these lower regions, you will see how unmeritedly I suffer a great and continued malignity of fortune.'
  prince-chapter-i	before=738	after=734	definitions=0	inline=0	last120='e in freedom; and are acquired either by the arms of the prince himself, or of others, or else by fortune or by ability.'
  prince-chapter-ii	before=1463	after=1459	definitions=0	inline=0	last120=' his rule the memories and motives that make for change are lost, for one change always leaves the toothing for another.'
  prince-chapter-iii	before=17157	after=16474	definitions=6	inline=6	last120='s been brought about either by astuteness or else by force, and both are distrusted by him who has been raised to power.'
  prince-chapter-iv	before=5417	after=5413	definitions=0	inline=0	last120='t occasioned by the little or abundance of ability in the conqueror, but by the want of uniformity in the subject state.'
  prince-chapter-v	before=2535	after=2531	definitions=0	inline=0	last120=' them to allow the memory of their former liberty to rest; so that the safest way is to destroy them or to reside there.'
  prince-chapter-vi	before=6853	after=6795	definitions=1	inline=1	last120='s he was able to build any edifice: thus, whilst he had endured much trouble in acquiring, he had but little in keeping.'
  prince-chapter-vii	before=15371	after=14126	definitions=7	inline=7	last120=' to forget old injuries is deceived. Therefore, the duke erred in his choice, and it was the cause of his ultimate ruin.'
  prince-chapter-viii	before=8903	after=8662	definitions=2	inline=2	last120='l not help you, for they will be considered as forced from you, and no one will be under any obligation to you for them.'
  prince-chapter-ix	before=7257	after=7041	definitions=2	inline=2	last120='ys in every sort and kind of circumstance have need of the state and of him, and then he will always find them faithful.'
  prince-chapter-x	before=3943	after=3939	definitions=0	inline=0	last120='prince to keep the minds of his citizens steadfast from first to last, when he does not fail to support and defend them.'
  prince-chapter-xi	before=5102	after=5004	definitions=2	inline=2	last120='hers made it great in arms, he will make it still greater and more venerated by his goodness and infinite other virtues.'
  prince-chapter-xii	before=11301	after=9476	definitions=8	inline=8	last120='evised by them to avoid, as I have said, both fatigue and dangers; thus they have brought Italy to slavery and contempt.'
  prince-chapter-xiii	before=7988	after=7036	definitions=5	inline=5	last120=' the Great, and many republics and princes have armed and organized themselves, to which rules I entirely commit myself.'
  prince-chapter-xiv	before=4917	after=4835	definitions=1	inline=1	last120='that they may be available to him in adversity, so that if fortune chances it may find him prepared to resist her blows.'
  prince-chapter-xv	before=2942	after=2938	definitions=0	inline=0	last120='lowed, would be his ruin; whilst something else, which looks like vice, yet followed brings him security and prosperity.'
  prince-chapter-xvi	before=4506	after=4502	definitions=0	inline=0	last120='be compelled through seeking a reputation for liberality to incur a name for rapacity which begets reproach with hatred.'
  prince-chapter-xvii	before=6168	after=5900	definitions=2	inline=2	last120='self on that which is in his own control and not in that of others; he must endeavour only to avoid hatred, as is noted.'
  prince-chapter-xviii	before=7203	after=5383	definitions=5	inline=5	last120='o both he is most hostile, and either, if he had kept it, would have deprived him of reputation and kingdom many a time.'
  prince-chapter-xix	before=20262	after=19933	definitions=1	inline=1	last120='ound his state, and from Marcus those which are proper and glorious to keep a state that may already be stable and firm.'
  prince-chapter-xx	before=9117	after=8555	definitions=1	inline=1	last120=' as well as him who does not, and I shall blame whoever, trusting in them, cares little about being hated by the people.'
  prince-chapter-xxi	before=8341	after=7069	definitions=1	inline=1	last120='berality; nevertheless, always maintaining the majesty of his rank, for this he must never consent to abate in anything.'
  prince-chapter-xxii	before=2502	after=2498	definitions=0	inline=0	last120='sed, they can trust each other, but when it is otherwise, the end will always be disastrous for either one or the other.'
  prince-chapter-xxiii	before=4047	after=3825	definitions=1	inline=1	last120='sels, whencesoever they come, are born of the wisdom of the prince, and not the wisdom of the prince from good counsels.'
  prince-chapter-xxiv	before=2881	after=2877	definitions=0	inline=0	last120='ch does not depend upon yourself; those only are reliable, certain, and durable that depend on yourself and your valour.'
  prince-chapter-xxv	before=6699	after=6467	definitions=1	inline=1	last120='ays, woman-like, a lover of young men, because they are less cautious, more violent, and with more audacity command her.'
  prince-chapter-xxvi	before=8014	after=7721	definitions=2	inline=2	last120='hall put to flight: For the old Roman valour is not dead, Nor in th’ Italians’ brests extinguished.\n\nEdward Dacre, 1640.'
  ```

- Moje odluke i napomene:

## Faza 3: embedder interfejs i persistent FAISS store

- Cilj: dodati zamenljiv embedder ugovor, lokalni sentence-transformers adapter i FAISS `IndexFlatIP` skladište nad L2-normalizovanim vektorima, sa trajnim ID mapiranjem i manifestom za otkrivanje zastarelog indeksa.
- Kreiran `src/embeddings.py` — `Embedder` protokol i lenjo učitani `SentenceTransformerEmbedder`; nedostajući paket daje jasnu grešku pri adapteru.
- Kreiran `src/vector_store.py` — normalizacija vektora, cosine pretraga preko `IndexFlatIP`, serijalizacija chunk mapiranja/indeksa/manifest-a i provera hash-a teksta, chunk parametara i modela.
- Kreiran `tests/test_vector_store.py` — offline testovi sa lažnim embedderom i lažnim FAISS modulom za cosine redosled, score, mapiranje, čuvanje/učitavanje i zastareo manifest.
- Izmenjen `.gitignore` — `.cache/` je ignorisan kako dogovoreni generisani indeks ne bi bio commit-ovan.
- Nisu instalirani paketi. U okruženju ovog izvršavanja `transformers` i `numpy` postoje, ali `sentence-transformers` i `faiss-cpu` nisu dostupni; zato pravi model/FAISS adapter nije izvršavan. Testovi koriste samo lažne komponente za taj sloj.

### Testovi

- Komanda: `.venv/bin/python -m pytest -q -rs`
- Stvarni izlaz:

  ```text
  .............................                                            [100%]
  29 passed in 1.64s
  ```

- Moje odluke i napomene:

## Ponovna provera Faze 3 posle ručne instalacije zavisnosti

- Cilj: ponovo pokrenuti punu test-suitu i proveriti lokalni MiniLM + FAISS build, cosine pretragu, upis/učitavanje indeksa.
- Komanda: `.venv/bin/python -m pytest -q -rs`
- Stvarni izlaz:

  ```text
  .............................                                            [100%]
  29 passed in 2.25s
  ```

- Instalirane verzije koje je prijavio `.venv`: `sentence-transformers 6.1.0`, `faiss-cpu 1.15.1`, `transformers 5.19.0`, `numpy 2.5.4`, `flashrank 0.2.10`; Python `3.12.15`. `google-genai` nije dostupan.
- Lokalni checkpoint `sentence-transformers/all-MiniLM-L6-v2` je pronađen. Bez mreže je indeksirano 265 chunk-ova dimenzije 384, obavljena pretraga i round-trip upis/učitavanje. Najviši rezultat za pitanje o plaćeničkim vojnicima bio je `prince-chapter-xii-chunk-0` sa cosine skorom `0.5809`; nakon učitavanja vraćeno je svih 265 mapiranih chunk-ova.
- Komanda za integracionu proveru: jednokratni `.venv/bin/python -u -c` koji postavlja `HF_HUB_OFFLINE=1` i `TRANSFORMERS_OFFLINE=1`, učitava korpus, gradi `build_book_chunks`, pravi `FaissVectorStore`, pretražuje top 4, čuva u privremeni direktorijum, zatim učitava sa očekivanim hash-om, chunk parametrima i imenom modela. Privremeni direktorijum je obrisan po završetku.
- Stvarni izlaz integracione provere:

  ```text
  imports ok
  Loading weights: 100%|██████████| 103/103 [00:00<00:00, 17516.66it/s]
  model ready
  store built 384
  search [('prince-chapter-xii-chunk-0', 0.5809), ('prince-chapter-xiii-chunk-3', 0.5673), ('prince-chapter-xiii-chunk-4', 0.5349), ('prince-chapter-xxvi-chunk-10', 0.5196)]
  round_trip 265
  ```

- Prva probna komanda je uvezla `faiss` pre inicijalizacije sentence-transformers modela i završila se sa exit code 139; zatim je tačno korišćen redosled aplikacije (model se inicijalizuje pre nego što `FaissVectorStore.build` lenjo uveze FAISS) i puna integraciona provera je prošla. To je važno ograničenje redosleda inicijalizacije u ovom okruženju.
- Nisu instalirani paketi u ovoj sesiji. `data/book.txt`, `src/` i `tests/` nisu menjani tokom ove ponovne provere.
- Moje odluke i napomene:


## Faza 4: vector candidates, FlashRank reranking i kalibracija odbijanja

- Cilj: iz FAISS top-N kandidata dobiti FlashRank top-k sa skorovima, sačuvati malu označenu kalibracionu listu i izabrati prag iz stvarnih skorova.
- Kreiran `src/reranking.py` sa FlashRank adapterom, `retrieve_and_rerank`, threshold evaluatorom sa balanced accuracy kriterijumom i pravilom da se pri izjednačenju bira viši prag (konzervativnije odbijanje).
- Kreiran `data/refusal_calibration.json` sa 4 podržana i 4 tematski bliska nepodržana pitanja iz isključenih dodataka/uvoda.
- Kreiran `data/refusal_calibration_scores.json` sa stvarnim modelima, parametrima, chunk ID-ovima, rerank skorovima i izabranim pragom.
- Kreiran `tests/test_reranking.py` za adapter, validaciju ID-jeva, top-N/top-k spajanje, odbijanje bez dokaza i reprodukciju sačuvanog praga.
- FlashRank model je preuzet u ignorisani `.cache/flashrank/` tokom stvarne kalibracije. Nisu instalirani paketi.

### Stvarna kalibracija

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`; chunking `800/120`; FAISS kandidati `N=20`; FlashRank model `ms-marco-TinyBERT-L-2-v2`; kalibracioni skor je skor najboljeg rerankovanog kandidata.
- Prag izabran maksimizovanjem balanced accuracy; izjednačenja biraju viši prag. Rezultat na 8 kalibracionih primera: threshold `0.48179012`, TP=4, TN=4, FP=0, FN=0, balanced accuracy=1.0000.
- Ovaj mali skup služi za prag, ne za nezavisnu procenu kvaliteta. Skorovi i caveat su sačuvani u `data/refusal_calibration_scores.json`.
- Komanda: jednokratni `.venv/bin/python -u -c` koji učitava lokalni korpus i embedder, pravi FAISS indeks u memoriji, izvršava svih osam pitanja kroz top-20 retrieval i FlashRank top-1, pa poziva `calibrate_refusal_threshold`. Transformer cache je u offline režimu; FlashRank težine su već u lokalnom cache-u.
- Stvarni izlaz kalibracije (uz model/ONNX statusne poruke):

  ```text
  Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
  Loading weights: 100%|██████████| 103/103 [00:00<00:00, 17325.58it/s]
  2026-10-11 00:40:42.169 Python[53782:2653616] 2026-10-11 00:40:42.169345 [W:onnxruntime:Default, telemetry_1ds.cc:600 operator()] Failed to persist telemetry device ID; using an in-memory identifier
  question_id	supported	score	top_chunk	top_chapter
  supported_mercenaries	True	0.99769092	prince-chapter-xiii-chunk-3	XIII
  supported_hereditary_states	True	0.99736220	prince-chapter-ii-chunk-0	II
  supported_fear_and_hatred	True	0.99036950	prince-chapter-xvii-chunk-4	XVII
  supported_fortune_river	True	0.95687979	prince-chapter-xxv-chunk-1	XXV
  unsupported_castruccio_biography	False	0.00042990	prince-chapter-vi-chunk-9	VI
  unsupported_valentino_appendix	False	0.00670044	prince-chapter-vii-chunk-3	VII
  unsupported_translator_introduction	False	0.00002541	prince-chapter-vii-chunk-2	VII
  unsupported_historical_dates	False	0.00005190	prince-chapter-viii-chunk-9	VIII
  threshold=0.48179012 TP=4 TN=4 FP=0 FN=0 balanced_accuracy=1.0000
  ```

### Testovi

- Komanda: `.venv/bin/python -m pytest -q -rs`
- Stvarni izlaz:

  ```text
  .....................................                                    [100%]
  37 passed in 2.21s
  ```

- Moje odluke i napomene:


## Faza 5: Gemini generator, citati i orkestracija

- Cilj: dodati Gemini adapter koji koristi `GEMINI_API_KEY` i `GEMINI_MODEL`, traži prirodnu sintezu uz dozvoljene chunk ID citate i bezbedno obrađuje odbijanje, nedostupan LLM i nevalidne citate.
- Kreirani `src/generation.py` i `src/pipeline.py` — grounded prompt, Gemini SDK adapter, validacija svih bracket citata, prag odbijanja pre LLM poziva i rezultat koji čuva dokaze i grešku generisanja bez izmišljanja odgovora.
- Kreiran `tests/test_generation_pipeline.py` — offline fake LLM pokriva sintezu, validne/nevalidne i nedostajuće citate, odbijanje bez LLM poziva, nedostajući ključ i grešku generisanja uz sačuvane dokaze.
- `google-genai` nije instaliran u `.venv`; adapter koristi lazy import. Nije korišćen API ključ, nije bilo Gemini mrežnog poziva, pa stvarni SDK/API runtime nije proveren.

### Testovi

- Komanda: `.venv/bin/python -m pytest -q -rs`
- Stvarni izlaz:

  ```text
  ............................................                             [100%]
  44 passed in 2.19s
  ```

- Moje odluke i napomene:


## Faza 6: CLI integracija i kompatibilnost

- Cilj: dodati Prince workflow u CLI uz očuvanje postojećih `default` i `training-rag` profila, kao i njihovih podrazumevanih starter parametara.
- Izmenjen `src/cli.py` — dodat profil `prince`, book chunk opcije, candidate N, prag i cache putanje; postojeći default chunk size 140, overlap 30 i top-k 2 ostaju isti. Prince izlaz prikazuje generisani odgovor ili grešku, citate, FAISS skor, rerank skor i tekst dokaza.
- Izmenjen `README.md` — dodati Prince CLI primeri i promenljive okruženja za Gemini.
- Kreiran `tests/test_cli.py` — provereni default argumenti, default/training-rag CLI i Prince izlaz sa prikazom rezultata i generation greške.
- Prince CLI pravi/učitava indeks u `.cache/prince-faiss/`; model cache putanje su pod `.cache/`, koji je ignorisan.

### Testovi

- Komanda: `.venv/bin/python -m pytest -q -rs`
- Stvarni izlaz:

  ```text
  ................................................                         [100%]
  48 passed in 2.22s
  ```

### Stvarni CLI smoke testovi

- Podržano pitanje: `What does Chapter II say about hereditary principalities and change?`; Prince profil je vratio `Refused: False`, dokaz iz `prince-chapter-ii-chunk-0`, vector score `0.597500` i rerank score `0.99736220`. Pošto je `GEMINI_API_KEY` za ovu jednokratnu proveru uklonjen iz procesa, CLI je prikazao `Not generated.` i jasnu grešku `GEMINI_API_KEY is not set`, uz sva četiri pronađena dokaza.
- Nepodržano pitanje: `What names of parents were given for Castruccio Castracani, and how was the child raised?`; CLI je vratio `Refused: True` sa pragom `0.48179012`, bez Gemini poziva, i prikazao pronađene dokaze i skorove ispod praga.
- Hugging Face metadata HEAD zahtevi su u prvom CLI pokušaju bili blokirani i retry-ovali. Ponovljeno je uz `HF_HUB_OFFLINE=1` i lokalni cache; uspešan supported CLI izlaz je sačuvao indeks u `.cache/prince-faiss/`. Nije korišćen API ključ i nije bilo Gemini poziva.
- `google-genai` nije instaliran; zato stvarna generisana sinteza još nije proverena. CLI pokazuje traženu generation grešku i dokaze bez izmišljanja odgovora.
- Moje odluke i napomene:


## Faza 7: evidence skripta i završni snapshot

- Cilj: iz stvarnih chunking/retrieval/CLI/test izvršavanja generisati `evidence.md` po Outcome 1–5 strukturi bez izmišljanja odgovora.
- Kreiran `scripts/generate_evidence.py` — snima chunking poređenje, Prince CLI top-k 2/3 izlaze, podržani i nepodržani slučaj, i test output. `--offline` koristi već keširane modele; `--allow-gemini` je potreban da skripta prosledi `GEMINI_API_KEY` Gemini API pozivima. Bez tog prekidača skripta uklanja ključ samo iz svojih CLI child procesa i ne može slučajno pokrenuti plaćeni poziv.
- Kreiran `evidence.md` iz stvarnih izvršavanja. Podržani slučaj ima dokaze i generation error jer `GEMINI_API_KEY` nije prosleđen; nema izmišljenog LLM odgovora ili citata. Zbog toga Outcome 3 i podržani deo Outcome 5 nisu završeni.
- Izmenjen `docs/homework-checklist.md` — čekirano samo ono što je provereno; pair accountability ostaje student-owned.
- Google SDK nije instaliran i nije korišćen API ključ. Skripta je strukturno proverena stvarnim izvršavanjem; poziv sa `--allow-gemini` nije izvršen.

### Evidence i testovi

- Komanda: `.venv/bin/python scripts/generate_evidence.py --offline`
- Stvarni izlaz:

  ```text
  Evidence written to /Users/jovanagrujic/Downloads/rag-week-6-homework-starter/evidence.md
  INCOMPLETE: the supported query did not produce a Gemini answer. The captured output records the real generation error and retrieved evidence; no answer has been fabricated.
  ```

- Skripta je pokrenula testove. Stvarni output snapshot u `evidence.md`:

  ```text
  ................................................                         [100%]
  48 passed in 2.25s
  ```

- Moje odluke i napomene:
## Gemini evidence and RAGAS refresh (2026-10-12)

- `scripts/generate_evidence.py` now loads the ignored `.env` file only when `--allow-gemini` is explicitly supplied. A first attempt exposed sandbox DNS blocking; the command was rerun with approved network access.
- Live command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/generate_evidence.py --offline --allow-gemini`. Gemini returned HTTP 200. `evidence.md` now contains the actual supported answer, its validated Chapter II citations, retrieved evidence, and the current test output (49 passed).
- Live RAGAS command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/evaluate_ragas.py`. All four supported answers passed citation validation and all four metrics were scored for each case. Summary means: Faithfulness 1.0000, Answer Relevancy 0.7718, Context Precision 0.8750, Context Recall 1.0000. Full per-case outputs are in `evidence/ragas-metrics.json`; `evidence.md` includes the summary.
- Updated `README.md` and `docs/homework-checklist.md` to explain key loading, live generation, current evidence, and the completed RAGAS run. No API key was printed or written to evidence.

## Evidencija: grounding, reranking i live provere (2026-10-12)

- Pravilo protiv haluciniranja: kada retrieval ne nađe dovoljno relevantnih dokaza, Prince pipeline odbija odgovor pre Gemini poziva na osnovu kalibrisanog FlashRank praga. Za generisane odgovore citati se ograničavaju na dobavljene chunk-ove i proveravaju pre prikaza. U Gradio interfejsu se za odbijanje/neodgovor više ne prikazuju pronađeni chunk-ovi kao da podržavaju odgovor. Ovo je zaštitni mehanizam, a ne apsolutna garancija da model nikada neće pogrešiti.
- RAGAS istorija (stvarne ranije sačuvane evaluacije): prvi izveštaj ocenio je 2/4 odgovora — Faithfulness 0.9167, Answer Relevancy 0.7631, Context Precision 0.7500, Context Recall 1.0000. Dva slučaja nisu ocenjena zbog citata van retrieved skupa. U poslednjem sačuvanom izveštaju ocenjena su sva 4/4: Faithfulness 1.0000, Answer Relevancy 0.7718, Context Precision 0.8750, Context Recall 1.0000. Pojedinačni rezultati i konteksti su u `evidence/ragas-metrics.json`; dodatni raniji pokušaj generisao je sva četiri odgovora, ali RAGAS scoring nije završen zbog Gemini HTTP 429 i zato nije novi merni izveštaj.
- Razlika nakon dodavanja rerankera: retrieval sada prvo uzima FAISS kandidate (`candidate_n=20`), pa ih FlashRank poređa i prosleđuje top 4 generatoru. U sačuvanom primeru za pitanje o naslednim kneževinama reranker stavlja direktno relevantno Poglavlje II prvo (0.9974), Poglavlje I kao kontekstualnu definiciju drugo (0.9518), a manje direktno Poglavlje III treće (0.3819). U RAGAS izveštajima Context Precision je porastao sa 0.7500 na 0.8750, dok su Faithfulness 0.9167→1.0000, Answer Relevancy 0.7631→0.7718, a Context Recall ostao 1.0000. Broj ocenjenih slučajeva se promenio sa 2 na 4, pa ovo nije kontrolisano poređenje samo efekta rerankera; rezultati su zabeleženi kao istorijske evaluacije, ne kao dokaz uzročnosti.
- Korisnica prijavljuje da je sada prošlo 5 live testova. Zabeleženo kao korisnički prijavljen rezultat; tačni upiti i pojedinačni izlazi nisu dostavljeni, pa ih ovaj zapis ne pripisuje nezavisnoj proveri niti ih izmišlja.
- Izmene dokumentacije i UI iz ovog nastavka nisu pokretale nove live pozive, RAGAS evaluaciju ni test suite.
