# Week 6 RAG evidence

Generated at: `2026-10-12T16:40:21+00:00`

This file is generated from actual executions by `scripts/generate_evidence.py`. Pair contributions and demo lead are intentionally omitted for the students to complete.

## Outcome 1: Chunking impact

The comparison below runs the same query against Chapter XV with the starter, word-aware book, and sentence-aware chunkers.

```text
Question: What qualities are praised or blamed in men and princes?
starter 140/30: chunks=27
  prince-chapter-xv-chunk-0 score=0.307692 text='CONCERNING THINGS FOR WHICH MEN, AND ESPECIALLY PRINCES, ARE PRAISED OR BLAMED It remains now to see what ought to be the rules of conduct f'
  prince-chapter-xv-chunk-11 score=0.222222 text='ich are real, I say that all men when they are spoken of, and chiefly princes for being more highly placed, are remarkable for some of those'
book 800/120: chunks=5
  prince-chapter-xv-chunk-0 score=0.114865 text='CONCERNING THINGS FOR WHICH MEN, AND ESPECIALLY PRINCES, ARE PRAISED OR BLAMED It remains now to see what ought to be the rules of conduct for a prince towards subject and friends. And as I know that '
  prince-chapter-xv-chunk-1 score=0.099338 text='how one lives is so far distant from how one ought to live, that he who neglects what is done for what ought to be done, sooner effects his ruin than his preservation; for a man who wishes to act enti'
sentence-aware 800/120: chunks=5
  prince-chapter-xv-sentence-chunk-0 score=0.176471 text='CONCERNING THINGS FOR WHICH MEN, AND ESPECIALLY PRINCES, ARE PRAISED OR BLAMED It remains now to see what ought to be the rules of conduct for a prince towards subject and friends. And as I know that '
  prince-chapter-xv-sentence-chunk-2 score=0.099237 text='Therefore, putting on one side imaginary things concerning a prince, and discussing those which are real, I say that all men when they are spoken of, and chiefly princes for being more highly placed, '
Qualitative relevance: the first result in all three configurations starts with the Chapter XV heading named by the query and contains the surrounding discussion, so it directly addresses the question. The starter result is precise but cuts off mid-sentence and needs neighboring chunks for fuller context. The larger word-aware result preserves more explanation in one passage. Sentence-aware chunking avoids most sentence breaks and improves readability, though chunk lengths are less uniform and whole-sentence overlap is approximate. The starter's second result is a short overlapping fragment; the word-aware result continues the same argument, while the sentence-aware second result captures a later list of qualities. Those later chunks add context, but rank below the opening passage because they do not contain the chapter's full framing of the question.
The starter retriever uses lexical overlap divided by chunk token count. Scores across chunk sizes are not directly comparable; this is a qualitative comparison, not a measured relevance benchmark.
```

## Outcome 2: Top-k retrieval

Query: What does Chapter II say about hereditary principalities and change?

### top_k=2 output

Command: `'/Users/jovanagrujic/Downloads/rag-week-6-homework-starter/.venv/bin/python' '-m' 'src.cli' '--corpus-profile' 'prince' '--question' 'What does Chapter II say about hereditary principalities and change?' '--top-k' '2'` (exit code 0)

```text
Question:
What does Chapter II say about hereditary principalities and change?

Answer:
Hereditary principalities that are long accustomed to the ruling family are easier to maintain than new states. [prince-chapter-ii-chunk-0] A prince of average ability can maintain his hereditary state by avoiding transgression of ancestral customs and responding prudently to events. [prince-chapter-ii-chunk-0] Only an extraordinary or excessive force can deprive a ruler of an hereditary principality. [prince-chapter-ii-chunk-0]

Corpus profile: prince

Refused: False
Generation error: None
Citations: prince-chapter-ii-chunk-0, prince-chapter-ii-chunk-0, prince-chapter-ii-chunk-0

Retrieved evidence:
[prince-chapter-ii-chunk-0] chapter=II vector_score=0.597500 rerank_score=0.99736220
CONCERNING HEREDITARY PRINCIPALITIES I will leave out all discussion on republics, inasmuch as in another place I have written of them at length, and will address myself only to principalities. In doing so I will keep to the order indicated above, and discuss how such principalities are to be ruled and preserved. I say at once there are fewer difficulties in holding hereditary states, and those long accustomed to the family of their prince, than new ones; for it is sufficient only not to transgress the customs of his ancestors, and to deal prudently with circumstances as they arise, for a prince of average powers to maintain himself in his state, unless he be deprived of it by some extraordinary and excessive force; and if he should be so deprived of it, whenever anything sinister happens
[prince-chapter-i-chunk-0] chapter=I vector_score=0.532172 rerank_score=0.95175695
HOW MANY KINDS OF PRINCIPALITIES THERE ARE, AND BY WHAT MEANS THEY ARE ACQUIRED All states, all powers, that have held and hold rule over men have been and are either republics or principalities. Principalities are either hereditary, in which the family has been long established; or they are new. The new are either entirely new, as was Milan to Francesco Sforza, or they are, as it were, members annexed to the hereditary state of the prince who has acquired them, as was the kingdom of Naples to that of the King of Spain. Such dominions thus acquired are either accustomed to live under a prince, or to live in freedom; and are acquired either by the arms of the prince himself, or of others, or else by fortune or by ability.

Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 18585.21it/s]
INFO:google_genai.models:AFC is enabled with max remote calls: 10.
INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent "HTTP/1.1 200 OK"
```

### top_k=3 output

Command: `'/Users/jovanagrujic/Downloads/rag-week-6-homework-starter/.venv/bin/python' '-m' 'src.cli' '--corpus-profile' 'prince' '--question' 'What does Chapter II say about hereditary principalities and change?' '--top-k' '3'` (exit code 0)

```text
Question:
What does Chapter II say about hereditary principalities and change?

Answer:
Not generated.

Corpus profile: prince

Refused: False
Generation error: GEMINI_API_KEY is not set; Gemini generation is unavailable.
Citations: None

Retrieved evidence:
[prince-chapter-ii-chunk-0] chapter=II vector_score=0.597500 rerank_score=0.99736220
CONCERNING HEREDITARY PRINCIPALITIES I will leave out all discussion on republics, inasmuch as in another place I have written of them at length, and will address myself only to principalities. In doing so I will keep to the order indicated above, and discuss how such principalities are to be ruled and preserved. I say at once there are fewer difficulties in holding hereditary states, and those long accustomed to the family of their prince, than new ones; for it is sufficient only not to transgress the customs of his ancestors, and to deal prudently with circumstances as they arise, for a prince of average powers to maintain himself in his state, unless he be deprived of it by some extraordinary and excessive force; and if he should be so deprived of it, whenever anything sinister happens
[prince-chapter-i-chunk-0] chapter=I vector_score=0.532172 rerank_score=0.95175695
HOW MANY KINDS OF PRINCIPALITIES THERE ARE, AND BY WHAT MEANS THEY ARE ACQUIRED All states, all powers, that have held and hold rule over men have been and are either republics or principalities. Principalities are either hereditary, in which the family has been long established; or they are new. The new are either entirely new, as was Milan to Francesco Sforza, or they are, as it were, members annexed to the hereditary state of the prince who has acquired them, as was the kingdom of Naples to that of the King of Spain. Such dominions thus acquired are either accustomed to live under a prince, or to live in freedom; and are acquired either by the arms of the prince himself, or of others, or else by fortune or by ability.
[prince-chapter-iii-chunk-0] chapter=III vector_score=0.474683 rerank_score=0.38186765
CONCERNING MIXED PRINCIPALITIES But the difficulties occur in a new principality. And firstly, if it be not entirely new, but is, as it were, a member of a state which, taken collectively, may be called composite, the changes arise chiefly from an inherent difficulty which there is in all new principalities; for men change their rulers willingly, hoping to better themselves, and this hope induces them to take up arms against him who rules: wherein they are deceived, because they afterwards find by experience they have gone from bad to worse. This follows also on another natural and common necessity, which always causes a new prince to burden those who have submitted to him with his soldiery and with infinite other hardships which he must put upon his new acquisition. In this way you have

Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 20489.13it/s]
```

Ranking explanation for this query: top‑1 is Chapter II because it directly states that hereditary states are easier to hold and gives the ancestral-customs/prudence reason. Top‑2 is Chapter I because it defines hereditary versus new principalities, giving relevant background but not the requested discussion of change. Top‑3 is Chapter III because it discusses why people change rulers in new principalities; this is related to ‘change’ but less directly answers the hereditary-principalities question. The reranker score ordering agrees with this relevance distinction.
Relevance uses L2-normalized MiniLM embeddings in FAISS `IndexFlatIP` (cosine similarity), followed by FlashRank scores; the displayed final order is reranker order.

## Additional retriever comparison: multi-vector and parent retrieval

Both strategies use the same Chapter XV corpus, 500/80 child chunks, MiniLM candidates, and FlashRank ranking. Multi-vector retrieval returns the matching child passages. Parent retrieval groups those child hits and returns their 1,600-character containing window.
### multi-vector vs parent retrieval

Command: `'/Users/jovanagrujic/Downloads/rag-week-6-homework-starter/.venv/bin/python' 'scripts/compare_retrievers.py' '--offline'` (exit code 0)

```text
Query: What qualities are praised or blamed in men and princes?
Shared setup: Chapter XV; MiniLM cosine candidates; FlashRank reranking; same 500/80 child vectors.
Multi-vector retrieval returns the highest-ranked child passages; parent retrieval groups those hits and returns the containing 1,600-character parent window.

Multi-vector (child passages):
1. prince-chapter-xv-parent-0-chunk-0 parent=prince-chapter-xv-parent-0 vector=0.698723 rerank=0.999574; CONCERNING THINGS FOR WHICH MEN, AND ESPECIALLY PRINCES, ARE PRAISED OR BLAMED It remains now to see what ought to be the rules of conduct for a prince towards subject and friends. And as I know that many have 
2. prince-chapter-xv-parent-0-chunk-3 parent=prince-chapter-xv-parent-0 vector=0.584536 rerank=0.957182; are spoken of, and chiefly princes for being more highly placed, are remarkable for some of those qualities which bring them either blame or praise; and thus it is that one is reputed liberal, another miserly, 
3. prince-chapter-xv-parent-0-chunk-2 parent=prince-chapter-xv-parent-0 vector=0.608687 rerank=0.508531; his preservation; for a man who wishes to act entirely up to his professions of virtue soon meets with what destroys him among so much that is evil. Hence it is necessary for a prince wishing to hold his own to

Parent retrieval (expanded context):
1. prince-chapter-xv-parent-0 matched_child=prince-chapter-xv-parent-0-chunk-0 rerank=0.999574; parent_chars=1594; CONCERNING THINGS FOR WHICH MEN, AND ESPECIALLY PRINCES, ARE PRAISED OR BLAMED It remains now to see what ought to be the rules of conduct for a prince towards subject and friends. And as I know that many have 
2. prince-chapter-xv-parent-1 matched_child=prince-chapter-xv-parent-1-chunk-1 rerank=0.004343; parent_chars=1542; praise; and thus it is that one is reputed liberal, another miserly, using a Tuscan term (because an avaricious person in our language is still he who desires to possess by robbery, whilst we call one miserly w

Interpretation: multi-vector gives precise, compact evidence and can surface multiple passages from one chapter; parent retrieval restores surrounding context and may reduce boundary loss, but returns more text and can include irrelevant material. These are retrieval examples, not a labeled benchmark; assess answer coverage and context size together.

Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 20521.25it/s]
```

Interpretation: child-level multi-vector results are compact and precise, while parent retrieval adds surrounding context and can bridge a boundary at the cost of more irrelevant text. This is a qualitative single-query comparison, not a labeled benchmark; output length and answer coverage should be considered together.

## Outcome 3: Citation quality

Supported query: What does Chapter II say about hereditary principalities and change?

Generated answer captured from Gemini.

The actual captured CLI output follows; it includes the retrieved evidence and, when generation is available, the answer and its validated citations.

### supported query output

Command: `'/Users/jovanagrujic/Downloads/rag-week-6-homework-starter/.venv/bin/python' '-m' 'src.cli' '--corpus-profile' 'prince' '--question' 'What does Chapter II say about hereditary principalities and change?' '--top-k' '2'` (exit code 0)

```text
Question:
What does Chapter II say about hereditary principalities and change?

Answer:
Hereditary principalities that are long accustomed to the ruling family are easier to maintain than new states. [prince-chapter-ii-chunk-0] A prince of average ability can maintain his hereditary state by avoiding transgression of ancestral customs and responding prudently to events. [prince-chapter-ii-chunk-0] Only an extraordinary or excessive force can deprive a ruler of an hereditary principality. [prince-chapter-ii-chunk-0]

Corpus profile: prince

Refused: False
Generation error: None
Citations: prince-chapter-ii-chunk-0, prince-chapter-ii-chunk-0, prince-chapter-ii-chunk-0

Retrieved evidence:
[prince-chapter-ii-chunk-0] chapter=II vector_score=0.597500 rerank_score=0.99736220
CONCERNING HEREDITARY PRINCIPALITIES I will leave out all discussion on republics, inasmuch as in another place I have written of them at length, and will address myself only to principalities. In doing so I will keep to the order indicated above, and discuss how such principalities are to be ruled and preserved. I say at once there are fewer difficulties in holding hereditary states, and those long accustomed to the family of their prince, than new ones; for it is sufficient only not to transgress the customs of his ancestors, and to deal prudently with circumstances as they arise, for a prince of average powers to maintain himself in his state, unless he be deprived of it by some extraordinary and excessive force; and if he should be so deprived of it, whenever anything sinister happens
[prince-chapter-i-chunk-0] chapter=I vector_score=0.532172 rerank_score=0.95175695
HOW MANY KINDS OF PRINCIPALITIES THERE ARE, AND BY WHAT MEANS THEY ARE ACQUIRED All states, all powers, that have held and hold rule over men have been and are either republics or principalities. Principalities are either hereditary, in which the family has been long established; or they are new. The new are either entirely new, as was Milan to Francesco Sforza, or they are, as it were, members annexed to the hereditary state of the prince who has acquired them, as was the kingdom of Naples to that of the King of Spain. Such dominions thus acquired are either accustomed to live under a prince, or to live in freedom; and are acquired either by the arms of the prince himself, or of others, or else by fortune or by ability.

Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 18585.21it/s]
INFO:google_genai.models:AFC is enabled with max remote calls: 10.
INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent "HTTP/1.1 200 OK"
```

## Outcome 4: Refusal behavior

Unsupported query: What names of parents were given for Castruccio Castracani, and how was the child raised?

The fixed threshold was selected from the real scores in `data/refusal_calibration_scores.json` using the eight repository-held labeled examples.
Calibrated threshold=0.48179012; best retrieved rerank score=0.00014715, below threshold, so the query is refused.

### unsupported query output

Command: `'/Users/jovanagrujic/Downloads/rag-week-6-homework-starter/.venv/bin/python' '-m' 'src.cli' '--corpus-profile' 'prince' '--question' 'What names of parents were given for Castruccio Castracani, and how was the child raised?' '--top-k' '4'` (exit code 0)

```text
Question:
What names of parents were given for Castruccio Castracani, and how was the child raised?

Answer:
I do not have enough grounded evidence in the provided book text to answer that question.

Corpus profile: prince

Refused: True
Generation error: None
Citations: None

Retrieved evidence:
[prince-chapter-vi-chunk-9] chapter=VI vector_score=0.295473 rerank_score=0.00014715
is Hiero the Syracusan. This man rose from a private station to be Prince of Syracuse, nor did he, either, owe anything to fortune but opportunity; for the Syracusans, being oppressed, chose him for their captain, afterwards he was rewarded by being made their prince. He was of so great ability, even as a private citizen, that one who writes of him says he wanted nothing but a kingdom to be a king. This man abolished the old soldiery, organized the new, gave up old alliances, made new ones; and as he had his own soldiers and allies, on such foundations he was able to build any edifice: thus, whilst he had endured much trouble in acquiring, he had but little in keeping.
[prince-chapter-xix-chunk-6] chapter=XIX vector_score=0.319282 rerank_score=0.00011399
on account of it he has the people for an enemy, and thus cannot hope for any escape. Endless examples could be given on this subject, but I will be content with one, brought to pass within the memory of our fathers. Messer Annibale Bentivogli, who was prince in Bologna (grandfather of the present Annibale), having been murdered by the Canneschi, who had conspired against him, not one of his family survived but Messer Giovanni, who was in childhood: immediately after his assassination the people rose and murdered all the Canneschi. This sprung from the popular goodwill which the house of Bentivogli enjoyed in those days in Bologna; which was so great that, although none remained there after the death of Annibale who was able to rule the state, the Bolognese, having information that there
[prince-chapter-viii-chunk-5] chapter=VIII vector_score=0.393829 rerank_score=0.00008410
uncle, Giovanni Fogliani, and in the early days of his youth sent to fight under Pagolo Vitelli, that, being trained under his discipline, he might attain some high position in the military profession. After Pagolo died, he fought under his brother Vitellozzo, and in a very short time, being endowed with wit and a vigorous body and mind, he became the first man in his profession. But it appearing a paltry thing to serve under others, he resolved, with the aid of some citizens of Fermo, to whom the slavery of their country was dearer than its liberty, and with the help of the Vitelleschi, to seize Fermo. So he wrote to Giovanni Fogliani that, having been away from home for many years, he wished to visit him and his city, and in some measure to look upon his patrimony; and although he had
[prince-chapter-iii-chunk-23] chapter=III vector_score=0.269183 rerank_score=0.00007914
on these matters I spoke at Nantes with Rouen, when Valentino, as Cesare Borgia, the son of Pope Alexander, was usually called, occupied the Romagna, and on Cardinal Rouen observing to me that the Italians did not understand war, I replied to him that the French did not understand statecraft, meaning that otherwise they would not have allowed the Church to reach such greatness. And in fact it has been seen that the greatness of the Church and of Spain in Italy has been caused by France, and her ruin may be attributed to them. From this a general rule is drawn which never or rarely fails: that he who is the cause of another becoming powerful is ruined; because that predominancy has been brought about either by astuteness or else by force, and both are distrusted by him who has been raised

Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 21624.45it/s]
```

## Outcome 5: Final evidence pair

Supported command: `'/Users/jovanagrujic/Downloads/rag-week-6-homework-starter/.venv/bin/python' '-m' 'src.cli' '--corpus-profile' 'prince' '--question' 'What does Chapter II say about hereditary principalities and change?' '--top-k' '2'`

Supported output:

```text
Question:
What does Chapter II say about hereditary principalities and change?

Answer:
Hereditary principalities that are long accustomed to the ruling family are easier to maintain than new states. [prince-chapter-ii-chunk-0] A prince of average ability can maintain his hereditary state by avoiding transgression of ancestral customs and responding prudently to events. [prince-chapter-ii-chunk-0] Only an extraordinary or excessive force can deprive a ruler of an hereditary principality. [prince-chapter-ii-chunk-0]

Corpus profile: prince

Refused: False
Generation error: None
Citations: prince-chapter-ii-chunk-0, prince-chapter-ii-chunk-0, prince-chapter-ii-chunk-0

Retrieved evidence:
[prince-chapter-ii-chunk-0] chapter=II vector_score=0.597500 rerank_score=0.99736220
CONCERNING HEREDITARY PRINCIPALITIES I will leave out all discussion on republics, inasmuch as in another place I have written of them at length, and will address myself only to principalities. In doing so I will keep to the order indicated above, and discuss how such principalities are to be ruled and preserved. I say at once there are fewer difficulties in holding hereditary states, and those long accustomed to the family of their prince, than new ones; for it is sufficient only not to transgress the customs of his ancestors, and to deal prudently with circumstances as they arise, for a prince of average powers to maintain himself in his state, unless he be deprived of it by some extraordinary and excessive force; and if he should be so deprived of it, whenever anything sinister happens
[prince-chapter-i-chunk-0] chapter=I vector_score=0.532172 rerank_score=0.95175695
HOW MANY KINDS OF PRINCIPALITIES THERE ARE, AND BY WHAT MEANS THEY ARE ACQUIRED All states, all powers, that have held and hold rule over men have been and are either republics or principalities. Principalities are either hereditary, in which the family has been long established; or they are new. The new are either entirely new, as was Milan to Francesco Sforza, or they are, as it were, members annexed to the hereditary state of the prince who has acquired them, as was the kingdom of Naples to that of the King of Spain. Such dominions thus acquired are either accustomed to live under a prince, or to live in freedom; and are acquired either by the arms of the prince himself, or of others, or else by fortune or by ability.

Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 18585.21it/s]
INFO:google_genai.models:AFC is enabled with max remote calls: 10.
INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent "HTTP/1.1 200 OK"
```

Unsupported command: `'/Users/jovanagrujic/Downloads/rag-week-6-homework-starter/.venv/bin/python' '-m' 'src.cli' '--corpus-profile' 'prince' '--question' 'What names of parents were given for Castruccio Castracani, and how was the child raised?' '--top-k' '4'`

Unsupported output:

```text
Question:
What names of parents were given for Castruccio Castracani, and how was the child raised?

Answer:
I do not have enough grounded evidence in the provided book text to answer that question.

Corpus profile: prince

Refused: True
Generation error: None
Citations: None

Retrieved evidence:
[prince-chapter-vi-chunk-9] chapter=VI vector_score=0.295473 rerank_score=0.00014715
is Hiero the Syracusan. This man rose from a private station to be Prince of Syracuse, nor did he, either, owe anything to fortune but opportunity; for the Syracusans, being oppressed, chose him for their captain, afterwards he was rewarded by being made their prince. He was of so great ability, even as a private citizen, that one who writes of him says he wanted nothing but a kingdom to be a king. This man abolished the old soldiery, organized the new, gave up old alliances, made new ones; and as he had his own soldiers and allies, on such foundations he was able to build any edifice: thus, whilst he had endured much trouble in acquiring, he had but little in keeping.
[prince-chapter-xix-chunk-6] chapter=XIX vector_score=0.319282 rerank_score=0.00011399
on account of it he has the people for an enemy, and thus cannot hope for any escape. Endless examples could be given on this subject, but I will be content with one, brought to pass within the memory of our fathers. Messer Annibale Bentivogli, who was prince in Bologna (grandfather of the present Annibale), having been murdered by the Canneschi, who had conspired against him, not one of his family survived but Messer Giovanni, who was in childhood: immediately after his assassination the people rose and murdered all the Canneschi. This sprung from the popular goodwill which the house of Bentivogli enjoyed in those days in Bologna; which was so great that, although none remained there after the death of Annibale who was able to rule the state, the Bolognese, having information that there
[prince-chapter-viii-chunk-5] chapter=VIII vector_score=0.393829 rerank_score=0.00008410
uncle, Giovanni Fogliani, and in the early days of his youth sent to fight under Pagolo Vitelli, that, being trained under his discipline, he might attain some high position in the military profession. After Pagolo died, he fought under his brother Vitellozzo, and in a very short time, being endowed with wit and a vigorous body and mind, he became the first man in his profession. But it appearing a paltry thing to serve under others, he resolved, with the aid of some citizens of Fermo, to whom the slavery of their country was dearer than its liberty, and with the help of the Vitelleschi, to seize Fermo. So he wrote to Giovanni Fogliani that, having been away from home for many years, he wished to visit him and his city, and in some measure to look upon his patrimony; and although he had
[prince-chapter-iii-chunk-23] chapter=III vector_score=0.269183 rerank_score=0.00007914
on these matters I spoke at Nantes with Rouen, when Valentino, as Cesare Borgia, the son of Pope Alexander, was usually called, occupied the Romagna, and on Cardinal Rouen observing to me that the Italians did not understand war, I replied to him that the French did not understand statecraft, meaning that otherwise they would not have allowed the Church to reach such greatness. And in fact it has been seen that the greatness of the Church and of Spain in Italy has been caused by France, and her ruin may be attributed to them. From this a general rule is drawn which never or rarely fails: that he who is the cause of another becoming powerful is ruined; because that predominancy has been brought about either by astuteness or else by force, and both are distrusted by him who has been raised

Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 21624.45it/s]
```

## RAGAS evaluation (latest saved run)

Generated: `2026-10-12T17:00:38+00:00` UTC with RAGAS 0.4.3.
Scored cases: 4; not scored: 0.

- faithfulness: 1.0000
- answer_relevancy: 0.7718
- context_precision: 0.8750
- context_recall: 1.0000

Per-case answers, citations, retrieved context, and scores are in `evidence/ragas-metrics.json`.

## Test output snapshot

Command: `'/Users/jovanagrujic/Downloads/rag-week-6-homework-starter/.venv/bin/python' '-m' 'pytest' '-q'` (exit code 0)

```text
.................................................                        [100%]
49 passed in 2.54s
```

## Dodatni zapis live provera i zaštite od haluciniranja (2026-10-12)

- Korisnica prijavljuje da je prošlo 5 live testova. Pojedinačni upiti i izlazi nisu dostavljeni, pa se rezultat čuva kao korisnički prijavljen, bez izmišljanja detalja.
- Za nedovoljno potkrepljena pitanja pipeline odbija generisanje pre Gemini poziva na osnovu kalibrisanog praga. Generisani citati se proveravaju prema dobavljenim chunk ID-jevima. Interfejs skriva dokaze kada je odgovor odbijen ili prepoznat kao odgovor o nedovoljnim dokazima. Ovo smanjuje rizik od haluciniranja, ali nije garancija savršenog odgovora.
- RAGAS istorija: raniji izveštaj (2/4 ocenjena) Faithfulness 0.9167, Answer Relevancy 0.7631, Context Precision 0.7500, Context Recall 1.0000; poslednji izveštaj (4/4) 1.0000, 0.7718, 0.8750, 1.0000. Poslednji detaljni rezultati su u `evidence/ragas-metrics.json`.
- Posle dodavanja FlashRank rerankera, ilustrativni upit o naslednim kneževinama rangirao je Poglavlje II iznad I i III prema direktnoj relevantnosti (0.9974, 0.9518, 0.3819). Context Precision u istorijskim RAGAS rezultatima je 0.7500→0.8750. Broj ocenjenih slučajeva se razlikuje između ta dva izveštaja, zato se promena ne pripisuje samo rerankeru.
