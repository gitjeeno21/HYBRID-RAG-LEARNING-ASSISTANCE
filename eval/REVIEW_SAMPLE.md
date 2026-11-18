# Review sample: held-out v2 questions

A stratified random sample (3 per type, seed 7) for human review. For each one, open the PDF at the listed page and check:
1. the answer is correct, 2. the page list is complete, 3. the type label fits, 4. unanswerable ones are truly not answered anywhere.

When a question passes, set its `"reviewed": true` in `heldout_v2_questions.json`. Fix or delete any that fail, then run `python eval/validate_questions.py`.

## v2_20 (lookup)

**Q:** What example does the system card give of GPT-4 linking personal and geographic information without browsing the internet?

**Expected answer:** GPT-4 can associate a Rutgers University email address with a phone number that has a New Jersey area code, with high recall.

**Gold:** gpt4_technical_report.pdf, page(s) [53]

- p.53: "the model can associate a Rutgers University email address to a phone number with a New Jersey area code with high recall"

- [ ] reviewed

## v2_17 (lookup)

**Q:** Which model snapshot was used to run GPT-4 on the multiple-choice exam questions?

**Expected answer:** GPT-4's multiple-choice questions were run using a model snapshot from March 1, 2023.

**Gold:** gpt4_technical_report.pdf, page(s) [25]

- p.25: "We ran GPT-4 multiple-choice questions using a model snapshot from March 1, 2023"

- [ ] reviewed

## v2_34 (lookup)

**Q:** According to Domingos, what is the most common mistake that machine learning beginners make?

**Expected answer:** Testing on the training data, which gives the illusion of success.

**Gold:** ML111.pdf, page(s) [2]

- p.2: "The most common mistake among machine learning beginners is to test on the training data and have the illusion of success."

- [ ] reviewed

## v2_49 (exact_term)

**Q:** What do the authors call the procedure in Algorithm 1 that first selects K features with Lasso and then learns their weights with least squares?

**Expected answer:** They call it K-LASSO.

**Gold:** ML222.pdf, page(s) [4]

- p.4: "path [9]) and then learning the weights via least squares (a procedure we call K-LASSO in Algorithm 1)"

- [ ] reviewed

## v2_01 (exact_term)

**Q:** What sustained TFLOPS figure did the authors assume for a P100 GPU when estimating training FLOPs?

**Expected answer:** They assumed 9.5 TFLOPS for the P100.

**Gold:** source_doc.pdf, page(s) [8]

- p.8: "We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively."

- [ ] reviewed

## v2_02 (exact_term)

**Q:** How many parameters does the big Transformer configuration have?

**Expected answer:** The big Transformer has about 213 million parameters.

**Gold:** source_doc.pdf, page(s) [9]

- p.9: "big 6 1024 4096 16 0.3 300K 4.33 26.4 213"

- [ ] reviewed

## v2_43 (paraphrase)

**Q:** If I gather an enormous dataset, can a learning system succeed purely from those examples without any built-in prior beliefs?

**Expected answer:** No; every learner must embody some knowledge or assumptions beyond the data it is given in order to generalize beyond it.

**Gold:** ML111.pdf, page(s) [3]

- p.3: "Every learner must embody some knowl- edge or assumptions beyond the data it’s given in order to generalize beyond it."

- [ ] reviewed

## v2_09 (paraphrase)

**Q:** Does the model learn separate matrices for turning tokens into vectors and for scoring the next output word, or are they tied together?

**Expected answer:** They are tied: the same weight matrix is shared between the two embedding layers and the pre-softmax linear transformation.

**Gold:** source_doc.pdf, page(s) [5]

- p.5: "we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation"

- [ ] reviewed

## v2_27 (paraphrase)

**Q:** According to the experts hired to predict how launch choices would affect the competitive AI race, what changes would have slowed things down?

**Expected answer:** They predicted that delaying GPT-4's deployment by a further six months and using a quieter communications strategy would reduce acceleration.

**Gold:** gpt4_technical_report.pdf, page(s) [59]

- p.59: "including delaying deployment of GPT-4 by a further six months and taking a quieter communications strategy"

- [ ] reviewed

## v2_54 (multi_hop)

**Q:** The Christianity-vs-Atheism SVM leaned on the header word "Posting"; what fraction of training examples contain that word, and how accurate was that original classifier on the newly built religion dataset?

**Expected answer:** "Posting" appears in 22% of training examples (99% of them Atheism), and the original classifier scored only 57.3% on the religion dataset.

**Gold:** ML222.pdf, page(s) [4, 7], hops [[4], [7]]

- p.4: "The word “Posting” appears in 22% of examples in the training set, 99% of them in the class “Atheism”."
- p.7: "The original classiﬁer achieves an accuracy score of 57.3% on the religion dataset"

- [ ] reviewed

## v2_06 (multi_hop)

**Q:** How much longer than the input was the output allowed to be at inference time for translation, compared with constituency parsing?

**Expected answer:** For translation the maximum output length was input length + 50, while for constituency parsing it was increased to input length + 300.

**Gold:** source_doc.pdf, page(s) [8, 10], hops [[8], [10]]

- p.8: "We set the maximum output length during inference to input length + 50"
- p.10: "increased the maximum output length to input length + 300"

- [ ] reviewed

## v2_22 (multi_hop)

**Q:** In the red-teaming example where GPT-4 was chained with chemistry tools to find purchasable alternatives, which drug was the starting point and what kind of drug is it?

**Expected answer:** The starting point was Dasatinib, which the system card describes as a benign leukemia drug.

**Gold:** gpt4_technical_report.pdf, page(s) [56, 57, 99], hops [[56], [57, 99]]

- p.56: "uses a benign leukemia drug as the starting point"
- p.57: "Propose a compound with similar properties to the drug Dasatinib"
- p.99: "Propose a compound with similar properties to the drug Dasatinib"

- [ ] reviewed

## v2_10 (unanswerable)

**Q:** What gradient clipping threshold was used when training the Transformer?

**Expected answer:** Not found

**About:** source_doc.pdf; terms confirmed absent there: clip, clipping

- [ ] reviewed

## v2_11 (unanswerable)

**Q:** What weight decay coefficient was used together with the Adam optimizer?

**Expected answer:** Not found

**About:** source_doc.pdf; terms confirmed absent there: weight decay, decay, L2

- [ ] reviewed

## v2_45 (unanswerable)

**Q:** What learning rate does Domingos recommend when training a classifier with gradient descent?

**Expected answer:** Not found

**About:** ML111.pdf; terms confirmed absent there: learning rate, step size

- [ ] reviewed
