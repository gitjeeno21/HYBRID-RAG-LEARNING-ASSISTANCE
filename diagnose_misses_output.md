# Retrieval miss diagnostic: q35, q38, q39, q40

Read-only retrieval using the existing 1500/50 index (281 chunks), local embeddings, BM25 over those same chunks, RRF k=60, 10 candidates per method, and top 3. No questions or retrieval settings were changed; no hosted API was called.

## q35

**Question:** How does the decoder use information from the input sequence while producing target representations?
**Expected source/page:** `source_doc.pdf`, p. 5
**Expected keywords:** queries come from the previous decoder layer, keys and values come from the output of the encoder

**Dense top 3**
1. `source_doc.pdf`, p. 5 — keywords: none — `mation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]. In the embeddi`
2. `source_doc.pdf`, p. 5 — keywords: none — `encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder. • Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including`
3. `source_doc.pdf`, p. 2 — keywords: none — `1 Introduction Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language m`

**Hybrid top 3**
1. `source_doc.pdf`, p. 5 — keywords: none — `encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder. • Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including`
2. `source_doc.pdf`, p. 2 — keywords: none — `used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22]. End-to-end memory networks are based on a recurrent attenti`
3. `source_doc.pdf`, p. 2 — keywords: none — `1 Introduction Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language m`

- **Dense:** first expected source/page chunk rank 1; KEYWORD ISSUE — The expected source and page are in the top three, but none of the expected keywords appears there.
- **BM25:** first expected source/page chunk rank 2; HIT — The top three include the expected page and at least one expected keyword.
- **Hybrid:** first expected source/page chunk rank 1; KEYWORD ISSUE — The expected source and page are in the top three, but none of the expected keywords appears there.

## q38

**Question:** What caveat should readers keep in mind when interpreting some percentile results in the exam comparison?
**Expected source/page:** `gpt4_technical_report.pdf`, p. 6
**Expected keywords:** lower end of the range, wide scoring bins

**Dense top 3**
1. `gpt4_technical_report.pdf`, p. 30 — keywords: none — `Leetcode (hard) 0 % 3 / 45 3 / 45 3 / 45 3 / 45 Table 9. Contamination data for Exams (Summary). For each of the exams tested, we show the fraction of questions in the exam which are contaminated (i.e. present in the training dataset). We show the fi`
2. `gpt4_technical_report.pdf`, p. 25 — keywords: none — `republished on an SAT prep site [ 74]; for the GRE, we converted multiple-choice scores to the 130-170 scale using the official formula of multiplying accuracy by 40 and adding 130; for the AP exams, we used the score calculators found on a public st`
3. `gpt4_technical_report.pdf`, p. 4 — keywords: none — `Overall scores were determined by combining multiple-choice and free-response question scores using publicly available methodologies for each exam. We estimate and report the percentile each overall score corresponds to. See Appendix A for further de`

**Hybrid top 3**
1. `gpt4_technical_report.pdf`, p. 30 — keywords: none — `Leetcode (hard) 0 % 3 / 45 3 / 45 3 / 45 3 / 45 Table 9. Contamination data for Exams (Summary). For each of the exams tested, we show the fraction of questions in the exam which are contaminated (i.e. present in the training dataset). We show the fi`
2. `gpt4_technical_report.pdf`, p. 6 — keywords: lower end of the range, wide scoring bins — `AP Calculus BCAMC 12Codeforces RatingAP English LiteratureAMC 10Uniform Bar ExamAP English LanguageAP ChemistryGRE QuantitativeAP Physics 2USABO Semifinal 2020AP MacroeconomicsAP StatisticsLSATGRE WritingAP MicroeconomicsAP BiologyGRE VerbalAP World `
3. `gpt4_technical_report.pdf`, p. 25 — keywords: none — `republished on an SAT prep site [ 74]; for the GRE, we converted multiple-choice scores to the 130-170 scale using the official formula of multiplying accuracy by 40 and adding 130; for the AP exams, we used the score calculators found on a public st`

- **Dense:** first expected source/page chunk rank 4; RANKING FAILURE — The expected source and page rank 4 in the top 10 but fall below the top three.
- **BM25:** first expected source/page chunk rank 1; HIT — The top three include the expected page and at least one expected keyword.
- **Hybrid:** first expected source/page chunk rank 2; HIT — The top three include the expected page and at least one expected keyword.

## q39

**Question:** How did the authors limit look-ahead when forecasting GPT-4's coding performance?
**Expected source/page:** `gpt4_technical_report.pdf`, p. 4
**Expected keywords:** registered predictions, prior to training

**Dense top 3**
1. `gpt4_technical_report.pdf`, p. 10 — keywords: none — `5 Limitations Despite its capabilities, GPT-4 has similar limitations as earlier GPT models. Most importantly, it still is not fully reliable (it “hallucinates” facts and makes reasoning errors). Great care should be taken when using language model o`
2. `gpt4_technical_report.pdf`, p. 3 — keywords: none — `Observed Prediction gpt-4 100p 10n 1µ 100µ 0.01 1 Compute 1.0 2.0 3.0 4.0 5.0 6.0 Bits per word OpenAI codebase next word prediction Figure 1. Performance of GPT-4 and smaller models. The metric is final loss on a dataset derived from our internal co`
3. `gpt4_technical_report.pdf`, p. 1 — keywords: none — `that were tested against the final run to increase confidence in our training. Despite its capabilities, GPT-4 has similar limitations to earlier GPT models [1, 37, 38]: it is not fully reliable (e.g. can suffer from “hallucinations”), has a limited `

**Hybrid top 3**
1. `gpt4_technical_report.pdf`, p. 3 — keywords: none — `Observed Prediction gpt-4 100p 10n 1µ 100µ 0.01 1 Compute 1.0 2.0 3.0 4.0 5.0 6.0 Bits per word OpenAI codebase next word prediction Figure 1. Performance of GPT-4 and smaller models. The metric is final loss on a dataset derived from our internal co`
2. `gpt4_technical_report.pdf`, p. 10 — keywords: none — `5 Limitations Despite its capabilities, GPT-4 has similar limitations as earlier GPT models. Most importantly, it still is not fully reliable (it “hallucinates” facts and makes reasoning errors). Great care should be taken when using language model o`
3. `gpt4_technical_report.pdf`, p. 37 — keywords: none — `GPT-4 visual input example, Pixel to Paper Summaries: User Below is part of the InstuctGPT paper. Could you read and summarize it to me? GPT-4 The InstructGPT paper focuses on training large language models to follow instructions with human feedback.`

- **Dense:** first expected source/page chunk rank 10; RANKING FAILURE — The expected source and page rank 10 in the top 10 but fall below the top three.
- **BM25:** first expected source/page chunk rank not in top 10; RETRIEVAL FAILURE — No chunk from the expected source and page appears in the top 10.
- **Hybrid:** first expected source/page chunk rank not in top 10; RETRIEVAL FAILURE — No chunk from the expected source and page appears in the top 10.

## q40

**Question:** On the Uniform Bar Examination comparison, how did GPT-4's score without image input compare to its regular score?
**Expected source/page:** `gpt4_technical_report.pdf`, p. 5
**Expected keywords:** 298 / 400, ~90th

**Dense top 3**
1. `gpt4_technical_report.pdf`, p. 30 — keywords: 298 / 400, ~90th — `Exam Contam GPT-4 (no vision) Non-contaminated GPT-4 (no vision) GPT-4 Non-contaminated GPT-4 Uniform Bar Exam (MBE+MEE+MPT) 0 % 298 / 400 (~90th) 298 / 400 (~90th) 298 / 400 (~90th) 298 / 400 (~90th) LSAT 39 % 161 (~83rd) 167 (~95th) 163 (~88th) 169`
2. `gpt4_technical_report.pdf`, p. 8 — keywords: none — `as it does on text-only inputs. An example of GPT-4’s visual input can be found in Table 3. The standard test-time techniques developed for language models (e.g. few-shot prompting, chain-of- thought, etc) are similarly effective when using both imag`
3. `gpt4_technical_report.pdf`, p. 5 — keywords: 298 / 400, ~90th — `Exam GPT-4 GPT-4 (no vision) GPT-3.5 Uniform Bar Exam (MBE+MEE+MPT) 298 / 400 (~90th) 298 / 400 (~90th) 213 / 400 (~10th) LSAT 163 (~88th) 161 (~83rd) 149 (~40th) SAT Evidence-Based Reading & Writing 710 / 800 (~93rd) 710 / 800 (~93rd) 670 / 800 (~87`

**Hybrid top 3**
1. `gpt4_technical_report.pdf`, p. 6 — keywords: none — `AP Calculus BCAMC 12Codeforces RatingAP English LiteratureAMC 10Uniform Bar ExamAP English LanguageAP ChemistryGRE QuantitativeAP Physics 2USABO Semifinal 2020AP MacroeconomicsAP StatisticsLSATGRE WritingAP MicroeconomicsAP BiologyGRE VerbalAP World `
2. `gpt4_technical_report.pdf`, p. 25 — keywords: none — `A.4 Images Oftentimes, an exam question may include an image. Models like GPT-3.5, which consume text (but not images) as input might not have access to all the information needed to correctly solve a problem. When evaluating text models on multiple-`
3. `gpt4_technical_report.pdf`, p. 30 — keywords: none — `Leetcode (hard) 0 % 3 / 45 3 / 45 3 / 45 3 / 45 Table 9. Contamination data for Exams (Summary). For each of the exams tested, we show the fraction of questions in the exam which are contaminated (i.e. present in the training dataset). We show the fi`

- **Dense:** first expected source/page chunk rank 3; HIT — The top three include the expected page and at least one expected keyword.
- **BM25:** first expected source/page chunk rank not in top 10; RETRIEVAL FAILURE — No chunk from the expected source and page appears in the top 10.
- **Hybrid:** first expected source/page chunk rank 7; RANKING FAILURE — The expected source and page rank 7 in the top 10 but fall below the top three.

## Summary table

| Question | Dense label | Hybrid label | Right chunk rank (dense / BM25 / hybrid) |
|---|---|---|---|
| q35 | KEYWORD ISSUE | KEYWORD ISSUE | 1 / 2 / 1 |
| q38 | RANKING FAILURE | HIT | 4 / 1 / 2 |
| q39 | RANKING FAILURE | RETRIEVAL FAILURE | 10 / not in top 10 / not in top 10 |
| q40 | HIT | RANKING FAILURE | 3 / not in top 10 / 7 |

## Interpretation

Across the dense and hybrid method-level misses, 3 are ranking failures, so a re-ranker could help in those cases.
1 is a retrieval failure, where the expected source/page chunk is absent from the top 10, pointing to chunking or embedding retrieval rather than ranking.
2 are keyword issues, where the expected page is present in the top three but the evaluator's expected keywords are absent.

## scratch_step4.py: first 10 lines

The scratch script was not present.
