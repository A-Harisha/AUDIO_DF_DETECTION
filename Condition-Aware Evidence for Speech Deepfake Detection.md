## Condition-Aware Evidence for Speech Deepfake Detection

## Research Proposal

O ct 5, 2026 · @Harisha

## 1. The problems we solve

A speech deepfake detector reports a score. The score is reliable on clean test data. It is not reliable on real audio. We attack three causes of this.

## P1. The detector fails outside the laboratory.

- AASIST reaches 1.89% EER on its own test set. It reaches 41.85% EER on the ML-ITW benchmark. The best model there still reaches 27.65%.

- Open-source detectors lose about 48% of their AUC on real audio.

- Noise alone moves EER from 0.83% to 16.24% at 10 dB SNR.

## P2. The system reports the wrong confidence.

- Almost all papers report EER. EER uses the best threshold for that test set. You find this threshold after you see the labels. You cannot do this in deployment.

- ASVspoof 5 found poor calibration in its top systems. These systems separate real from fake well. Their scores are not reliable as probabilities.

- A court needs a stated evidential strength. It does not need a binary verdict.

## P3. The explanation is not trustworthy.

- A score gives no reason. An analyst cannot act on a score.

- Attribution maps are fragile. An attacker can distort the map while the predicted label stays the same.

- Detectors learn shortcuts. Leading-silence duration alone predicts the label at up to 85% accuracy.

## Two conditions make all three worse.

- Language. Most training data is English or Chinese. Indian and other low-resource languages are thin.

- Channel. Codecs, compression and noise reach every real recording.


## 2. Prior art and the gap

The obvious ideas are taken. You must position against this work.

| Work | What it does | Why it is not enough |
| --- | --- | --- |
| Negroni et al., artifact- | Each expert finds one | It calibrates once, globally, on |
| specific experts (2026) | artifact. The system | clean data. No language or |
|   | calibrates each output into a | channel conditioning. |
|   | log-likelihood ratio. It reports |   |
|   | Cllr with EER. |   |
| From Scores to Evidence | Auditable decision records | Same limit. The calibration is |
| (Sept 2026) | with EER, minDCF and | fixed. |
|   | expected calibration error. |   |
| Meta-learned LoRA, | Simulates domain shift in | It seeks one invariant model. |
| MLDG (ICASSP 2026) | training. Average EER 8.84% | It does not report when it is |
|   | to 5.30%. | out of its depth. |
| MoLEx, MoE-LoRA, | Mixture of experts routes by | Routing serves accuracy only. |
| DADGMoE (2025 to 2026) | attack type or learned | The gate carries no evidential |
|   | domain prototype. | meaning. |
| IB-CAAN (2026) | Information bottleneck | Suppresses the condition. We |
|   | removes nuisance variability. | argue you should measure it |
|   |   | instead. |
| Evidential and Dirichlet | Give an uncertainty value. | The uncertainty is not tied to |
| methods |   | a stated error rate or to a |
|   |   | validated range. |

The common flaw. Every system calibrates once, on clean data. Three findings show why this fails.

- Codec transport removes a large part of attribution performance. Losses reach 53.5 and 70.3 Macro-F1 points. The damage runs from -0.4 to +53.5 points inside one grid, so one number cannot summarise it.

- Benign edits, such as speech restoration, get scored as spoofing. The authors conclude that binary systems model the distribution of raw speech rather than authenticity itself.

- Speech enhancement gives a paradox. SEGAN had worse perceptual quality than MetricGAN+, but gave better detection. Enhancement appears to remove useful cues with the noise.


The gap in one line. An LLR calibrated on clean English overstates the evidence on 8 kHz Telugu. This is a calibration failure that looks like a detection failure. Nobody has separated the two.

## 3. The core claim

Detectors fail in the field because their calibration is wrong, not because their discrimination is wrong. We measure the condition of each recording and scale the evidence to that condition.

Three parts follow from this claim.

- 1. Estimate the condition. Find the codec, the bandwidth, the noise level and the language.

- 2. Calibrate for that condition. The evidence weakens as the condition moves away from the validated range.

- 3. Refuse outside the range. Report insufficient evidence instead of a number.

The explanation comes from inside the model, not from a post-hoc attribution map.

## 4. The proposed architecture

One model. No ensemble. The condition vector is the only thing that is shared, and it does two jobs: it tells the evidence block where its cues are weak, and it tells the calibration block how much the evidence is worth.


## The condi on vector steers both the evidence and the calibration

architecture · 4 blocks, one shared condition vector

Block A. Condition encoder. A small head on the backbone predicts the channel and the language. Your augmentation grid supplies the labels, so they cost nothing. The output is a vector, not a class, so unseen conditions land between known ones.

Block B. Evidence from a sparse autoencoder. A sparse autoencoder on the SSL features recovers separable directions. You name the directions that carry the fake evidence. The explanation then comes from inside the model. This matters because post-hoc attribution maps can be distorted while the predicted label stays the same.

Block C. Calibration hypernetwork. This is the core. A small network reads the condition vector and generates the calibration parameters. There is no single fixed calibration map. For the same raw score, the LLR shrinks toward zero as the condition leaves the validated region.

Block D. Competence gate. Conformal prediction over the validated condition region gives a guaranteed error rate. Outside that region the system outputs insufficient evidence. This matches the ENFSI framework: the expert reports evidential strength, and the court decides.

Three output classes, not two. Bona fide, processed bona fide, and synthetic. The third class stops the system from flagging denoised or restored real speech as fake.


## 5. Technology choices

| Part | Choice | Reason |
| --- | --- | --- |
| Backbone | XLS-R or WavLM Large, frozen | wav2vec2 and WavLM front ends |
|   |   | dominated ASVspoof 5. A |
|   |   | multilingual front end helps cross- |
|   |   | lingual transfer. |
| Adaptation | LoRA adapters | A LoRA model updated about |
|   |   | 3.6M parameters, roughly 1.1% of |
|   |   | full fine-tuning, and still beat it on |
|   |   | five of six corpora. |
| Condition | Small multi-task head | Labels come free from your |
| encoder |   | augmentation grid. |
| Conditioning | FiLM, or a hypernetwork | FiLM scales and shifts the |
| method |   | features. The hypernetwork |
|   |   | generates the calibration |
|   |   | parameters. The two act at |
|   |   | different places. |
| Evidence | Sparse autoencoder on SSL | Gives directions you can name. |
|   | features | The explanation sits inside the |
|   |   | model. |
| Calibration | Logistic calibration, parameters | Standard in forensic speaker |
|   | generated per condition | recognition. The novelty is that |
|   |   | the parameters vary. |
| Abstention | Mondrian conformal prediction | Gives a guaranteed error rate per |
|   |   | group. Groups are language and |
|   |   | codec. |
| Augmentation | Domain randomization over the | Wide random sampling beats |
|   | condition grid | matching a fixed test set. |
|   |   | RawBoost cut noisy EER from |
|   |   | 16.24% to 2.55%. |

## What we do not use.

- No ensemble and no mixture of experts. MoE for this task is already crowded, and the gate carries no evidential meaning.

- No information bottleneck to erase the condition. We measure the condition instead of removing it.


## 6. How we gain generalization, robustness and accuracy

These three are not the same goal. The design serves them differently. State this plainly in the paper.

## Cross-dataset generalization

Five mechanisms. None is novel alone. Together they keep us competitive on EER.

| Mechanism | What it does | Evidence |
| --- | --- | --- |
| Domain randomization | Trains on wide random | RawBoost returned noisy EER |
| over the condition grid | codecs, SNR and | from 16.24% to 2.55%. Pooled |
|   | reverberation, not one fixed | clean and noisy training |
|   | set | reached 1.53%. |
| FiLM conditioning on the | One model behaves like | Opposite choice to IB-CAAN, |
| condition vector | many specialized ones, | which removes nuisance |
|   | instead of averaging over all | variability. We argue the |
|   | channels and suiting none | channel is informative. |
| Multilingual frozen front | Transfers across languages | Meta-learned LoRA cut |
| end with LoRA | at low cost | average EER from 8.84% to |
|   |   | 5.30%, updating about 1.1% of |
|   |   | parameters. |
| Three output classes | Processed real speech gets | Binary systems conflate |
|   | its own class, removing a | benign transformations with |
|   | family of false alarms | spoofing. |
| Shortcut removal in | Trimmed silence and | Silence duration alone |
| training | controlled duration force the | predicts the label at up to |
|   | model onto real cues | 85% accuracy. |

## Robustness

- Train on the degradation. Do not repair it at test time. Enhancement is a condition, not a fix. SEGAN gave worse perceptual quality than MetricGAN+ but better detection, so repair removes useful cues with the noise.

- Report per condition, never pooled. Inside one published grid, losses ran from -0.4 to +53.5 points. A pooled number hides this entirely.


## Accuracy: why calibration is an accuracy result

A reviewer will say that better calibration without better accuracy is meaningless. Answer the objection with a number, not a principle.

EER and AUC are threshold-free. They ask how well the scores separate. Deployment is not threshold-free. You must fix the threshold before you see the data. At that fixed threshold, calibration decides how many errors you make.

A worked example. These numbers are illustrative, not measured. Set the threshold on clean English at a 1% false alarm rate.

| Test condition | Discrimination | Actual false alarm rate at that |
| --- | --- | --- |
|   | (EER) | fixed threshold |
| Clean English | 5% | 1%, as designed |
| 8 kHz Telugu, global calibration | 5% | 18%, because the score |
|   |   | distribution shifted |
| 8 kHz Telugu, condition- | 5% | about 1%, restored |
| dependent calibration |   |   |

The EER is the same in all three rows. The number of wrong decisions is not. Row two blocks 18 real callers in every 100. Row three blocks one. That difference is accuracy, and a deployment feels it.

State the claim as accuracy. Do not claim better calibration. Claim this:

At a fixed operating point, the system makes fewer errors under channel and language shift.

Report it with accuracy metrics: the miss rate at a 1% false alarm rate, and the false alarm rate at the threshold set on development data. Cllr and actDCF then support the claim. They do not carry it.

## Two further accuracy gains.

- Accuracy on retained data. The competence gate removes cases the model cannot judge. Report accuracy at a stated coverage, with the abstention rate beside it.

- Raw EER. The five mechanisms above carry this. Report their gains separately, so the paper holds two results: a better EER, and a far lower error rate at a usable threshold.


## The positioning risk

If you promise lower cross-dataset EER alone, you compete with DADGMoE, which reports up to 40.8% relative EER reduction, and with meta-learned LoRA. That race is crowded and you may lose it.

If you promise fewer errors at a fixed operating point under shift, you compete with almost nobody, and the result is still an accuracy result.

Build the five mechanisms to stay competitive on EER. Make the fixed-threshold error rate the headline.

## 7. The novelty achieved

Ranked by strength. Claim the top two. Treat the rest as support.

| Rank Claim |   | Why it holds |
| --- | --- | --- |
| 1 | Condition-dependent calibration of | Every prior system calibrates once, |
|   | forensic LLRs, generated by a | globally, on clean data. I found no audio |
|   | hypernetwork | work that makes the calibration a |
|   |   | function of the estimated channel and |
|   |   | language. |
| 2 | A validated competence envelope with a | Evidential and Dirichlet methods give |
|   | stated error rate | uncertainty, but not a guarantee tied to |
|   |   | a validated region. Conformal prediction |
|   |   | for speech anti-spoofing appears |
|   |   | unused. |
| 3 | Faithful evidence from a sparse | Faithfulness metrics exist. Mechanistic |
|   | autoencoder, tested against localization | interpretability is barely used in audio |
|   | ground truth | forensics. |
| 4 | Three-class handling of benign | The benign-transformation problem was |
|   | transformations inside an evidence | named in 2026. No evidence framework |
|   | framework | has absorbed it. |
| 5 | Per-language calibration loss as a | Nobody reports this today. Weak alone, |
|   | reported quantity | strong as support for claim 1. |

The sentence to defend. Calibration, not discrimination, is what breaks in the wild. Measured as errors at a fixed operating point, this is an accuracy claim, not only a calibration claim. Section 6 gives the worked example.


Verify before you submit. Check claim 2 against the literature yourself. I searched and found no conformal prediction applied to speech anti-spoofing, but absence of evidence is not proof. Several cited papers are only weeks old.

## 8. Target datasets

Train on a few. Test on many. Never test on what you trained on.

## Training and calibration

| Dataset | Content | Role |
| --- | --- | --- |
| ASVspoof | ~2,000 speakers, 16 unseen attacks, 7 | Main training set. Its codec grid |
| 5 | adversarial, codec conditions | matches our purpose. |
| MLAAD | 687.4 hours, 140 TTS systems, 51 | Multilingual training and generator |
|   | languages | diversity. |
| IndicSynth ~4,000 hours, 989 speakers, 12 Indian |   | Indian-language coverage. Public on |
|   | languages | Hugging Face. |

## Held-out evaluation

| Dataset | Content | What it tests |
| --- | --- | --- |
| ML-ITW | 14 languages, 7 platforms, 180 | The real-world collapse. The |
|   | public figures, 28.39 hours | headline test. |
| In-the-Wild | Celebrity and politician clips | The original in-the-wild |
|   |   | benchmark. |
| Fake Speech Wild | 254 hours from four platforms | Social-media conditions. |
| SEA-Spoof | 700+ hours, six languages | Cross-lingual, including Indian |
|   | including Tamil and Hindi | languages. |
| Deepfake-Eval-2024 | 56.5 hours of audio flagged by | Security-relevant audio, not |
|   | real users | synthetic benchmarks. |
| Indic-CodecFake | Codec-synthesized speech in | A newer attack family in your |
|   | Indic languages | target languages. |
| Low-resource | 66 languages | Breadth of language coverage. |
| spoofing corpus |   |   |


## Your own condition grid

Build this with ffmpeg and open noise corpora. It gives free labels for the condition encoder.

- Codecs. MP3, Opus, AMR, Speex 8 kHz, AAC, EnCodec, and MP3 followed by EnCodec.

- Noise. Additive noise at several SNR values, down to 0 dB.

- Room. Reverberation from public impulse responses.

- Benign edits. Denoising and speech restoration, applied to real speech only. These become the third class.

## Availability note

IndicFake reports 4.2 million samples and 7,350 hours across English and 17 Indian languages, but I could not confirm a public download. Check the authors' lab page before you plan around it.

## 9. Benchmarks and the numbers to beat

## Reference numbers to beat

These are published figures from the literature. Use them to set your targets and to pick your baselines. You compare against them on your own protocol. You do not enter any challenge.

| Source | Reported result | What it tells us |
| --- | --- | --- |
| AASIST on ML-ITW | 1.89% EER in domain, 41.85% | The size of the collapse. Our |
|   | on ML-ITW; best model 27.65% | headline contrast. |
| Deepfake-Eval-2024 | Open-source detectors lose | Confirms the collapse on user- |
|   | about 48% of audio AUC | flagged audio. |
| MLDG-LoRA | Average EER 8.84% to 5.30% | The generalization baseline to |
|   | across six corpora, updating | match. |
|   | about 1.1% of parameters |   |
| MoE-LoRA | Average out-of-domain EER | The mixture-of-experts |
|   | 8.55% to 6.08% | baseline. |
| DADGMoE, | Up to 40.8% relative EER | The newest routing baseline. |
| Interspeech 2026 | reduction out of dataset |   |
| Fake Speech Wild | 3.54% average EER after | Shows how much matched data |
|   | adding in-domain training data | alone buys. |


| Source | Reported result | What it tells us |
| --- | --- | --- |
| Noise and RawBoost | 0.83% to 16.24% EER at 10 dB | Augmentation already fixes |
|   | SNR; RawBoost returns it to | much of the noise gap. Beat this, |
|   | 2.55%, pooled training to | or your noise claim is weak. |
|   | 1.53% |   |
| Codec attribution | Losses of 53.5 and 70.3 Macro- | Our strongest supporting |
| study | F1 points after single-stage | evidence. |
|   | transport |   |
| Vicomtech, ASVspoof | 5.02% EER and 0.1348 | A worked example of condition- |
| 5 | minDCF; 1.17% clean, severely | dependent failure. |
|   | degraded by MP3 with |   |
|   | EnCodec and Speex 8 kHz |   |
| Phoneme-level | Pooled 8.52% EER on | Shows the per-attack spread |
| explainable system | ASVspoof 5; attacks A24 and | that pooling hides. |
|   | A28 above 20% |   |

## Threshold-free state of the art

This is the crowded race. These are the numbers a reviewer will quote at you.

| Benchmark | Best reported | System |
| --- | --- | --- |
| ASVspoof 2021 LA | 0.87% EER | Wav2DF-TSL |
| ASVspoof 2021 DF | 1.49 to 1.95% EER | Nes2NetX, Wav2DF-TSL |
| In-the-Wild | 6.83% EER, claimed as SOTA by its | Wav2DF-TSL |
|   | authors |   |
| In-the-Wild, | 1.26% EER | Whispeak, on the Speech DF |
| leaderboard |   | Arena |
| Pooled across the | 3.05% average EER, 3% pooled | Whispeak |
| Arena |   |   |

The last two rows disagree with the third. Different protocols give 1.26% and 6.83% on the same dataset. Cite this disagreement in the paper. It supports our argument that a single headline number says little.

## Calibrated state of the art

This is our setting. ASVspoof 5 is the only benchmark that reports actDCF and Cllr widely.


| System | minDCF actDCF Cllr |   | EER |
| --- | --- | --- | --- |
| Best fused system, open condition 0.0937 | 0.1375 | 0.1927 | 3.42% |
| TFPARN, closed condition 0.2430 | 0.2897 | 0.9243 12.52% |   |
| AASIST baseline, closed — | 0.4966 | 2.6545 — |   |
| RawNet2 baseline, closed — | 0.7214 | 2.8672 — |   |
| BUT, closed evaluation set 0.5809 | 0.8537 | 4.0994 23.34% |   |

## How to read this table.

- Cllr above 1 means poor calibration. Many systems sit far above it.

- actDCF near 1.0 is the worst case. One study found actDCF close to or equal to 1 for most attacks.

- The organizers noted that T45 had the lowest minDCF but much worse Cllr and actDCF, while T27 performed well on both. Discrimination and calibration come apart.

- Under codecs, AASIST reached minDCF between 0.627 and 0.711 on conditions C04, C07, C08, C09 and C10.

The gap is our target. Subtract minDCF from actDCF. The BUT system loses 0.27. The best open system loses 0.044. That gap is pure calibration loss. Nobody optimizes it under condition shift.

## No state of the art exists for our exact setting

I found no system that reports actDCF or Cllr per language, or per codec, under cross- dataset shift.

This cuts both ways.

- The opportunity. You set the first numbers.

- The burden. You must build every baseline yourself. Budget for it in phase 1.

## If block B uses localization

| Source | Reported result |
| --- | --- |
| BISL on PartialSpoof | 2.52% EER, 97.40% F1 |
| BISL on Half-Truth | 0.07% EER, 99.97% F1 |
| BISL on LlamaPartialSpoof, cross-domain 37.10% EER, 53.61% F1 |   |


The third row is the point. In-domain localization is near saturated. Cross-domain is not.

## A warning on these figures

The numbers come from different protocols and are not directly comparable. Re-run every baseline under your own protocol before you claim an improvement.

## 10. Test protocols and metrics

## The metrics

Report EER, because reviewers expect it. The claim rests on the fixed-threshold rows, which are accuracy metrics. Lead with those.

| Metric | What it shows | Why we need it |
| --- | --- | --- |
| EER | Discrimination at the best | The baseline comparison. Not |
|   | threshold | the claim. |
| minDCF | Cost-weighted error at the best | ASVspoof 5 made this its |
|   | threshold | primary metric. |
| actDCF | Cost at the threshold you | The gap from minDCF is the |
|   | actually set | calibration loss. This is the |
|   |   | claim. |
| Cllr | Quality of the LLRs as evidence Standard in forensic reporting. |   |
|   |   | Above 1 means poor |
|   |   | calibration. |
| Rates of misleading | How often strong evidence | Required for forensic |
| evidence | points the wrong way | validation. |
| Miss rate at 1% and | Performance at a usable | What a deployment actually |
| 0.1% false alarm | operating point | cares about. |
| Conformal coverage | Does the guarantee hold, and | Tests block D. |
| and set size | how often does it abstain |   |

## The five protocols

- 1. Cross-generator. Train on one set of generators. Test on unseen ones. Report per- generator results.

- 2. Cross-corpus. Train on ASVspoof 5 and MLAAD. Test on ML-ITW, In-the-Wild and Deepfake-Eval-2024 with no retraining.


- 3. Cross-lingual. Hold out whole languages. Report calibration loss per language, not just EER.

- 4. Codec stress. Apply the condition grid to a held-out corpus. Report every metric per condition, never pooled. One number hides the effect: inside one published grid, losses ran from -0.4 to +53.5 points.

- 5. Benign-edit control. Feed denoised and restored real speech. Count false alarms. A binary system fails here by design.

## Three protocols that are not standard, and matter

- Threshold transfer. Set the threshold on clean development data. Apply it unchanged to degraded test data. Report how far actDCF moves. Few papers report this. It is the clearest way to show your claim.

- Shortcut control. Trim leading and trailing silence, and control duration. Add a silence- only classifier as a baseline. Silence duration alone predicts the label at up to 85% accuracy, so any gain that survives trimming is real.

- Evidence faithfulness. Use partial-spoof data with ground-truth regions. Check whether the evidence directions overlap the manipulated region, using segment-level IoU matching. High evidence with no overlap means the explanation is false.

## Required baselines

- A globally calibrated version of your own model. This is the direct contrast.

- Temperature scaling. A reviewer will ask whether the simple method is enough.

- Empirical risk minimization. Theory says it is hard to beat in domain generalization, so expect it to be strong.

- One published detector, such as AASIST, run under your protocol. Its ASVspoof 5 numbers are known, so it anchors your setup against the literature.

- A recent generalization method, such as meta-learned LoRA or a mixture-of-experts system. Section 6 names these as the EER competition, so you must measure against one.

- An evidential or Dirichlet uncertainty method. It is the closest existing answer to abstention.

## 11. Experiment plan

Decision. We implement all three phases. Phase 0 below runs first, because it can stop the project cheaply.


## Phase 0. Falsify the premise. Week 1 to 3

The whole project rests on one assumption: that the calibration gap under shift is large, and that a simple fix does not close it. Test the assumption before you build anything.

## The steps.

- 1. Take AASIST, pretrained. Do not train anything.

- 2. Build the condition grid with ffmpeg.

- 3. Run it on IndicSynth and In-the-Wild.

- 4. Set the threshold on clean development data. Apply it unchanged to every condition.

- 5. Measure how far actDCF moves from minDCF, per condition and per language.

- 6. The lookup-table test. Fit one calibration per condition and store it in a table. Measure how much of the gap it closes.

## The two outcomes.

| Result | What it means | What to do |
| --- | --- | --- |
| The gap is large, and the | The premise holds | Proceed to phase 1 |
| lookup table leaves much of it |   |   |
| The gap is small, or the | The premise fails, or a | Stop. Redesign before |
| lookup table closes it | hypernetwork is unjustified | spending a year |

Why the lookup table matters. It is the obvious objection a reviewer will raise. Detect the condition, look up the calibration, apply it. Fifty lines of code, no hypernetwork. It probably captures most of the gain on common conditions.

Design one experiment whose only purpose is to break it. Hold out whole codecs and whole languages, so the table has no cell to use. The hypernetwork earns its place in three cases only:

- Unseen conditions. The table must fall back. The hypernetwork interpolates in a continuous space.

- Mixed conditions. MP3 then EnCodec, with noise. The combinations explode, so the table needs a cell for each.

- Sparse cells. Low-resource languages give few calibration samples, so a per-cell fit is unstable. The hypernetwork shares strength across conditions.

If the hypernetwork does not win there, it wins nowhere.


## Phase 1. Establish the problem

Show that a globally calibrated system reports confident, wrong LLRs under codec and language shift. Use a published detector and your condition grid. Report actDCF, Cllr and the fixed-threshold error rate per language and per codec. Section 9 shows that nobody reports these under shift, so this phase creates the baselines the later phases need. It also stands alone as a short paper.

## Phase 2. Build blocks A and C

Add the condition encoder, FiLM conditioning and the calibration hypernetwork. Build the five generalization mechanisms of section 6 at the same time, because they carry the EER result. Compare against the same model with one global calibration. That contrast is the paper.

## Phase 3. Add blocks B and D

Add the sparse autoencoder and the competence gate. Test faithfulness and conformal coverage.

## The ablation table you must produce

A reviewer will ask which part gives the gain. Prepare this from the start.

| Configuration | EER actDCF Cllr Coverage |
| --- | --- |
| Backbone + global calibration (baseline) |   |
| + temperature scaling |   |
| + condition encoder and FiLM, global calibration |   |
| + condition-dependent calibration |   |
| + sparse autoencoder evidence |   |
| + competence gate (full system) |   |

Add one more row: the per-condition calibration lookup table from phase 0. It is the baseline that decides whether the hypernetwork is justified, so it belongs in the main ablation, not a footnote. Run every row on every protocol in section 10. Report bootstrap confidence intervals. A difference of 0.2% EER often means nothing.


## Measure the condition encoder itself

The condition estimator can fail out of domain. Report its accuracy on unseen codecs and unseen languages. If it fails, the whole system fails quietly, which is worse than failing loudly.

## Timeline

| Phase Duration |   | Output |
| --- | --- | --- |
| 0 | 3 weeks | A go or no-go decision |
| 1 | 2 to 3 months The measurement paper, and the baselines every |   |
|   |   | later phase needs |
| 2 | 5 to 7 months The method paper. The main result. |   |
| 3 | 4 to 6 months The evidence paper. Block B stays optional. |   |

Phase 2 is the long one, because the five mechanisms of section 6 need building and ablating beside the calibration.

## Engineering rules

These prevent the quiet failures that waste months.

- Freeze the condition grid in phase 0. Never change it afterwards. If you change it, your phase 1 baselines stop being comparable, and they are your only reference point.

- Build the evaluation code once, and correctly. actDCF, Cllr, minDCF, miss rate at a fixed false alarm rate, and bootstrap intervals. An error here corrupts all three phases silently.

- Separate train, development and test by speaker, sentence and generator. Leakage is the most common quiet failure in this area.

- Version and release from day one. Retrofitting reproducibility is painful.

- Recheck prior art before each submission. This area moves in weeks.

- Verify every figure in this document against its primary source before you build on it. Many come from 2026 preprints without peer review.


## 12. Risks and limitations

| Risk | How it shows | What to do |
| --- | --- | --- |
| A reviewer calls this an | Rejection on novelty | Defend with a result, not a |
| engineering combination |   | diagram. Show the wrong, |
|   |   | confident LLRs, then show |
|   |   | yours. |
| The condition encoder fails | The system is quietly | Measure it separately, and |
| out of domain | wrong | report its accuracy on |
|   |   | unseen conditions. |
| Abstention becomes an | The system abstains often | Report the abstention rate. A |
| excuse | and claims success | system that abstains on half |
|   |   | the data has not solved the |
|   |   | problem. |
| Prior art appears while you | Your claim weakens | Recheck before submission. |
| work |   | Several cited papers are |
|   |   | weeks old. |
| The sparse autoencoder | Block B gives nothing | Treat block B as optional. The |
| directions are not |   | paper survives on A and C. |
| interpretable |   |   |
| Dataset licences and consent Blocked publication |   | Check terms before you use |
|   |   | any speech, especially of |
|   |   | public figures. |

## Honest limits of the claim.

- Calibration does not improve threshold-free discrimination. EER and AUC may stay flat. It improves the error rate at a fixed operating point, which is the accuracy a deployment feels. Say both parts plainly.

- The guarantee from conformal prediction holds under assumptions that heavy domain shift can break. State the assumption.

- Some figures in this document come from preprints that are not peer reviewed. Verify each one before you cite it.

- The example in section 6 uses illustrative numbers. Replace them with measured ones before you submit.


## 13. Publication plan

Publish in three steps. Each step stands alone.

| Paper | Scope | Claim | Venue |
| --- | --- | --- | --- |
| 1. The | Published detectors | Calibration, not | TMLR first choice: |
| measurement | on your condition grid, | discrimination, breaks | free, correctness- |
|   | across Indian and | in the wild | focused, no page |
|   | other languages |   | limit. IEEE Access as |
|   |   |   | the paid alternative. |
|   |   |   | Avoid ICASSP here, |
|   |   |   | because four pages |
|   |   |   | cannot hold the per- |
|   |   |   | condition tables. |
| 2. The method | Blocks A and C, the | Condition-dependent | ICASSP or |
|   | five generalization | calibration fixes it | Interspeech main |
|   | mechanisms, and the |   | track |
|   | full ablation |   |   |
| 3. The evidence | Blocks B and D, with | Evidence a forensic | TASLP, Computer |
| framework | faithfulness and | analyst can use | Speech and |
|   | forensic validation |   | Language, or |
|   |   |   | Forensic Science |
|   |   |   | International |

Order matters. Paper 1 needs no new model and little compute. It also builds the evidence that motivates paper 2. Start there.

## Practices that raise acceptance.

- Post each paper on arXiv first.

- Release the code and the condition grid scripts.

- Report cross-dataset results, never in-domain only.

- Name the limitations yourself, before a reviewer does.

## Sources

The figures in this document come from the papers and benchmarks named in sections 1, 2, 5, 6, 8 and 9. Several are 2026 preprints without peer review. Verify each figure against the primary paper before you cite it in a submission.
