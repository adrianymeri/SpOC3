# Internal review report — thesis documentation audit (2026-07-12)

*Scope: THESIS.md / Thesis_EN.md (§1–§15.8, references), Thesis_AL.md,
PLANTED_STRUCTURE_CERTIFICATES.md, RESULTS.md, README.md, RESEARCH_PROPOSALS.md,
figures, campaign data provenance. Standard applied: would each claim survive
a hostile referee with access to the repository. Each finding lists its
disposition; unresolved items are the pre-submission checklist.*

## A. Findings resolved during this review

**A1 (major — data provenance).** fig15_certificates.png and §15.5's unlock
analysis were originally computed on the 11-July pool, i.e. BEFORE the
validity purge of §15.8; several contributing achievers were subsequently
voided. *Disposition:* figure regenerated from the post-sanitisation valid
envelope (12 July); §15.5 re-measured on valid orderings (the 82→99 unlock
reproduces: five whole components, 480/525 moved vertices) with the finer
11-July figures explicitly downgraded to "indicative". Critically, all six
certificate points re-verified on the clean pool: bound met exactly at
w ∈ {299, 332, 365, 399, 449, 499} by valid orderings — §15.2–15.3 unaffected.

**A2 (major — reporting discipline).** Campaign numbers were quoted at
different intra-day snapshots across documents. *Disposition:* §15 header now
states the reporting convention (every score is a dated, verifier-checked
snapshot; `campaign_scores.csv` + re-verification supersede in-text values;
certificates are timeless, magnitudes are not). RESULTS.md banner updated to
the latest verified snapshot with the same convention.

**A3 (moderate).** §15.4's lever magnitudes (+6,311 / +18,259) were measured
pre-sanitisation. *Disposition:* provenance sentence added; direction
re-confirmed on the valid pool; §15.8 cross-referenced.

**A4 (moderate — completeness).** The 12-July methods (CQS, bandit portfolio,
family-transfer negative) existed only in the companion doc. *Disposition:*
new §15.7b added with first-day measurements; references extended (Parter
1961; Kuhn et al. 2016 for exact 2-D HSSP; Auer et al. 2002 for UCB1).

**A5 (minor).** Companion doc (PLANTED_STRUCTURE_CERTIFICATES.md) mixes formal
content with campaign-log register ("kill/retarget", ops notes). *Disposition:*
acceptable as the declared lab record — the formal statement lives in §15;
no change beyond the correction/appendix structure already present (§6b–6f).

## B. Verified sound (no action)

- §15.2 theorem and proof: correct as stated (monotone fill; first
  torso-eliminated clique member argument; disjointness for summation).
  Anchored to Parter/Rose elimination-game definitions of §1.
- §15.8 incident report: dates, counts (6/20 capped; 1,742/2,576 dropped) match
  tool outputs preserved in logs; the claim that certified points were
  CPU-produced and valid re-verified in A1.
- References: competition (GECCO companion DOI), both personal communications
  with permission noted, Biedl et al. 2005, treewidth canon, GBDT canon
  (LightGBM/XGBoost), LNS (Shaw; Pisinger & Ropke), hypervolume (Zitzler et
  al.), now HSSP + UCB1 + Parter. No orphaned citations found.
- Ablation tables (§6b.2, §10.2, rank_ablation_v2.txt) trace to reproducible
  commands; negative results (beam decode §STRATEGIC, ranker §15.6, five
  constructions §15.6-companion, transfer §15.7b) are reported with the same
  prominence as positives — referee-proof posture.

## C. Pre-submission checklist (open, owner: author)

1. **Rebuild PDFs** (docs/build pandoc pipeline) — all three PDFs predate §15.
2. **Freeze pass**: at thesis freeze, re-run the score block once, replace
   every "at time of writing" number from the ledger, regenerate fig15 and the
   gap-vs-time figure from campaign_scores.csv (script:
   plot from ledger; 2 lines of matplotlib), and re-run
   `verify_submission.py` on all three cap20 files that will be cited.
3. **Thesis_AL.md**: §15 summary present; propagate §15.5/15.7b updates and the
   final frozen numbers at freeze (translation pass).
4. **Leaderboard re-check** against the live ESA board immediately before any
   external claim (targets drift; noted in Final-results section).
5. **Bannach follow-up**: send PLANTED_STRUCTURE_CERTIFICATES.md as promised;
   any reply about component treewidths strengthens §15.1's completeness claim.
6. Optional strengthening: medium-quotient per-width lower bounds (§7 of the
   companion doc) remain the one analytical gap between "characterised" claims
   on large vs medium.
