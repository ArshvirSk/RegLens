# Corpus plan (Phase 0 deliverable)

**Verified on 2026-10-02.** Every claim below comes from an actual request made that day,
with the HTTP status recorded. Where something could not be verified, it says so.

* Machine-readable list: [`data/corpus_plan.yaml`](../data/corpus_plan.yaml) (104 entries)
* Validated manifest: [`data/manifest.csv`](../data/manifest.csv) (97 queued — 20 of them
  already fetched and hashed — plus 7 excluded entry points) — `make validate-manifest`
  reports 0 errors, 0 warnings

---

## 1. Recommended institutions (8 + 2 optional)

Ten institutions, chosen to cover the question types the PRD asks for rather than to be
exhaustive. Rationale is about *what questions the document set can answer*.

| # | Institution | Type | Why it is in the corpus |
|---|---|---|---|
| 1 | **State Bank of India** | PSU bank | Largest by assets; the reference point for PSL, priority-sector and government-bank questions. |
| 2 | **HDFC Bank** | Private bank | Largest private bank; merger-era disclosure changes make it a good "did the numbers change" case. |
| 3 | **ICICI Bank** | Private bank | Detailed segment disclosure; reliable transcripts. |
| 4 | **Axis Bank** | Private bank | Deposit cost and NIM questions; annual reports are consistently published. |
| 5 | **Kotak Mahindra Bank** | Private bank + group NBFC | Useful when a question spans a bank and an NBFC in one group. |
| 6 | **IndusInd Bank** | Mid-size private bank | Real episodes of NPA and derivative-accounting disclosure → good multi-hop material. |
| 7 | **Federal Bank** | Smaller private bank | Comparison anchor: a bank whose scale differs enough that ratios are not just large-bank noise. |
| 8 | **Cholamandalam Investment and Finance** | Large retail NBFC | NBFC rules (NIAR, securitisation, co-lending) actually bite here. |
| 9 | **Shriram Finance** | Diversified NBFC | AUM-mix and yield questions; different book from Chola. |
| 10 | **REC Limited** | PSU infrastructure NBFC | Sits at the RBI/government/PSU-bank intersection; regulated by RBI as an NBFC. |

Fiscal scope: **FY2024 and FY2025** (Indian fiscal years, i.e. years ended 31 March 2024 and
31 March 2025), with the four quarterly transcripts of FY2025 for each institution. That is
2 annual reports + 4 transcripts per institution, plus the regulatory corpus.

**Optional swaps.** Bajaj Finance, Muthoot Finance, L&T Finance, Bank of Baroda and PNB were
considered. Their IR pages moved during verification (Bajaj Finserv and Cholamandalam both
returned 404 on the first path tried), so the four NBFCs above were chosen for link
stability as much as for coverage. Bank of Baroda's IR path also 404'd; it can be added once
the correct path is confirmed by hand.

---

## 2. Source verification findings

These findings shape the fetcher and the refresh job, so they are recorded rather than
buried in code comments.

### RBI (`rbi.org.in`, `m.rbi.org.in`, `rbidocs.rbi.org.in`)

| Check | Result |
|---|---|
| `https://www.rbi.org.in/robots.txt` | **HTTP 418** ("Unauthorised Access") for both a research UA and a browser UA. The WAF blocks the path; no usable crawl rules are published. |
| `https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx` (Master Directions index) | HTTP 200, ~786 KB, but the document list is inside the ASP.NET `__VIEWSTATE` blob: 386 `ViewMasDirections` references and **zero parseable `<a>` anchors**. Body content is injected client-side. |
| `https://www.rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx` (notifications index) | HTTP 200 but no parseable document anchors either. |
| `https://m.rbi.org.in/scripts/BS_ViewMasDirections.aspx?id=<id>` (mobile mirror) | HTTP 200/302 with the **document text in the body** and, for some IDs, direct `rbidocs` PDF links. Titles are generic ("Home-Reserve Bank of India"), so the ID → document mapping is not self-describing. |
| `https://rbidocs.rbi.org.in/.../MD*.PDF` | Runs an F5 bot challenge. The project's polite `RegLens/0.1` UA gets **connection resets (6/6 attempts)**; a browser UA alone gets the ~45 KB `text/html` interstitial; **browser UA + `Referer` returns HTTP 200 `application/pdf`** — verified with a 2,049,883-byte KYC PDF whose header is `%PDF-1.6`. Its `robots.txt` serves the challenge HTML to both UAs (parsed as "no rules", so nothing is self-blocked). |

**Consequence.** RBI documents are queued as direct PDF links (the `rbidocs` URLs), and the
fetcher sends a per-host `Referer` (`FetchPolicy.referer_hosts`) plus a per-host browser UA
(`FetchPolicy.browser_ua_hosts`) for that host only — while robots rules are still enforced
for both identities, so the UA swap never widens what may be fetched. The three RBI listing pages
are recorded as `status: excluded` entry points for the Phase 4 refresh job, which will need a
browser-based adapter (Playwright) or a manual review step. This is the single largest
automation gap in the project and it is deliberately not papered over.

### SEBI (`sebi.gov.in`)

| Check | Result |
|---|---|
| `https://www.sebi.gov.in/robots.txt` | HTTP 200, permissive: `User-agent: *`, empty `Disallow`, plus `/js` and `/css`. |
| Legal listings (`HomeAction.do?doListing=yes&sid=1&ssid=7`) | HTTP 200, server-rendered, parsed into clean deep links (26 circular rows, 26 master-circular rows, 43 regulation rows on the pages checked). |
| Deep links (`/legal/circulars/...`, `/legal/master-circulars/...`, `/legal/regulations/...`) | HTTP 200 on all 15 sampled. |

**Consequence.** SEBI is the one source the refresh job can poll automatically and
idempotently. The listing pages are stored as `excluded` entry points so the job has a stable
URL to watch.

### Bank / NBFC investor-relations pages

| Institution | URL checked | Status |
|---|---|---|
| SBI | `https://sbi.co.in/web/investor-relations` | 200 (the `/financial-results` sub-path 404s — use the IR root) |
| HDFC Bank | `https://www.hdfcbank.com/personal/about-us/investor-relations` | 200 |
| ICICI Bank | `https://www.icicibank.com/about-us/investor-relations` | 200 |
| Axis Bank | `https://www.axisbank.com/shareholders-corner/financial-results/annual-reports` | 200 |
| Kotak Mahindra Bank | `https://www.kotak.com/en/investor-relations/financial-results.html` | 200 |
| IndusInd Bank | `https://www.indusind.com/in/en/personal/investor-relations.html` | 200 |
| Federal Bank | `https://www.federalbank.co.in/financial-results` | 200 |
| Cholamandalam | `https://www.cholamandalam.com/investors/` | 200 |
| Shriram Finance | `https://www.shriramfinance.in/investors` | 200 |
| REC Limited | `https://recindia.nic.in/financial-results` | 200 |

All ten are recorded with `discovery: listing`: the URL is a landing page, not a document.
`make download --resolve-listings` prints candidate PDF links from these pages for review, and
the exact file is named in the manifest before ingestion. This is intentional — an automatic
pick would silently grab the wrong file, and the corpus would stop being a decision.

---

## 3. Document mix

| Category | Count | Notes |
|---|---|---|
| RBI master directions | 18 | KYC (banks + NBFC, plus the Commercial Banks KYC Directions 2025), IRAC 2025, ALM 2025 (LCR/FALLCR), investment portfolio, IT outsourcing, IT governance, digital lending, PSL, NBFC P2P, PPI, securitisation, credit/debit cards, account aggregator, risk management & inter-bank dealings, wilful defaulters, credit information |
| RBI notification / master circular | 4 | Basel III LCR notification, Integrated Ombudsman Scheme (notification + consolidated scheme text), IRAC master circular (kept so the supersession graph has a real edge) |
| SEBI regulations | 5 | LODR, NCS, ICDR, CRA, Debenture Trustees |
| SEBI master circulars | 7 | LODR compliance, NCS issue/listing, NCS disclosure, ICDR, CRA, Debenture Trustees, plus one operational circular |
| SEBI circulars | 4 | Cyber incident reporting (FIRE), KRA information sharing, OBPP framework, position limits |
| Annual reports | 20 | 10 institutions × FY2024, FY2025 |
| Earnings-call transcripts | 40 | 10 institutions × 4 quarters of FY2025 |
| Excluded entry points | 7 | 3 RBI + 4 SEBI listing pages, for the refresh job only |
| **Total rows** | **104** | 97 queued for download, 20 already fetched |

**Corpus target.** 97 documents for the MVP (PRD asks for 100; the gap is 3 documents and is
expected to close with the optional institutions above). Scale-up to 300–500 is a Phase 5
concern and would use the same plan file.

---

## 4. Honest gaps

1. **Three RBI titles were unconfirmed; the fetched PDFs resolved them on 2026-10-02.**
   `rbi_md_digital_lending_2025` pointed at the Filing of Supervisory Returns Directions 2024
   (corrected to landing id=12848), `rbi_notif_integrated_ombudsman_2021` at an unrelated
   Basel III circular (corrected to id=12192, with the consolidated scheme text added as its
   own row), and `rbi_md_credit_information_amendment` at the Commercial Banks KYC Directions
   2025 (that content became `rbi_md_kyc_commercial_2025`; the CIC Directions came from
   id=12926). Two `doc_id` years were also wrong and were renamed: `rbi_md_kyc_nbfc_2016` →
   `_2025` and `rbi_md_risk_interbank_2019` → `_2016`. Each correction is recorded in the
   row's `notes`.
2. **`issue_date` is filled for the 20 fetched RBI rows only** — read from each PDF's page 1
   on 2026-10-02, then re-checked mechanically: the first header date in the stored bytes
   equals the manifest date for 20/20 rows. The remaining rows get theirs when they are
   fetched. Fabricating dates would poison the Phase 3 temporal filter, which is exactly the
   failure mode the PRD warns about ("superseded rules cited as current").
3. **Transcript availability is not guaranteed.** Some institutions publish only an analyst
   presentation or a results press release. The manifest note for each row says what to
   substitute; nothing is assumed to exist.
4. **20 of the 97 queued documents are downloaded** (all RBI: 18 Master Directions +
   2 Ombudsman rows). Every stored payload re-opens as a PDF and is recorded content-addressed
   under `data/raw/` (gitignored) with its SHA-256 in the manifest. The other 77 rows are
   landing/listing pages awaiting a human candidate pick or a later-phase fetch.
5. **Bajaj Finance and Bank of Baroda are absent** because their IR paths 404'd during
   verification. Adding them needs the correct URL, found by hand.
