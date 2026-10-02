# Corpus plan (Phase 0 deliverable)

**Verified on 2026-10-02.** Every claim below comes from an actual request made that day,
with the HTTP status recorded. Where something could not be verified, it says so.

* Machine-readable list: [`data/corpus_plan.yaml`](../data/corpus_plan.yaml) (102 entries)
* Validated manifest: [`data/manifest.csv`](../data/manifest.csv) (95 queued, 7 excluded
  entry points) — `make validate-manifest` reports 0 errors, 0 warnings

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
| `https://rbidocs.rbi.org.in/.../MD*.PDF` | Returns **`text/html`** (an interstitial, ~45 KB) with a plain GET. With a `Referer` header it returns **HTTP 200 `application/pdf`** — verified with a 2,049,883-byte KYC PDF whose header is `%PDF-1.6`. |

**Consequence.** RBI documents are queued as direct PDF links (the `rbidocs` URLs), and the
fetcher sends a per-host `Referer` (`FetchPolicy.referer_hosts`). The three RBI listing pages
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
| RBI master directions | 17 | KYC (banks + NBFC), IRAC 2025, ALM 2025 (LCR/FALLCR), investment portfolio, IT outsourcing, IT governance, digital lending, PSL, NBFC P2P, PPI, securitisation, credit/debit cards, account aggregator, risk management & inter-bank dealings, wilful defaulters, credit information |
| RBI notification / master circular | 3 | Basel III LCR notification, Integrated Ombudsman Scheme, IRAC master circular (kept so the supersession graph has a real edge) |
| SEBI regulations | 5 | LODR, NCS, ICDR, CRA, Debenture Trustees |
| SEBI master circulars | 7 | LODR compliance, NCS issue/listing, NCS disclosure, ICDR, CRA, Debenture Trustees, plus one operational circular |
| SEBI circulars | 4 | Cyber incident reporting (FIRE), KRA information sharing, OBPP framework, position limits |
| Annual reports | 20 | 10 institutions × FY2024, FY2025 |
| Earnings-call transcripts | 40 | 10 institutions × 4 quarters of FY2025 |
| Excluded entry points | 7 | 3 RBI + 4 SEBI listing pages, for the refresh job only |
| **Total rows** | **102** | 95 queued for download |

**Corpus target.** 95 documents for the MVP (PRD asks for 100; the gap is 5 documents and is
expected to close with the optional institutions above). Scale-up to 300–500 is a Phase 5
concern and would use the same plan file.

---

## 4. Honest gaps

1. **Three RBI titles are unconfirmed.** `rbi_md_kyc_nbfc_2016`, `rbi_md_credit_information_amendment`
   and `rbi_md_digital_lending_2025` have titles taken from third-party citations plus the
   mobile mirror's body text, because RBI's own pages do not expose a machine-readable title.
   Each is flagged in the manifest `notes` with "confirm in Phase 1", where the PDF itself is
   the authority.
2. **RBI `issue_date` is empty for all rows.** The manifest validator warns until Phase 1
   extracts dates from the PDF cover pages. Fabricating dates now would poison the Phase 3
   temporal filter, which is exactly the failure mode the PRD warns about ("superseded rules
   cited as current").
3. **Transcript availability is not guaranteed.** Some institutions publish only an analyst
   presentation or a results press release. The manifest note for each row says what to
   substitute; nothing is assumed to exist.
4. **No document has been downloaded yet.** Nothing in this report claims otherwise: the
   hashes are empty until `make download -- --yes --accept-terms` runs.
5. **Bajaj Finance and Bank of Baroda are absent** because their IR paths 404'd during
   verification. Adding them needs the correct URL, found by hand.
